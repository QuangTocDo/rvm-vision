import os
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import threading
import queue
import torch
import math
from collections import deque, Counter
import logging
from logging.handlers import RotatingFileHandler

import config

# ADD GLOBAL ROOT PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tracker import Tracker
from utils.camera_utils import FindCamera, open_camera
from utils.helper_functions import crop_from_box
from utils.hierarchy import evaluate_hierarchical_class, estimate_volume
from utils.hud import draw_premium_hud, draw_predictions
from workers.inference import InferenceWorker
from workers.classifier import ClassifierWorker

def setup_logger():
    logger = logging.getLogger("RVM_Vision")
    logger.setLevel(logging.INFO)
    
    # Thiết lập RotatingFileHandler: tối đa 5MB/file, giữ lại tối đa 5 file backups
    log_file = "rvm_vision_system.log"
    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Đồng thời xuất ra màn hình console (tiện debug dev mode)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

HOME = Path.home()
mac_add = "rvm"
__env = HOME / ".env"
__camera_env = HOME / ".camera.env"
CODE = "8.4.0_v2"
CAMERAS = (-1, 0, 0, 640, 480)

detext = False
beginTime = 0

def average(lst):
    if lst is None or len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

__dict = {0: 0, 1:1 , 2: 2, 3:2, 4:2, 5:2}

def transform_id(id_detect):
    global __dict
    if __dict is None:
        return id_detect
    return __dict.get(str(id_detect), id_detect)

def calc_avg(calc_ids):
    if calc_ids is None or len(calc_ids) == 0:
        return -1
    counts = Counter(calc_ids)
    most_common_class, count = counts.most_common(1)[0]
    return most_common_class

def global_emit(event, data):
    logger.info(f"[EMIT MOCK] event: {event} | data: {data}")

class RealtimeDevPipeline:
    def __init__(self, camera_idx):
        global CAMERAS
        self.camera_idx = camera_idx
        self.cap = open_camera(camera_idx)
        
        # Cờ báo hiệu đang kết nối lại camera ngầm và Khóa đồng bộ luồng
        self.camera_reconnecting = False
        self.cache_lock = threading.Lock()
        
        self.in_queue = queue.Queue(maxsize=2)
        self.out_queue = queue.Queue(maxsize=2)
        
        self.worker = InferenceWorker(self.in_queue, self.out_queue)
        self.worker.start()
        
        # Khởi tạo Queues và Thread phụ cho Classifier để chạy bất đồng bộ
        self.cls_in_queue = queue.Queue(maxsize=4)
        self.cls_out_queue = queue.Queue(maxsize=4)
        self.cls_worker = ClassifierWorker(self.cls_in_queue, self.cls_out_queue)
        self.cls_worker.start()
        
        self.colors = [(255, 0, 0), (0, 128, 0), (0, 0, 200), (120, 120, 120), (180, 140, 180)]
        self.__path = config.YOLO_DET_MODEL_PATH
        
        # Lấy tọa độ ROI từ config.py
        self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2 = config.ROI_COORDS
        
        # Cache cho thể tích và tuổi của track
        self.volume_cache = {}
        self.track_age_vol = {}
        
        # Cache cho classification
        self.track_cache = {}
        self.track_age_cls = {}
        
        # Lịch sử tâm cx của từng ID bám vết
        self.track_history = {}
        self.triggered_ids = set()
        
        # Cửa sổ trượt lưu lịch sử phát hiện trong ROI của từng ID (Sliding Window Filter)
        self.roi_sliding_windows = {}
        self.SLIDING_WINDOW_SIZE = config.SLIDING_WINDOW_ROI

        # Biến đo và làm mượt FPS
        self.prev_loop_time = time.time()
        self.smoothed_fps = 0.0

    def reconnect_camera_async(self):
        """Khởi chạy luồng ngầm kết nối lại camera bất đồng bộ"""
        if self.camera_reconnecting:
            return
            
        def target():
            self.camera_reconnecting = True
            logger.info("Đang tìm kết nối lại camera ở luồng ngầm...")
            camera_ii = -1
            while camera_ii < 0:
                camera_ii = FindCamera()
                if camera_ii >= 0:
                    self.cap = open_camera(camera_ii)
                    logger.info(f"Kết nối lại thành công camera tại index: {camera_ii}")
                    break
                time.sleep(2.0)
            self.camera_reconnecting = False

        threading.Thread(target=target, daemon=True).start()

    def run(self):
        global detext, beginTime, CAMERAS
        logger.info("--- FULLY AUTOMATIC DETECTION MODE (HIERARCHICAL LOGIC V2 + PYTORCH GLASS CNN) ---")
        logger.info("Press 'q' to quit.")

        id = -1
        frameCount = 0
        calc_ids = []
        sizes = []
        __error_times__ = 0
        
        detection_armed = True # Bắt đầu ở trạng thái sẵn sàng

        while True:
            ret, frame_curr = self.cap.read()
            if not ret:
                # Kích hoạt kết nối lại bất đồng bộ
                self.reconnect_camera_async()
                
                # Hiển thị màn hình chờ kết nối mà không bị đóng băng UI
                placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_frame, "CAMERA DISCONNECTED! RECONNECTING...", 
                            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("RVM DEV MODE (AUTO)", placeholder_frame)
                cv2.waitKey(100)
                continue
            __error_times__ = 0
            
            next_start_time = time.time()
            
            try:
                processed_frame, valid_boxes, start_time = self.out_queue.get(timeout=0.01)
                
                # --- TÍNH TOÁN FPS REALTIME & LÀM MƯỢT BẰNG EMA ---
                current_time = time.time()
                time_diff = current_time - self.prev_loop_time
                self.prev_loop_time = current_time
                if time_diff > 0:
                    instant_fps = 1.0 / time_diff
                    alpha = config.FPS_ALPHA_GUI  # Hệ số làm mượt để tránh nhấp nháy liên tục
                    self.smoothed_fps = alpha * instant_fps + (1.0 - alpha) * self.smoothed_fps
                
                # --- ĐỌC KẾT QUẢ PHÂN LOẠI TỪ LUỒNG PHỤ PHÂN LOẠI ---
                while not self.cls_out_queue.empty():
                    try:
                        res_id, res_class, res_score = self.cls_out_queue.get_nowait()
                        with self.cache_lock:
                            self.track_cache[res_id] = (res_class, res_score)
                    except queue.Empty:
                        break
                
                # --- TÍNH TOÁN VÀ ĐỒNG BỘ AI CHO TẤT CẢ VẬT THỂ ĐANG ĐƯỢC BÁM VẾT ---
                active_ids = {box[3] for box in valid_boxes}
                with self.cache_lock:
                    for cache_dict in (self.track_cache, self.track_age_cls, self.volume_cache, self.track_age_vol, self.roi_sliding_windows, self.track_history):
                        inactive_keys = cache_dict.keys() - active_ids
                        for k in inactive_keys:
                            if k in cache_dict:
                                del cache_dict[k]
                    self.triggered_ids &= active_ids
                
                # Gửi yêu cầu dọn dẹp sang ClassifierWorker
                if not self.cls_in_queue.full():
                    self.cls_in_queue.put(("cleanup", active_ids))
                
                # Bộ ghi nhận xem các ID có xuất hiện trong ROI ở frame này hay không
                detected_in_roi_this_frame = { _id: False for _id in active_ids }

                for box in valid_boxes:
                    bbox, idx_class, conf, _id = box
                    if idx_class not in [0, 1, 2, 4]:
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    if self.roi_x1 < cx < self.roi_x2 and self.roi_y1 < cy < self.roi_y2:
                        detected_in_roi_this_frame[_id] = True
                        
                        # 1. Đo thể tích vật thể
                        with self.cache_lock:
                            self.track_age_vol[_id] = self.track_age_vol.get(_id, 0) + 1
                            is_vol_calc_frame = (self.track_age_vol[_id] == 1 or self.track_age_vol[_id] % config.CLASSIFY_INTERVAL == 0)
                            
                        if is_vol_calc_frame:
                            w_box, h_box = x2 - x1, y2 - y1
                            length_px, diameter_px = max(w_box, h_box), min(w_box, h_box)
                            if length_px < 300:
                                vol = estimate_volume(length_px, diameter_px, config.PIXEL_TO_CM_RATIO, config.VOLUME_SCALING_UP)
                            elif length_px > 700:
                                vol = estimate_volume(length_px, diameter_px, config.PIXEL_TO_CM_RATIO, config.VOLUME_SCALING_DOWN)
                            else:
                                vol = estimate_volume(length_px, diameter_px, config.PIXEL_TO_CM_RATIO, 1)
                            
                            with self.cache_lock:
                                self.volume_cache[_id] = vol
                                
                        # 2. Nhận diện thương hiệu (chỉ khi thể tích hợp lệ và idx_class là 0, 1, hoặc 2)
                        with self.cache_lock:
                            current_vol = self.volume_cache.get(_id, -1)
                        if config.MIN_ACCEPTABLE_VOLUME < current_vol < config.MAX_ACCEPTABLE_VOLUME and idx_class in [0, 1, 2]:
                            with self.cache_lock:
                                self.track_age_cls[_id] = self.track_age_cls.get(_id, 0) + 1
                                is_cls_calc_frame = (self.track_age_cls[_id] == 1 or self.track_age_cls[_id] % config.CLASSIFY_INTERVAL == 0)
                                
                            if is_cls_calc_frame:
                                crop = crop_from_box(processed_frame, bbox)
                                if crop is not None and crop.size > 0:
                                    if not self.cls_in_queue.full():
                                        # Định tuyến phân loại theo loại database (Standard vs Special)
                                        db_type = "special" if idx_class == 2 else "standard"
                                        self.cls_in_queue.put(("classify", _id, crop, db_type))
                
                # Cập nhật Sliding Window cho tất cả các ID đang hoạt động
                with self.cache_lock:
                    for _id in active_ids:
                        if _id not in self.roi_sliding_windows:
                            self.roi_sliding_windows[_id] = deque(maxlen=self.SLIDING_WINDOW_SIZE)
                        is_detected = detected_in_roi_this_frame.get(_id, False)
                        self.roi_sliding_windows[_id].append(1 if is_detected else 0)

                # --- LỌC BỎ CÁC VẬT THỂ SAI KÍCH THƯỚC ĐỂ COI NHƯ KHÔNG CÓ VẬT THỂ ---
                decision_boxes = []
                with self.cache_lock:
                    for box in valid_boxes:
                        bbox, idx_class, conf, _id = box
                        if idx_class == 5:
                            continue  # Bỏ qua tay khi lọc chai
                        
                        # Nếu đang trong quá trình detect ID này, không lọc bỏ để tiếp tục thu thập sample đánh giá volume/class
                        if detext and _id == id:
                            decision_boxes.append(box)
                            continue
                            
                        vol = self.volume_cache.get(_id, -1)
                        if vol != -1 and not (config.MIN_ACCEPTABLE_VOLUME < vol < config.MAX_ACCEPTABLE_VOLUME):
                            continue
                        decision_boxes.append(box)

                len_decision_boxes = len(decision_boxes)

                is_object_in_roi = False
                if len_decision_boxes > 0:
                    for box in decision_boxes:
                        x1, y1, x2, y2 = box[0]
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        if self.roi_x1 < cx < self.roi_x2 and self.roi_y1 < cy < self.roi_y2:
                            is_object_in_roi = True
                            break

                # --- KIỂM TRA SỰ XUẤT HIỆN CỦA TAY TRONG VÙNG ROI ---
                is_hand_in_roi = False
                for box in valid_boxes:
                    bbox, idx_class, conf, _id = box
                    if idx_class == 5:
                        cx = (bbox[0] + bbox[2]) // 2
                        cy = (bbox[1] + bbox[3]) // 2
                        if self.roi_x1 < cx < self.roi_x2 and self.roi_y1 < cy < self.roi_y2:
                            is_hand_in_roi = True
                            break
                
                frameCount += 1
                display_img = processed_frame.copy()
                shape = processed_frame.shape
                ii = len(calc_ids)
                
                status_text = "WAITING"
                status_color = (0, 255, 255)
                if detection_armed:
                    status_text = "ARMED"
                    status_color = (0, 255, 0)
                if detext:
                    status_text = "DETECTING"
                    status_color = (0, 0, 255)
                
                # --- PREMIUM STACK STATISTICS HUD ---
                cls_status = "Idle" if self.cls_worker.fps < 0.1 else f"{int(self.cls_worker.fps)} FPS ({self.cls_worker.latency:.1f}ms)"
                draw_premium_hud(display_img, status_text, status_color, self.smoothed_fps, self.worker.fps, self.worker.latency, cls_status, is_hand_in_roi, (self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2))

                if len_decision_boxes > 0:
                    if len_decision_boxes > 1 and detext:
                        global_emit('command', 2)
                else:
                    global_emit('command', 0)

                # Vẽ bounding boxes, thương hiệu và thể tích (đồng bộ khóa bảo vệ cache)
                with self.cache_lock:
                    draw_predictions(display_img, valid_boxes, self.volume_cache, self.track_cache, self.colors)

                cv2.imshow("RVM DEV MODE (AUTO)", display_img)
                
                if detext and is_hand_in_roi:
                    logger.info("--- Hand detected in ROI! Aborting detection immediately! ---")
                    global_emit('result', {'data': -1, 'model': str(self.__path), 'ver': CODE,
                                           "id": mac_add, "images": "mock_images", "size": 0, "item": -1})
                    global_emit('command', 0)
                    detext = False
                    sizes = []
                    calc_ids = []
                    id = -1
                    detection_armed = True
                    logger.info("\n--- System re-armed due to Hand safety trigger. ---")

                # --- CẬP NHẬT QUỸ ĐẠO VÀ KIỂM TRA VƯỢT VẠCH ẢO ---
                virtual_line_y = self.roi_y2 - (self.roi_y2 - self.roi_y1) // 3
                trigger_this_frame = False
                triggered_id = -1
                is_trigger_invalid_volume = False
                invalid_volume_val = -1

                with self.cache_lock:
                    for box in valid_boxes:  # Duyệt trên valid_boxes thay vì decision_boxes
                        bbox, idx_class, conf, _id = box
                        if idx_class == 5:
                            continue  # Bỏ qua tay
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        cy = (y1 + y2) // 2
                        
                        if _id not in self.track_history:
                            self.track_history[_id] = deque(maxlen=5)
                        self.track_history[_id].append(cy)
                        
                        is_above_now = cy <= virtual_line_y
                        is_box_complete = y2 < self.roi_y2 + 15
                        
                        if is_above_now and is_box_complete and (_id not in self.triggered_ids):
                            vol = self.volume_cache.get(_id, -1)
                            if vol != -1:  # Đã có số đo thể tích
                                if not (config.MIN_ACCEPTABLE_VOLUME < vol < config.MAX_ACCEPTABLE_VOLUME):
                                    is_trigger_invalid_volume = True
                                    invalid_volume_val = vol
                                
                                trigger_this_frame = True
                                triggered_id = _id

                # --- TỰ ĐỘNG KÍCH HOẠT HOẶC TỪ CHỐI NGAY LẬP TỨC ---
                if detection_armed and trigger_this_frame and not detext and not is_hand_in_roi:
                    if is_trigger_invalid_volume:
                        logger.info(f"ARMED and object {triggered_id} crossed virtual line with INVALID volume ({invalid_volume_val:.1f}ml). Rejecting immediately!")
                        global_emit('result', {'data': 7, 'model': str(self.__path), 'ver': CODE,
                                               "id": mac_add, "images": "mock_images", "size": 0, "item": triggered_id,
                                               "volume": float(invalid_volume_val)})
                        with open("log.txt", "a", encoding="utf-8") as f:
                            f.write(f"{datetime.now()} | data=7 | id={mac_add} | images=[] | size=0 | item={triggered_id} | volume={invalid_volume_val:.1f} | (Rejected: Volume out of bounds)\n")
                        
                        self.triggered_ids.add(triggered_id)
                    else:
                        logger.info(f"--- Object {triggered_id} crossed virtual line {virtual_line_y}. Triggering detection! ---")
                        detext = True
                        beginTime = time.time()
                        detection_armed = False
                        id = triggered_id
                        self.triggered_ids.add(triggered_id)

                if detext:
                    if frameCount % 4 != 0 or frameCount < 4:
                        pass 
                    else:
                        endTime = time.time()

                        if ii >= config.MAX_DECISION_SAMPLES or endTime - beginTime > config.DETECTION_TIMEOUT:
                            avg = calc_avg(calc_ids)
                            logger.info(f"--- DETECTION FINISHED ---")
                            logger.info(f"Collected IDs: {calc_ids}")
                            logger.info(f"Final Average Result: {avg} (0=Aquafina, 1=Plastic, 2=Can, 7=Invalid)")
                            global_emit('result', {'data': avg, 'model': str(self.__path), 'ver': CODE,
                                        "id": mac_add, "images": "mock_images", "size": average(sizes), "item": id})
                            
                            detext = False
                            sizes = []
                            calc_ids = []
                            id = -1
                            detection_armed = True 
                            logger.info("\n--- System re-armed. Waiting for next object. ---")

                        else:
                            if len_decision_boxes > 0:
                                box = decision_boxes[0]
                                _id = box[3]
                                x1, y1, x2, y2 = box[0]
                                id = _id
                                sizes.append(max((y2 - y1) / shape[0] * 100, (x2 - x1) / shape[1] * 100))
                                
                                # --- HIERARCHICAL DECISION LOGIC V2 ---
                                with self.cache_lock:
                                    final_class, is_unknown_class2 = evaluate_hierarchical_class(box, self.volume_cache, self.track_cache, (self.roi_x1, self.roi_y1, self.roi_x2, self.roi_y2))
                                
                                if is_unknown_class2:
                                    logger.info("--- Unknown brand detected for class 2. Terminating detection immediately! ---")
                                    global_emit('result', {'data': 7, 'model': str(self.__path), 'ver': CODE,
                                                "id": mac_add, "images": "mock_images", "size": average(sizes) if sizes else 0, "item": id})
                                    global_emit('command', 0)
                                    detext = False
                                    sizes = []
                                    calc_ids = []
                                    id = -1
                                    detection_armed = True
                                    logger.info("\n--- System re-armed due to Class 2 Unknown brand. ---")
                                  
                                else:
                                    calc_ids.append(transform_id(final_class))
                            else:
                                global_emit('command', 0)
                            
                            logger.info(f"   [DETECTING] collecting... calc_ids: {calc_ids}")

            except queue.Empty:
                pass

            if not self.in_queue.full():
                self.in_queue.put((frame_curr, detext, next_start_time))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Exiting...")
                break

        self.worker.stop() 
        self.cls_worker.stop()
        
        while not self.in_queue.empty():
            try: self.in_queue.get_nowait()
            except queue.Empty: break
                
        try: self.in_queue.put(None, timeout=1)
        except queue.Full: pass

        while not self.cls_in_queue.empty():
            try: self.cls_in_queue.get_nowait()
            except queue.Empty: break
                
        try: self.cls_in_queue.put(None, timeout=1)
        except queue.Full: pass

        self.worker.join(timeout=2.0)
        self.cls_worker.join(timeout=2.0)
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_ii = CAMERAS[0]
    if camera_ii < 0:
        camera_ii = FindCamera()
        if camera_ii < 0:
            logger.error("Can not open camera")
            sys.exit(1)

    pipeline = RealtimeDevPipeline(camera_ii)
    pipeline.run()
