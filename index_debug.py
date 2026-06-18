import os
import sys
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import threading
import queue
import logging
from logging.handlers import RotatingFileHandler
from collections import deque

import config

# ADD GLOBAL ROOT PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.camera_utils import FindCamera, VideoCapture
from utils.helper_functions import crop_from_box
from utils.hierarchy import evaluate_hierarchical_class, estimate_volume
from utils.hud import draw_premium_hud, draw_predictions
from utils.roi import ROI
from utils.result_logger import ResultLogger
from core.state_machine import DetectionStateMachine
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
CODE = "8.5.0_debug"
CAMERAS = (-1, 0, 0, 640, 480)

# ── Đọc device ID ──────────────────────────────────────────────────────────
if os.path.exists(__env):
    with open(__env, 'rt') as file:
        mac_add = file.readline().strip()
elif os.path.exists(".env"):
    with open(".env", 'rt') as file:
        mac_add = file.readline().strip()
    with open(__env, 'wt') as file:
        file.write(mac_add)
    os.unlink(".env")

# ── Đọc cấu hình camera ────────────────────────────────────────────────────
def _parse_camera_env(path: str) -> tuple:
    with open(path, 'rt') as f:
        info = f.readline().split(",")
    return (int(info[0]), max(int(info[1]), 0), max(int(info[2]), 0),
            int(info[3]), int(info[4]))

def _write_camera_env(path: str, cam: tuple) -> None:
    with open(path, 'wt') as f:
        f.write(",".join(str(v) for v in cam))

if os.path.exists(__camera_env):
    CAMERAS = _parse_camera_env(str(__camera_env))
elif os.path.exists(".env.camera"):
    CAMERAS = _parse_camera_env(".env.camera")
    _write_camera_env(str(__camera_env), CAMERAS)
    os.unlink(".env.camera")
else:
    _write_camera_env(str(__camera_env), CAMERAS)

_transform_lock = threading.Lock()
_transform_dict = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2}

def transform_id(id_detect):
    with _transform_lock:
        d = _transform_dict
    if d is None:
        return id_detect
    return d.get(str(id_detect), id_detect)

def global_emit(event, data):
    logger.info(f"[EMIT MOCK] event: {event} | data: {data}")

class RealtimeDevPipeline:
    def __init__(self, camera_idx):
        self.camera_idx = camera_idx
        self.cap = VideoCapture(camera_idx)
        
        # Khóa đồng bộ luồng
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
        
        # Khởi tạo các tiện ích dùng chung
        self.roi = ROI.from_config(config.ROI_COORDS)
        self.sm = DetectionStateMachine()
        self.rlog = ResultLogger("results.jsonl")
        
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

    def run(self):
        logger.info("--- FULLY AUTOMATIC DETECTION MODE (HIERARCHICAL LOGIC V2 + PYTORCH GLASS CNN) ---")
        logger.info("Press 'q' to quit.")

        frameCount = 0
        
        # Các bí danh local để code chạy tương đồng với index.py
        volume_cache = self.volume_cache
        track_age_vol = self.track_age_vol
        track_cache = self.track_cache
        track_age_cls = self.track_age_cls
        track_history = self.track_history
        triggered_ids = self.triggered_ids
        roi_sliding_windows = self.roi_sliding_windows
        SLIDING_WINDOW_SIZE = self.SLIDING_WINDOW_SIZE
        cache_lock = self.cache_lock
        roi = self.roi
        sm = self.sm
        rlog = self.rlog

        # Khởi tạo trạng thái ARMED sẵn sàng
        sm.arm()

        while True:
            ret, frame_curr = self.cap.read()
            if not ret:
                # Hiển thị màn hình chờ kết nối mà không bị đóng băng UI
                placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_frame, "CAMERA DISCONNECTED! RECONNECTING...", 
                            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("RVM DEV MODE (AUTO)", placeholder_frame)
                cv2.waitKey(100)
                continue
            
            next_start_time = time.time()
            
            try:
                processed_frame, valid_boxes, start_time = self.out_queue.get(timeout=0.01)
                
                # --- TÍNH TOÁN FPS REALTIME & LÀM MƯỢT BẰNG EMA ---
                current_time = time.time()
                time_diff = current_time - self.prev_loop_time
                self.prev_loop_time = current_time
                if time_diff > 0:
                    instant_fps = 1.0 / time_diff
                    alpha = config.FPS_ALPHA_GUI
                    self.smoothed_fps = alpha * instant_fps + (1.0 - alpha) * self.smoothed_fps
                
                # --- ĐỌC KẾT QUẢ PHÂN LOẠI TỪ LUỒNG PHỤ PHÂN LOẠI ---
                while not self.cls_out_queue.empty():
                    try:
                        res_id, res_class, res_score = self.cls_out_queue.get_nowait()
                        with cache_lock:
                            track_cache[res_id] = (res_class, res_score)
                    except queue.Empty:
                        break
                
                # --- TÍNH TOÁN VÀ ĐỒNG BỘ AI CHO TẤT CẢ VẬT THỂ ĐANG ĐƯỢC BÁM VẾT ---
                active_ids = {box[3] for box in valid_boxes}
                with cache_lock:
                    for cache_dict in (track_cache, track_age_cls, volume_cache, track_age_vol, roi_sliding_windows, track_history):
                        for k in list(cache_dict.keys() - active_ids):
                            cache_dict.pop(k, None)
                    triggered_ids &= active_ids
                
                # Gửi yêu cầu dọn dẹp sang ClassifierWorker
                try:
                    self.cls_in_queue.put(("cleanup", active_ids), block=False)
                except queue.Full:
                    pass
                
                # Bộ ghi nhận xem các ID có xuất hiện trong ROI ở frame này hay không
                detected_in_roi_this_frame = { _id: False for _id in active_ids }
                classify_tasks = []

                with cache_lock:
                    for box in valid_boxes:
                        bbox, idx_class, conf, _id = box
                        if idx_class not in [config.YOLOClass.CAN, config.YOLOClass.PLASTIC, config.YOLOClass.GLASS, config.YOLOClass.PLASTIC_OTHER]:
                            continue
                        
                        if not roi.contains_center(bbox):
                            continue
                        
                        detected_in_roi_this_frame[_id] = True
                        
                        # 1. Đo thể tích vật thể
                        track_age_vol[_id] = track_age_vol.get(_id, 0) + 1
                        is_vol_calc_frame = (track_age_vol[_id] == 1 or track_age_vol[_id] % config.CLASSIFY_INTERVAL == 0)
                        
                        if is_vol_calc_frame:
                            x1, y1, x2, y2 = map(int, bbox)
                            w_box, h_box = x2 - x1, y2 - y1
                            length_px, diameter_px = max(w_box, h_box), min(w_box, h_box)
                            if length_px < 300:
                                k = config.VOLUME_SCALING_UP
                            elif length_px > 700:
                                k = config.VOLUME_SCALING_DOWN
                            else:
                                k = 1
                            volume_cache[_id] = estimate_volume(length_px, diameter_px, config.PIXEL_TO_CM_RATIO, k)
                                
                        # 2. Nhận diện thương hiệu (chỉ khi thể tích hợp lệ)
                        current_vol = volume_cache.get(_id, -1)
                        vol_valid = config.MIN_ACCEPTABLE_VOLUME < current_vol < config.MAX_ACCEPTABLE_VOLUME
                        if vol_valid and idx_class in [config.YOLOClass.CAN, config.YOLOClass.PLASTIC, config.YOLOClass.GLASS, config.YOLOClass.PLASTIC_OTHER]:
                            track_age_cls[_id] = track_age_cls.get(_id, 0) + 1
                            is_cls_calc_frame = (track_age_cls[_id] == 1 or track_age_cls[_id] % config.CLASSIFY_INTERVAL == 0)
                            
                            if is_cls_calc_frame:
                                crop = crop_from_box(processed_frame, bbox)
                                if crop is not None and crop.size > 0:
                                    db_type = "special" if idx_class == config.YOLOClass.GLASS else "standard"
                                    classify_tasks.append((_id, crop, db_type))
                
                # Đẩy các tác vụ phân loại ra ngoài Lock
                for t_id, crop, db_type in classify_tasks:
                    try:
                        self.cls_in_queue.put(("classify", t_id, crop, db_type), block=False)
                    except queue.Full:
                        logger.warning("Classifier Queue đầy, bỏ qua frame phân loại.")

                # Cập nhật Sliding Window cho tất cả các ID đang hoạt động
                with cache_lock:
                    for _id in active_ids:
                        if _id not in roi_sliding_windows:
                            roi_sliding_windows[_id] = deque(maxlen=SLIDING_WINDOW_SIZE)
                        is_detected = detected_in_roi_this_frame.get(_id, False)
                        roi_sliding_windows[_id].append(1 if is_detected else 0)

                # --- LỌC BỎ CÁC VẬT THỂ SAI KÍCH THƯỚC ---
                decision_boxes = []
                with cache_lock:
                    for box in valid_boxes:
                        bbox, idx_class, conf, _id = box
                        if idx_class == config.YOLOClass.HAND:
                            continue
                        
                        # Nếu đang trong quá trình detect ID này, không lọc bỏ để tiếp tục thu thập mẫu
                        if sm.is_detecting and _id == sm.target_id:
                            decision_boxes.append(box)
                            continue
                            
                        vol = volume_cache.get(_id, -1)
                        if vol != -1 and not (config.MIN_ACCEPTABLE_VOLUME < vol < config.MAX_ACCEPTABLE_VOLUME):
                            continue
                        decision_boxes.append(box)

                len_decision_boxes = len(decision_boxes)

                # --- KIỂM TRA SỰ XUẤT HIỆN CỦA TAY TRONG VÙNG ROI ---
                is_hand_in_roi = any(
                    box[1] == config.YOLOClass.HAND and roi.contains_center(box[0])
                    for box in valid_boxes
                )
                
                frameCount += 1
                display_img = processed_frame.copy()
                shape = processed_frame.shape
                
                status_text = "WAITING"
                status_color = (0, 255, 255)
                if sm.is_armed:
                    status_text = "ARMED"
                    status_color = (0, 255, 0)
                elif sm.is_detecting:
                    status_text = "DETECTING"
                    status_color = (0, 0, 255)
                
                # --- PREMIUM STACK STATISTICS HUD ---
                cls_status = "Idle" if self.cls_worker.fps < 0.1 else f"{int(self.cls_worker.fps)} FPS ({self.cls_worker.latency:.1f}ms)"
                draw_premium_hud(display_img, status_text, status_color, self.smoothed_fps, self.worker.fps, self.worker.latency, cls_status, is_hand_in_roi, roi.coords)

                if len_decision_boxes > 0:
                    if len_decision_boxes > 1 and sm.is_detecting:
                        global_emit('command', 2)
                else:
                    global_emit('command', 0)

                # Vẽ bounding boxes, thương hiệu và thể tích (đồng bộ khóa bảo vệ cache)
                with cache_lock:
                    draw_predictions(display_img, valid_boxes, volume_cache, track_cache, self.colors)

                cv2.imshow("RVM DEV MODE (AUTO)", display_img)

                # --- CẬP NHẬT QUỸ ĐẠO VÀ KIỂM TRA VƯỢT VẠCH ẢO ---
                trigger_this_frame = False
                triggered_id = -1
                is_trigger_invalid_vol = False
                invalid_volume_val = -1.0

                with cache_lock:
                    for box in valid_boxes:
                        bbox, idx_class, conf, _id = box
                        if idx_class == config.YOLOClass.HAND:
                            continue
                        
                        x1, y1, x2, y2 = map(int, bbox)
                        cy = (y1 + y2) // 2
                        
                        if _id not in track_history:
                            track_history[_id] = deque(maxlen=5)
                        track_history[_id].append(cy)
                        
                        is_above_now = roi.is_above_virtual_line(cy)
                        is_bbox_ok = roi.is_bbox_complete(y2)
                        track_age = track_age_vol.get(_id, 0)
                        
                        if is_above_now and is_bbox_ok and (_id not in triggered_ids) and track_age >= 5:
                            vol = volume_cache.get(_id, -1)
                            if vol != -1:
                                if not (config.MIN_ACCEPTABLE_VOLUME < vol < config.MAX_ACCEPTABLE_VOLUME):
                                    is_trigger_invalid_vol = True
                                    invalid_volume_val = vol
                                
                                trigger_this_frame = True
                                triggered_id = _id

                # --- TỰ ĐỘNG KÍCH HOẠT HOẶC TỪ CHỐI NGAY LẬP TỨC ---
                if sm.is_armed and trigger_this_frame and not sm.is_detecting and not is_hand_in_roi:
                    if is_trigger_invalid_vol:
                        logger.info(f"ARMED and object {triggered_id} crossed virtual line with INVALID volume ({invalid_volume_val:.1f}ml). Rejecting immediately!")
                        result = sm.reject_immediately(volume=invalid_volume_val, triggered_id=triggered_id, reason="Volume out of bounds")
                        global_emit('result', {
                            'data': result.data, 'model': str(self.__path), 'ver': CODE,
                            "id": mac_add, "images": "mock_images", "size": result.size,
                            "item": triggered_id, "volume": float(invalid_volume_val)
                        })
                        rlog.log_result(data=result.data, mac_id=mac_add,
                                        images=result.images, size=result.size,
                                        item=triggered_id, volume=invalid_volume_val,
                                        reason=result.reason)
                        triggered_ids.add(triggered_id)
                        sm.arm()
                    else:
                        if sm.trigger(triggered_id):
                            triggered_ids.add(triggered_id)
                            with cache_lock:
                                track_cache.pop(triggered_id, None)
                                track_age_cls[triggered_id] = 0
                            global_emit('auto_trigger', {'triggered': True})

                # --- HỦY KHẨN CẤP KHI PHÁT HIỆN TAY ---
                if sm.is_detecting and is_hand_in_roi:
                    logger.info("--- Hand detected in ROI! Aborting detection immediately! ---")
                    with cache_lock:
                        vol_val = float(volume_cache.get(sm.target_id, -1))
                    result = sm.abort("Hand safety abort")
                    result.volume = vol_val
                    global_emit('result', {
                        'data': result.data, 'model': str(self.__path), 'ver': CODE,
                        "id": mac_add, "images": "mock_images", "size": result.size,
                        "item": result.item, "volume": result.volume
                    })
                    rlog.log_result(data=result.data, mac_id=mac_add,
                                    images=result.images, size=result.size,
                                    item=result.item, volume=result.volume,
                                    reason=result.reason)
                    global_emit('command', 0)
                    sm.arm()
                    logger.info("\n--- System re-armed due to Hand safety trigger. ---")

                elif sm.is_detecting:
                    if frameCount % 4 != 0 or frameCount < 4:
                        pass 
                    else:
                        if sm.should_finalize():
                            with cache_lock:
                                result = sm.finalize(volume_cache)
                            logger.info(f"--- DETECTION FINISHED ---")
                            logger.info(f"Collected IDs: {sm.calc_ids}")
                            logger.info(f"Final Average Result: {result.data} (0=Aquafina, 1=Plastic, 2=Can, 7=Invalid)")
                            global_emit('result', {
                                'data': result.data, 'model': str(self.__path), 'ver': CODE,
                                "id": mac_add, "images": "mock_images", "size": result.size, "item": result.item,
                                "volume": result.volume
                            })
                            rlog.log_result(data=result.data, mac_id=mac_add,
                                            images=result.images, size=result.size,
                                            item=result.item, volume=result.volume)
                            sm.arm()
                            logger.info("\n--- System re-armed. Waiting for next object. ---")

                        else:
                            if len_decision_boxes > 0:
                                box = decision_boxes[0]
                                _id = box[3]
                                x1, y1, x2, y2 = box[0]
                                size_val = max((y2 - y1) / shape[0] * 100, (x2 - x1) / shape[1] * 100)
                                
                                # --- HIERARCHICAL DECISION LOGIC V2 ---
                                with cache_lock:
                                    final_class, is_unknown_class2 = evaluate_hierarchical_class(box, volume_cache, track_cache, roi.coords)
                                
                                if is_unknown_class2:
                                    logger.info("--- Unknown brand detected for class 2. Terminating detection immediately! ---")
                                    with cache_lock:
                                        vol_val = float(volume_cache.get(sm.target_id, -1))
                                    result = sm.abort("Unknown brand class 2")
                                    result.volume = vol_val
                                    global_emit('result', {
                                        'data': config.RVMClass.REJECT, 'model': str(self.__path), 'ver': CODE,
                                        "id": mac_add, "images": "mock_images", "size": result.size,
                                        "item": result.item, "volume": result.volume
                                    })
                                    rlog.log_result(data=config.RVMClass.REJECT, mac_id=mac_add,
                                                    images=result.images, size=result.size,
                                                    item=result.item, volume=result.volume,
                                                    reason=result.reason)
                                    global_emit('command', 0)
                                    sm.arm()
                                    logger.info("\n--- System re-armed due to Class 2 Unknown brand. ---")
                                else:
                                    sm.add_sample(transform_id(final_class), size_val)
                            else:
                                global_emit('command', 0)
                            
                            logger.info(f"   [DETECTING] collecting... calc_ids: {sm.calc_ids}")

            except queue.Empty:
                pass

            if not self.in_queue.full():
                self.in_queue.put((frame_curr, sm.is_detecting, next_start_time))

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
