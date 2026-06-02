import logging
import os
import time
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import getnode as get_mac
import sys
import threading
import queue
import cv2
import numpy as np
import torch
# pyrefly: ignore [missing-import]
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit, send

# ADD GLOBAL ROOT PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from config import get_model_path
from s3_worker import init, sync, unix
from tracker import Tracker

# IMPORT CORE AI
from ultralytics import YOLO
from onnx_v2.triplet_classifier import TripletClassifier
from onnx_v2.glass_classifier import GlassClassifier
from utils.helper_functions import crop_from_box
from collections import deque, Counter

HOME = Path.home()
mac_add = "rvm"
__env = HOME / ".env"
_camera_env = HOME / ".camera.env"
CODE = "8.4.0"
CAMERAS = (-1, 0, 0, 640, 480)

if os.path.exists(__env):
    with open(__env, 'rt') as file:
        mac_add = file.readline()
elif os.path.exists(".env"):
    with open(".env", 'rt') as file:
        mac_add = file.readline()
    with open(__env, 'wt') as file:
        file.write(mac_add)
    os.unlink(".env")

if os.path.exists(_camera_env):
    with open(_camera_env, 'rt') as file:
        size = file.readline()
        info = size.split(",")
        CAMERAS = (int(info[0]), max(int(info[1]), 0), max(int(info[2]), 0), int(info[3]), int(info[4]))

elif os.path.exists(".env.camera"):
    with open(".env.camera", 'rt') as file:
        size = file.readline()
        info = size.split(",")
        CAMERAS = (int(info[0]), max(int(info[1]), 0), max(int(info[2]), 0), int(info[3]), int(info[4]))

    with open(_camera_env, 'wt') as file:
        file.write(str(CAMERAS[0]) + "," + str(CAMERAS[1]) + "," +
                   str(CAMERAS[2]) + "," + str(CAMERAS[3]) + "," + str(CAMERAS[4]))
    os.unlink(".env.camera")
else:
    with open(_camera_env, 'wt') as file:
        file.write(str(CAMERAS[0]) + "," + str(CAMERAS[1]) + "," +
                   str(CAMERAS[2]) + "," + str(CAMERAS[3]) + "," + str(CAMERAS[4]))

if not mac_add.startswith("r-"):
    init()
    mac_add = unix()
    with open(__env, 'wt') as file:
        file.write(mac_add)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode=None)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
beginTime = 0

_dir = "./temp/" + datetime.strftime(datetime.utcnow(), "%Y-%m-%d")
if not os.path.exists(_dir):
    os.makedirs(_dir)

detext = False
flag_camera = False
detection_armed = False # Trạng thái chờ (armed)

def FindCamera():
    index = 0
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cap.release()
            return index
        index += 1
        i -= 1
    return -1


# +++ HÀM TÍNH THỂ TÍCH VẬT THỂ +++
def estimate_volume(width, height, scale=config.PIXEL_TO_CM_RATIO, k=1.0):
    length_cm = width * scale
    diameter_cm = height * scale
    r = diameter_cm / 2
    volume = math.pi * (r ** 2) * length_cm * k
    return volume


def average(lst):
    if lst == None or len(lst) == 0:
        return 0
    return sum(lst) / len(lst)


emit_times = {}


def global_emit(event, data):
    global emit_times
    try:
        lastTime = emit_times.get(event, 0)
        if time.time() - lastTime < 0.5 and event != "result":
            return
        emit_times[event] = time.time()

        with app.test_request_context('/'):
            emit(event, data, broadcast=True, namespace="/")

        print("emit", event, data, time.time())
    except Exception as e:
        print(e)
        pass


def sync_dir():
    seconds = 24 * 60 * 60
    while True:
        socketio.sleep(3)
        init()
        socketio.sleep(30)
        now = datetime.strftime(datetime.now(), "%H:%M")

        if now < "20:00":
            date = datetime.strftime(datetime.utcnow() - timedelta(days=1), "%Y-%m-%d")
        else:
            date = datetime.strftime(datetime.utcnow(), "%Y-%m-%d")

        sync(mac_add, date)

        if now < "20:00":
            __date = datetime.now()
            _delta_seconds = datetime(year=__date.year, month=__date.month, day=__date.day, hour=20, minute=1,
                                      second=0) - __date
            socketio.sleep(_delta_seconds.seconds)
        else:
            socketio.sleep(seconds)


__dict = {0: 2, 1:1, 10: 0}
def transform_id(id_detect):
    global __dict
    
    if id_detect == 0:
        old_class = 2
    elif id_detect == 1:
        old_class = 1
    else:
        old_class = id_detect
        
    if __dict is None:
        return old_class
  
    if str(old_class) in __dict:
        return __dict[str(old_class)]
    return __dict.get(old_class, old_class)


def calc_avg(calc_ids):
    if calc_ids is None or len(calc_ids) == 0:
        return -1
    
    # Sử dụng bỏ phiếu đa số (Majority Voting / Mode) thay vì tính trung bình cộng số học
    counts = Counter(calc_ids)
    most_common_class, count = counts.most_common(1)[0]
    return most_common_class

def open_camera(id_camera):
    camera = cv2.VideoCapture(id_camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


class InferenceWorker(threading.Thread):
    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True

        self.__path = config.YOLO_DET_MODEL_PATH
        self.detector = YOLO(self.__path)
        self.tracker = Tracker()
        
        self._stop_event = threading.Event()
        self.reset_tracker_flag = False

    def stop(self):
        self._stop_event.set()

    def request_tracker_reset(self):
        self.reset_tracker_flag = True
        
    def run(self):
        global CAMERAS
        while not self._stop_event.is_set():
            try:
                data = self.in_queue.get(timeout=0.1)
                if data is None:
                    break
                frame_curr, current_detext, start_time = data
                
                if self.reset_tracker_flag:
                    self.tracker.reset()
                    self.reset_tracker_flag = False
                
                # Lật gương ảnh ở luồng phụ thay vì luồng chính để tiết kiệm tài nguyên CPU
                frame_curr = cv2.flip(frame_curr, 1)
                img = frame_curr.copy()
                
                valid_boxes = []
                
                results = self.detector(img, conf=0.7, agnostic_nms=True, iou=0.81, verbose=False)
                
                detections = []
                if results[0].boxes.shape[0] > 0:
                    boxes = results[0].boxes
                    for i in range(boxes.shape[0]):
                        boxx = boxes[i]
                        a = boxx.xyxy
                        
                        x1 = int(a[0, 0])
                        y1 = int(a[0, 1])
                        x2 = int(a[0, 2])
                        y2 = int(a[0, 3])
                        
                        if x1 < CAMERAS[1] or y1 < CAMERAS[2]:
                            continue
                        
                        score = float(boxx.conf[0])
                        idx_class = int(boxx.cls[0])
                        
                        # Chỉ phát hiện và bám vết các lớp 0, 1, 2 và lớp bàn tay 5
                        if idx_class not in [0, 1, 2, 5]:
                            continue
                        
                        detections.append([x1, y1, x2, y2, idx_class, score])
                
                self.tracker.update(img, detections)
                
                for track in self.tracker.tracks:
                    bbox = track.bbox
                    track_id = track.track_id
                    valid_boxes.append((bbox, track.id, track.confidence, track_id))
                
                if len(valid_boxes) > 0:
                    valid_boxes.sort(key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]), reverse=True)
                    
                if not self._stop_event.is_set():
                    self.out_queue.put((frame_curr, valid_boxes, start_time))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker Error: {e}")

class ClassifierWorker(threading.Thread):
    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True

        # Tải database đặc trưng tiêu chuẩn (PC)
        db_path = config.DATABASE_EMBEDDING_PC
        self.classifier = TripletClassifier(
            model_path=config.TRIPLET_MODEL_PATH,
            database_path=str(db_path)
        )
        
        # --- NÂNG CẤP LỚP 2: Khởi tạo mô hình phân loại nhị phân PyTorch cho Thủy tinh Lanh dùng mô-đun riêng ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.glass_classifier = GlassClassifier(
            model_path=config.GLASS_MODEL_PATH,
            device=self.device,
            conf_thres=0.85
        )
        
        self.track_embeddings = {}
        self.voting_history = {}
        self.VOTING_WINDOW = 5
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _get_voted_class(self, track_id, current_pred, current_score):
        if track_id not in self.voting_history:
            self.voting_history[track_id] = deque(maxlen=self.VOTING_WINDOW)
        self.voting_history[track_id].append(current_pred)
        counts = Counter(self.voting_history[track_id])
        return counts.most_common(1)[0][0], current_score

    def run(self):
        while not self._stop_event.is_set():
            try:
                task = self.in_queue.get(timeout=0.1)
                if task is None:
                    break
                
                task_type = task[0]
                if task_type == "classify":
                    _, _id, crop, db_type = task
                    try:
                        if db_type == "special":
                            # --- PHÂN LOẠI NHỊ PHÂN CHO THỦY TINH CLASS 2 DÙNG MODULE RIÊNG ---
                            pred_class, p_score = self.glass_classifier.predict(crop)
                            
                            # Đưa qua bộ lọc bỏ phiếu Sliding Window để tối ưu hóa quyết định
                            v_class, v_score = self._get_voted_class(_id, pred_class, p_score)
                            self.out_queue.put((_id, v_class, v_score))
                        else:
                            # --- PHÂN LOẠI TRUYỀN THỐNG DÙNG TRIPLET CHO CHAI NHỰA/LON ---
                            embs = self.classifier.extract_batch([crop])
                            if embs.size > 0:
                                emb = embs[0]
                                norm = np.linalg.norm(emb)
                                emb_norm = emb / norm if norm > 0 else emb

                                # Làm mượt embedding qua thời gian (EMA) để chống rung/flickering
                                if _id not in self.track_embeddings:
                                    self.track_embeddings[_id] = emb_norm
                                else:
                                    alpha = 0.6  # Hệ số làm mượt
                                    smoothed = alpha * self.track_embeddings[_id] + (1.0 - alpha) * emb_norm
                                    smoothed_norm = np.linalg.norm(smoothed)
                                    self.track_embeddings[_id] = smoothed / smoothed_norm if smoothed_norm > 0 else smoothed

                                # Phân loại dựa trên vector đã được làm mượt
                                preds = self.classifier.predict_embeddings([self.track_embeddings[_id]],
                                                                            threshold=config.SIM_THRESHOLD, 
                                                                            margin_thres=config.MARGIN_THRESHOLD,
                                                                            outlier_floor=config.OUTLIER_RADIUS_FLOOR)
                                pred_class, p_score = preds[0]

                                # Đưa qua bộ lọc bỏ phiếu Sliding Window để tối ưu hóa quyết định
                                v_class, v_score = self._get_voted_class(_id, pred_class, p_score)
                                self.out_queue.put((_id, v_class, v_score))
                    except Exception as e:
                        print(f"Classifier Error for ID {_id}: {e}")

                elif task_type == "cleanup":
                    _, active_ids = task
                    # Dọn dẹp cache của các track không hoạt động
                    for cache_dict in (self.track_embeddings, self.voting_history):
                        inactive_keys = cache_dict.keys() - active_ids
                        for k in inactive_keys:
                            if k in cache_dict:
                                del cache_dict[k]

            except queue.Empty:
                continue
            except Exception as e:
                print(f"ClassifierWorker Error: {e}")

def run():
    global detext, beginTime, flag_camera, detection_armed
    
    __path = config.YOLO_DET_MODEL_PATH
    print("INIT AI CORE WITH HIERARCHICAL ROUTING V2", __path)

    caches_ids = []
    camera_ii = CAMERAS[0]

    if camera_ii < 0:
        camera_ii = FindCamera()
        if camera_ii < 0:
            print("Can not open camera")
            return

    camera = open_camera(camera_ii)

    id = -1
    frameCount = 0

    final_result = 0
    calc_ids = []
    images = []
    sizes = []
    __error_times__ = 0
    
    roi_x1, roi_y1 = 200, 25
    roi_x2, roi_y2 = 475, 400
    

    volume_cache = {}
    track_age_vol = {}  
    
    # Caches cho Classifier
    track_cache = {}
    track_age_cls = {}
    
    # Cửa sổ trượt lưu lịch sử phát hiện trong ROI (Sliding Window Filter)
    roi_sliding_windows = {}
    SLIDING_WINDOW_SIZE = 6
    
    in_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=2)
    
    worker = InferenceWorker(in_queue, out_queue)
    worker.start()

    # Khởi tạo và khởi động ClassifierWorker chạy bất đồng bộ
    cls_in_queue = queue.Queue(maxsize=4)
    cls_out_queue = queue.Queue(maxsize=4)
    cls_worker = ClassifierWorker(cls_in_queue, cls_out_queue)
    cls_worker.start()

    while True:
        ret, im = camera.read()
        if not ret:
            __error_times__ += 1
            if __error_times__ % 10 == 0:
                camera_ii = FindCamera()
                if camera_ii >= 0:
                    camera = open_camera(camera_ii)
            socketio.sleep(0.1)
            continue

        __error_times__ = 0
        
        next_start_time = time.time()

        try:
            processed_frame, valid_boxes, start_time = out_queue.get(timeout=0.01)
            
            # --- ĐỌC KẾT QUẢ PHÂN LOẠI TỪ LUỒNG PHỤ PHÂN LOẠI ---
            while not cls_out_queue.empty():
                try:
                    res_id, res_class, res_score = cls_out_queue.get_nowait()
                    track_cache[res_id] = (res_class, res_score)
                except queue.Empty:
                    break
            
            # --- TÍNH TOÁN VÀ ĐỒNG BỘ CÁC BỘ ĐỆM CHO TẤT CẢ VẬT THỂ ĐANG ĐƯỢC BÁM VẾT ---
            active_ids = {box[3] for box in valid_boxes}
            for cache_dict in (volume_cache, track_age_vol, track_cache, track_age_cls, roi_sliding_windows):
                inactive_keys = cache_dict.keys() - active_ids
                for k in inactive_keys:
                    if k in cache_dict:
                        del cache_dict[k]
            
            # Gửi yêu cầu dọn dẹp sang ClassifierWorker
            if not cls_in_queue.full():
                cls_in_queue.put(("cleanup", active_ids))


            detected_in_roi_this_frame = { _id: False for _id in active_ids }

            for box in valid_boxes:
                bbox, idx_class, conf, _id = box
                if idx_class not in [0, 1, 2]:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                    detected_in_roi_this_frame[_id] = True
                    
                    # 1. Đo thể tích vật thể
                    track_age_vol[_id] = track_age_vol.get(_id, 0) + 1
                    if track_age_vol[_id] == 1 or track_age_vol[_id] % 5 == 0:
                        w_box, h_box = x2 - x1, y2 - y1
                        length_px, diameter_px = max(w_box, h_box), min(w_box, h_box)
                        if length_px < 300:
                            vol = estimate_volume(length_px, diameter_px, config.PIXEL_TO_CM_RATIO, config.VOLUME_SCALING_UP)
                            volume_cache[_id] = vol
                        elif length_px > 700:
                            vol = estimate_volume(length_px, diameter_px, config.PIXEL_TO_CM_RATIO, config.VOLUME_SCALING_DOWN)
                            volume_cache[_id] = vol
                        else:
                            vol = estimate_volume(length_px, diameter_px, config.PIXEL_TO_CM_RATIO, 1)
                            volume_cache[_id] = vol
                    # 2. Nhận diện thương hiệu (chỉ khi thể tích hợp lệ và idx_class thuộc [0, 1, 2])
                    current_vol = volume_cache.get(_id, -1)
                    if config.MIN_ACCEPTABLE_VOLUME < current_vol < config.MAX_ACCEPTABLE_VOLUME and idx_class in [0, 1, 2]:
                        track_age_cls[_id] = track_age_cls.get(_id, 0) + 1
                        if track_age_cls[_id] == 1 or track_age_cls[_id] % 5 == 0:
                            crop = crop_from_box(processed_frame, bbox)
                            if crop is not None and crop.size > 0:
                                if not cls_in_queue.full():
                                    db_type = "special" if idx_class == 2 else "standard"
                                    cls_in_queue.put(("classify", _id, crop, db_type))

            # Cập nhật Cửa sổ trượt cho tất cả các ID đang hoạt động
            for _id in active_ids:
                if _id not in roi_sliding_windows:
                    roi_sliding_windows[_id] = deque(maxlen=SLIDING_WINDOW_SIZE)
                is_detected = detected_in_roi_this_frame.get(_id, False)
                roi_sliding_windows[_id].append(1 if is_detected else 0)

            # --- KIỂM TRA SỰ XUẤT HIỆN CỦA TAY TRONG VÙNG ROI ---
            is_hand_in_roi = False
            for box in valid_boxes:
                bbox, idx_class, conf, _id = box
                if idx_class == 5:
                    cx = (bbox[0] + bbox[2]) // 2
                    cy = (bbox[1] + bbox[3]) // 2
                    if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                        is_hand_in_roi = True
                        break

            # --- LỌC BỎ CÁC VẬT THỂ SAI KÍCH THƯỚC ĐỂ COI NHƯ KHÔNG CÓ VẬT THỂ ---
            decision_boxes = []
            for box in valid_boxes:
                bbox, idx_class, conf, _id = box
                if idx_class == 5:
                    continue  # Bỏ qua tay khi lọc chai
                vol = volume_cache.get(_id, -1)
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
                    if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                        is_object_in_roi = True
                        break

            # --- TỰ ĐỘNG KÍCH HOẠT NHẬN DIỆN QUA CỬA SỔ TRƯỢT ---
            is_stable = False
            if detection_armed and is_object_in_roi and not detext and not is_hand_in_roi:
                for box in decision_boxes:
                    _id = box[3]
                    window = roi_sliding_windows.get(_id, [])
                    if len(window) >= 5:
                        density = sum(window) / len(window)
                        if density >= 0.75:
                            is_stable = True
                            break
            
            if is_stable:
                print("ARMED and object is stable in ROI (Sliding Window >= 75%). Triggering detection.")
                detext = True
                beginTime = time.time()
                detection_armed = False
                global_emit('auto_trigger', {'triggered': True})

            frameCount += 1
            shape = processed_frame.shape
            ii = len(calc_ids)

            if len_decision_boxes > 0:
                if len_decision_boxes > 1 and detext:
                    global_emit('command', 2)
            else:
                global_emit('command', 0)

            # --- TỰ ĐỘNG NGẮT SOI KHẨN CẤP NẾU CÓ TAY XUẤT HIỆN TRONG ROI ---
            if detext and is_hand_in_roi:
                print("--- Hand detected in ROI! Aborting detection immediately! ---")
                global_emit('result', {'data': -1, 'model': str(__path), 'ver': CODE,
                                       "id": mac_add, "images": images, "size": 0, "item": -1,
                                       "volume": 0.0})
                global_emit('command', 0)
                detext = False
                final_result = 0
                images, sizes, calc_ids, id = [], [], [], -1
                flag_camera = False

            elif detext:
                if frameCount % 4 != 0 or frameCount < 4:
                    pass
                else:
                    endTime = time.time()

                    if ii >= 8 or endTime - beginTime > 4:
                        avg = calc_avg(calc_ids) if ii > 0 else -1
                        global_emit('result', {'data': avg, 'model': str(__path), 'ver': CODE,
                                               "id": mac_add, "images": images, "size": average(sizes) if sizes else 0, "item": id,
                                               "volume": float(volume_cache.get(id, -1))})
                        with open("log.txt", "a", encoding="utf-8") as f:
                            f.write(f"{datetime.now()} | data={avg} | id={mac_add} | images={images} | size={average(sizes) if sizes else 0} | item={id} | volume={float(volume_cache.get(id, -1))}\n")
                        detext = False
                        final_result = 0
                        images, sizes, calc_ids, id = [], [], [], -1
                        flag_camera = False
                    else:
                        if len_decision_boxes > 0:
                            box = decision_boxes[0]
                            x1, y1, x2, y2 = box[0]
                            _id = box[3]
                            idx_class = box[1]
                            id = _id
                            sizes.append(max((y2 - y1) / shape[0] * 100, (x2 - x1) / shape[1] * 100))
                            
                            # --- ĐỊNH TUYẾN PHÂN CẤP CLASS 3 LỚP (HIERARCHICAL ROUTING V2) ---
                            final_class = 7 # Default to INVALID
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            
                            is_unknown_class2 = False
                            
                            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                                current_vol = volume_cache.get(_id, -1)
                                if config.MIN_ACCEPTABLE_VOLUME < current_vol < config.MAX_ACCEPTABLE_VOLUME:
                                    if idx_class == 0:
                                        # brand, b_score = track_cache.get(_id, ("unknown", 0.0))
                                        # if brand.lower() == "unknown":
                                        final_class = 2
                                        # else:
                                        #     final_class = 2
                                    elif idx_class == 1:
                                        # brand, b_score = track_cache.get(_id, ("unknown", 0.0))
                                        # if brand.lower() == "aquafina":
                                        #     final_class = 1
                                        # else:
                                        #     final_class = 1
                                        final_class = 1
                                    elif idx_class == 2:
                                        # Only check brand if the classifier has produced a prediction
                                        if _id in track_cache:
                                            brand, b_score = track_cache[_id]
                                            if brand.lower() == "lanh":
                                                final_class = 1
                                            else:
                                                is_unknown_class2 = True
                                        else:
                                            # Temporarily assign class 1 until brand prediction is available
                                            final_class = 1
                            
                            if is_unknown_class2:
                                print("--- Unknown brand detected for class 2. Terminating detection immediately! ---")
                                global_emit('result', {'data': 7, 'model': str(__path), 'ver': CODE,
                                            "id": mac_add, "images": images, "size": average(sizes) if sizes else 0, "item": id,
                                            "volume": float(volume_cache.get(id, -1))})
                                global_emit('command', 0)
                                detext = False
                                final_result = 0
                                images, sizes, calc_ids, id = [], [], [], -1
                                flag_camera = False
                            else:
                                calc_ids.append(transform_id(final_class))
                                # calc_ids.append(final_class)

                        else:
                            global_emit('command', 0)
                        
            elif flag_camera:
                if frameCount % 3 == 0 and len_decision_boxes > 0:
                    _id = decision_boxes[0][3]
                    if _id < 1 or (_id not in caches_ids):
                        final_result += 1
                    if final_result > 4:
                        global_emit('detect', {'data': True, 'model': str(__path), 'ver': CODE, "id": mac_add})
                        final_result = 0
                        flag_camera = False
                        caches_ids.append(_id)
                        frameCount = 0
                elif frameCount % 6 == 0:
                    final_result = max(0, final_result - 1)
            else:
                final_result = 0
                if len(caches_ids) > 500: caches_ids = []
                if frameCount % 1000 == 0: worker.request_tracker_reset()

        except queue.Empty:
            pass
            
        if not in_queue.full():
            in_queue.put((im, detext, next_start_time))

        if frameCount > 99999: frameCount = 0
        socketio.sleep(0.01)


@socketio.on('connect')
def connect(data):
    print(data)
    pass


@socketio.event
def my_event(message):
    print(message)
    emit('my_response', {'data': message["data"]}, broadcast=True)


@socketio.event
def run_detect(data):
    global detext, beginTime, flag_camera, __dict, detection_armed
    print(data, "data", time.time() - beginTime, time.time())
    if "transform" in data:
        __dict = data["transform"]
        print(__dict, transform_id(4))
        # print("Ignore legacy transform:", data["transform"])

    if "detext" in data:
        if data["detext"]:
            if not detext and not detection_armed:
                print("Arming for detection...")
                detection_armed = True
        else:
            detext = False
            detection_armed = False
            print("Detection cancelled by user.")

    if "camera" in data:
        flag_camera = data["camera"]


@app.route('/camera')
def camera():
    return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    print("OPEN CAMERA", mac_add, CAMERAS, datetime.strftime(
        datetime.utcnow(), "%Y-%m-%d %X%Z"))
    socketio.start_background_task(target=run)
    socketio.start_background_task(target=sync_dir)
    print("RUNS")
    socketio.run(app)
