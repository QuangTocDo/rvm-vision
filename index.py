import logging
import os
import time
import math
from datetime import datetime, timedelta
from pathlib import Path
import sys
import threading
import queue
import cv2
import numpy as np
import torch
# pyrefly: ignore [missing-import]
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# ADD GLOBAL ROOT PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from s3_worker import init, sync, unix
from tracker import Tracker

from utils.camera_utils import FindCamera, open_camera
from utils.helper_functions import crop_from_box
from utils.hierarchy import evaluate_hierarchical_class, estimate_volume
from workers.inference import InferenceWorker
from workers.classifier import ClassifierWorker
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
    counts = Counter(calc_ids)
    most_common_class, count = counts.most_common(1)[0]
    return most_common_class

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
    
    roi_x1, roi_y1, roi_x2, roi_y2 = config.ROI_COORDS
    
    volume_cache = {}
    track_age_vol = {}  
    
    # Caches cho Classifier
    track_cache = {}
    track_age_cls = {}
    
    # Cửa sổ trượt lưu lịch sử phát hiện trong ROI (Sliding Window Filter)
    roi_sliding_windows = {}
    SLIDING_WINDOW_SIZE = config.SLIDING_WINDOW_ROI
    
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
                    if track_age_vol[_id] == 1 or track_age_vol[_id] % config.CLASSIFY_INTERVAL == 0:
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
                        if track_age_cls[_id] == 1 or track_age_cls[_id] % config.CLASSIFY_INTERVAL == 0:
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
                    if len(window) >= config.MIN_SAMPLES_ROI_CHECK:
                        density = sum(window) / len(window)
                        if density >= config.ROI_STABILITY_THRESHOLD:
                            is_stable = True
                            break
            
            if is_stable:
                print(f"ARMED and object is stable in ROI (Sliding Window >= {int(config.ROI_STABILITY_THRESHOLD * 100)}%). Triggering detection.")
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

                    if ii >= config.MAX_DECISION_SAMPLES or endTime - beginTime > config.DETECTION_TIMEOUT:
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
                            
                            # --- HIERARCHICAL DECISION LOGIC V2 ---
                            final_class, is_unknown_class2 = evaluate_hierarchical_class(box, volume_cache, track_cache, config.ROI_COORDS)
                            
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
