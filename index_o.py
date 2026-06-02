import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import getnode as get_mac
import sys

import cv2
import numpy as np
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit, send

# ADD GLOBAL ROOT PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import get_model_path
from s3_worker import init, sync, unix
from tracker import Tracker

# IMPORT CORE AI TỪ onnx_v2
from onnx_v2.yolo_det_onnx import YOLODetONNX
from onnx_v2.triplet_classifier_onnx import TripletClassifier
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

def average(lst):
    if lst == None or len(lst) == 0:
        return 0
    return sum(lst) / len(lst)

emit_times = {}

def global_emit(event, data):
    global emit_times
    try:
        lastTime = emit_times.get(event, 0)
        if time.time() - lastTime < 0.5 and event!="result":
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
            _delta_seconds = datetime(year=__date.year, month=__date.month, day=__date.day, hour=20, minute=1, second=0) - __date
            socketio.sleep(_delta_seconds.seconds)
        else:
            socketio.sleep(seconds)

__dict={0:0, 1:1, 2:2, 3:2, 4:2}

def transform_id(id_detect):
    global __dict
    if __dict is None:
        return id_detect
    return __dict.get(str(id_detect), id_detect)

def calc_avg(calc_ids):
    if calc_ids is None:
        return -1
    if len(calc_ids) ==0:
        return -1
    avg = (sum(calc_ids) * 1.0) / len(calc_ids)
    if avg < 0:
        return -1
    if avg < 0.5:
        return 0
    if avg < 1.5:
        return 1
    return 2

def open_camera(id_camera):
    camera = cv2.VideoCapture(id_camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera


def run():
    global detext, beginTime, flag_camera
    
    print("INIT NEW ONNX CORE AI")
    detector = YOLODetONNX()
    classifier = TripletClassifier()
    __path = "onnx_v2_model" # Dummy for emit

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
    
    # Giữ Tracker của RVM-Vision
    tracker = Tracker()
    __error_times__ = 0
    
    # State cho Classification
    track_cache = {}
    track_age = {}
    voting_history = {}
    VOTING_WINDOW = 5

    def _get_voted_class(track_id, current_pred, current_score):
        if track_id not in voting_history:
            voting_history[track_id] = deque(maxlen=VOTING_WINDOW)
        voting_history[track_id].append(current_pred)
        counts = Counter(voting_history[track_id])
        return counts.most_common(1)[0][0], current_score
    
    while True:
        ret, im = camera.read()
        if ret == False:
            __error_times__ = __error_times__+1
            if __error_times__ % 10 ==0:
                camera_ii = FindCamera()
                if camera_ii < 0:
                    print("Can not open camera")
                    continue
                camera = open_camera(camera_ii)
            socketio.sleep(0.1)
            continue
            
        __error_times__ = 0
        frameCount += 1
        
        if ret:
            img = im
            shape = img.shape
            ii = len(calc_ids)
            
            # --- DETECT BẰNG CORE AI MỚI ---
            raw_detections = detector.detect(img)
            
            detections = []
            for det in raw_detections:
                x1, y1, x2, y2 = det["box"]
                if x1 < CAMERAS[1] or y1 < CAMERAS[2]:
                    continue
                
                score = det["score"]
                idx_class = int(det["class_name"])
                detections.append([x1, y1, x2, y2, idx_class, score])
            
            # --- TRACKING BẰNG TRACKER CỦA RVM-VISION ---
            tracker.update(img, detections)
            
            valid_boxes = []
            for track in tracker.tracks:
                bbox = track.bbox
                track_id = track.track_id
                valid_boxes.append((bbox, track.id, track.confidence, track_id))
            
            len_valid_boxes = len(valid_boxes)
            if len_valid_boxes > 0:
                if len_valid_boxes > 1 and detext:
                    global_emit('command', 2)
                valid_boxes.sort(key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]), reverse=True)
            else:
                global_emit('command', 0)
                
            # --- PHÂN LOẠI NHÃN HIỆU CHO CÁC TRACKED BOXES ---
            
            # Lọc cache
            active_ids = [vb[3] for vb in valid_boxes]
            for k in list(track_cache.keys()):
                if k not in active_ids:
                    del track_cache[k]
            for k in list(track_age.keys()):
                if k not in active_ids:
                    del track_age[k]
            for k in list(voting_history.keys()):
                if k not in active_ids:
                    del voting_history[k]

            # Batch classify
            crops = []
            classify_ids = []
            for box in valid_boxes:
                bbox, idx_class, conf, _id = box
                x1, y1, x2, y2 = bbox
                track_age[_id] = track_age.get(_id, 0) + 1
                
                # Classify interval = 5
                if _id not in track_cache or track_age[_id] % 5 == 0:
                    crop = crop_from_box(img, [x1, y1, x2, y2])
                    if crop is not None and crop.size > 0:
                        crops.append(crop)
                        classify_ids.append(_id)
                        
            if len(crops) > 0:
                try:
                    preds = classifier.predict_batch(crops)
                    for t_id, (pred_class, p_score) in zip(classify_ids, preds):
                        v_class, v_score = _get_voted_class(t_id, pred_class, p_score)
                        track_cache[t_id] = (v_class, v_score)
                except Exception as e:
                    pass

            if detext:
                if frameCount % 4 != 0 or frameCount< 4:
                    continue

                endTime = time.time()

                if ii >= 3:
                    detext = False
                    final_result = 0
                    global_emit('result', {'data': calc_avg(calc_ids), 'model': str(__path), 'ver': CODE,
                                "id": mac_add, "images": images, "size": average(sizes), "item": id})
                    images = []
                    sizes = []
                    calc_ids = []
                    flag_camera = False
                    id = -1
                    continue

                if endTime - beginTime > 2.6:
                    detext = False
                    final_result = 0
                    if ii > 0:
                        avg = calc_avg(calc_ids)
                    else:
                        avg = -1

                    calc_ids = []

                    if id > 0:
                        caches_ids.append(id)

                    global_emit('result', {'data': avg, 'model': str(__path), 'ver': CODE,
                                "id": mac_add, "images": images, "size": average(sizes), "item": id})
                    flag_camera = False
                    images = []
                    sizes = []
                    id = -1
                    continue

                boxes = []

                if len_valid_boxes > 0:
                    flg_append = True
                    for box in valid_boxes:
                        idx_class = box[1]
                        x1, y1, x2, y2 = box[0]
                        conf = box[2]
                        _id = box[3]
                        if _id > 0:
                            caches_ids.append(_id)

                        boxes.append([x1, y1, x2, y2, conf, idx_class, frameCount,
                                     beginTime, endTime - beginTime, len(valid_boxes), _id])

                        if x2 > CAMERAS[3]:
                            continue
                        if y2 > CAMERAS[4]:
                            continue

                        if flg_append:
                            id = _id
                            sizes.append(
                                max((y2 - y1) / shape[0] * 100, (x2 - x1) / shape[1] * 100))
                            
                            calc_ids.append(transform_id(idx_class))
                            flg_append = False
                else:
                    boxes.append([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                                 frameCount, int(beginTime), endTime - beginTime, 0, -1])
                    global_emit('command', 0)
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
                img_path = _dir + "/" + str(mac_add) + "_" + CODE + "_" + datetime.strftime(datetime.utcnow(),
                                                                                            "%Y-%m-%d %X%Z") + "_" + str(frameCount)
                np.savez_compressed(img_path, image=img,
                                    box=np.array(boxes, dtype='float'))
                images.append(img_path + ".npz")
            elif flag_camera:
                if frameCount % 3 != 0:
                    continue

                if len(valid_boxes) > 0:
                    _id = valid_boxes[0][3]
                    print("OTHER", final_result, _id, caches_ids)
                    if _id < 1 or (_id not in caches_ids):
                        final_result += 1
                    if final_result > 4:
                        global_emit('detect', {'data': True, 'model': str(
                            __path), 'ver': CODE, "id": mac_add})
                        final_result = 0
                        flag_camera = False
                        caches_ids = [_id]
                        frameCount=0
                    pass
                elif frameCount % 6 == 0:
                    final_result = max(0, final_result - 1)
            else:
                final_result = 0

                if len(caches_ids) > 500:
                    caches_ids = []
                if frameCount % 1000 == 0:
                    tracker.reset()

        if frameCount > 99999:
            frameCount = 0
        socketio.sleep(0.05)
        pass


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
    global detext, beginTime, flag_camera, __dict
    print(data, "data", time.time() - beginTime, time.time())
    if "transform" in data:
        __dict=data["transform"]
        print(__dict, transform_id(4))
    
    if "detext" in data:
        if not detext:
            beginTime = time.time()
        detext = data["detext"]
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