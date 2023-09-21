import cv2
import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, send, emit
from flask_cors import CORS
from ultralytics import YOLO
from uuid import getnode as get_mac
import os
import time
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta, timezone
from s3_worker import unix, sync, init
from config import get_model_path
# import supervision as sv

from tracker import Tracker
HOME = Path.home()
mac_add = "rvm"
__env = HOME / ".env"
_camera_env = HOME / ".camera.env"
CODE = "8.4.0"
CAMERAS = (0, 0, 0, 640, 480)


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

# tracker = ROOT / "botsort.yaml"

_dir = "./temp/" + datetime.strftime(datetime.utcnow(), "%Y-%m-%d")
if not os.path.exists(_dir):
    os.makedirs(_dir)


detext = False
flag_camera = False


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
            _delta_seconds = datetime(year=__date.year, month=__date.month, day=__date.day, hour=23, minute=1, second=0) - __date
            socketio.sleep(_delta_seconds.seconds)
        else:
            socketio.sleep(seconds)
__dict={0:0, 1:1, 2:2, 3:2, 4:2}

def transform_id(id_detect):
    global __dict
    if __dict is None:
        return id_detect
     #idx_class if idx_class < 3 else 2
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


def run():
    global detext, beginTime, flag_camera
    __path = get_model_path()
    print("USE", __path, CODE)
    model = YOLO(__path)
    caches_ids = []
    camera = cv2.VideoCapture(CAMERAS[0])
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    id = -1
    frameCount = 0

    final_result = 0
    calc_ids = []
    images = []
    sizes = []
    tracker = Tracker()
    
    while True:
        ret, im = camera.read()
        frameCount += 1
        if ret:
            img = im
            shape = img.shape
            ii = len(calc_ids)
            results = model(img, conf=0.7, agnostic_nms=True, iou=0.81, verbose=False)
            detections = []
            if results[0].boxes.shape[0] > 0:
                for boxx in results[0].boxes:
                    a = boxx.xyxy
                    a = a.cpu().detach().numpy()
                    x1 = int(a[0, 0])
                    y1 = int(a[0, 1])
                    x2 = int(a[0, 2])
                    y2 = int(a[0, 3])
                    if x1 < CAMERAS[1]:
                        continue
                    if y1 < CAMERAS[2]:
                        continue
                    
                    score = float(boxx.conf.cpu().detach().numpy())
                    idx_class = int(boxx.cls.cpu().detach().numpy())
                    detections.append([x1, y1, x2, y2, idx_class, score])
            tracker.update(img, detections)
            valid_boxes = []
            for track in tracker.tracks:
                bbox = track.bbox
                track_id = track.track_id
                # print(track_id, "track_id", track.id, track.confidence)
                valid_boxes.append((bbox, track.id, track.confidence, track_id))
            len_valid_boxes = len(valid_boxes)
            if len_valid_boxes > 0:
                if len_valid_boxes > 1 and detext:
                    global_emit('command', 2)
                valid_boxes.sort(key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]), reverse=True)
            else:
                global_emit('command', 0)
            

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

                    # if len_valid_boxes > 1:
                    #     global_emit("command", len_valid_boxes)

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
                            #
                            #calc_id = idx_class if idx_class < 3 else 2
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
                        # if _id > 0:
                        #     camera_ids.append(_id)
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
        # cv2.waitKey(1)
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
        # print(data, begin_frame, detext)
    if "camera" in data:
        flag_camera = data["camera"]
    
    


@app.route('/camera')
def camera():
    return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    # run_thread = Thread(target=run)
    # run_thread.start()
    print("OPEN CAMERA", mac_add, CAMERAS, datetime.strftime(
        datetime.utcnow(), "%Y-%m-%d %X%Z"))
    # th = threading.Thread(target=run, args=())
    # th.setDaemon(True)
    # th.start()
    socketio.start_background_task(target=run)
    socketio.start_background_task(target=sync_dir)
    print("RUNS")
    # eventlet.spawn(run)
    socketio.run(app)
    # run_thread.join()
