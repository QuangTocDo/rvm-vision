import os
import sys
import zlib
import time
import cv2
from pathlib import Path
from ultralytics import YOLO
from tracker import Tracker
from config import get_model_path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
HOME = Path.home()  # relative
_camera_env = HOME/".camera.env"
# print(ROOT, len(FILE.parents))
_camera_env_old = ROOT / ".env.camera"

# tracker = ROOT / "botsort.yaml"
CAMERAS = (0, 0, 0, 640, 480)
print(_camera_env)
if os.path.exists(_camera_env):
    with open(_camera_env, 'rt') as file:
        size = file.readline()
        info = size.split(",")
        CAMERAS = (int(info[0]), int(info[1]), int(info[2]), int(info[3]), int(info[4]))

elif os.path.exists(_camera_env_old):
    with open(_camera_env_old, 'rt') as file:
        size = file.readline()
        info = size.split(",")
        CAMERAS = (int(info[0]), int(info[1]), int(info[2]), int(info[3]), int(info[4]))
    
    with open(_camera_env, 'wt') as file:
        file.write(str(CAMERAS[0]) + "," + str(CAMERAS[1]) + "," +
                   str(CAMERAS[2]) + "," + str(CAMERAS[3]) + "," + str(CAMERAS[4]))
    os.unlink(_camera_env_old)
else:
    with open(_camera_env, 'wt') as file:
        file.write(str(CAMERAS[0]) + "," + str(CAMERAS[1]) + "," +
                   str(CAMERAS[2]) + "," + str(CAMERAS[3]) + "," + str(CAMERAS[4]))



def open_camera(src):
    global CAMERAS
    weights = get_model_path()
    camera = cv2.VideoCapture(src)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    model = YOLO(weights)
    colors = [(255, 0, 0), (0, 128, 0), (0, 0, 200), (120, 120, 120), (180, 140, 180)]
    len_colr = len(colors)
    tracker = Tracker()
    f=0
    while True:
        ret, im = camera.read()
        f+=1
        if f%200==0:
            with open(_camera_env, 'rt') as file:
                size = file.readline()
                info = size.split(",")
                CAMERAS = (int(info[0]), int(info[1]), int(info[2]), int(info[3]), int(info[4]))
                cv2.destroyAllWindows()
        if ret:
            img = im[CAMERAS[2]:CAMERAS[4], CAMERAS[1]:CAMERAS[3], :]
            # results=model(img,conf=0.5,agnostic_nms=True, iou=0.4)
            results = model(img, conf=0.75, agnostic_nms=True, iou=0.4, verbose=False)
            valid_boxes = []
            detections = []
            if results[0].boxes.shape[0] > 0:
                for boxx in results[0].boxes:
                    a = boxx.xyxy
                    a = a.cpu().detach().numpy()
                    x1 = int(a[0, 0])
                    y1 = int(a[0, 1])
                    x2 = int(a[0, 2])
                    y2 = int(a[0, 3])
                    score = float(boxx.conf.cpu().detach().numpy())
                    idx_class = int(boxx.cls.cpu().detach().numpy())
                    detections.append([x1, y1, x2, y2, idx_class, score])
                    
            tracker.update(img, detections)
            for track in tracker.tracks:
                bbox = track.bbox
                track_id = track.track_id
                print(track_id, "track_id", track.id)
                valid_boxes.append((bbox, track.id, track.confidence, track_id))
            
            for box in valid_boxes:
                idx_class = box[1]
                x1, y1, x2, y2 = box[0]
                conf = box[2]
                id = box[3]
                print(id,conf)
                cv2.rectangle(img, (x1, y1), (x2, y2), colors[idx_class % len_colr], 2, cv2.LINE_AA)
                cv2.putText(img, str(idx_class) + " | " + str(int(conf * 100) / float(100)) + " | " + str(id),
                            (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx_class % len_colr],)
            cv2.imshow("CAMERA " + str(src)+" | "+str(CAMERAS[3])+"x"+str(CAMERAS[4]), img)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    camera.release()


if __name__ == "__main__":
    open_camera(0)
