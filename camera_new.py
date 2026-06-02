import os
import sys
import zlib
import time
import cv2
from pathlib import Path
from tracker import Tracker
from config import get_model_path

# ADD GLOBAL ROOT PATH
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# IMPORT CORE AI TỪ onnx_v2
from onnx_v2.yolo_det_onnx import YOLODetONNX

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
HOME = Path.home()  # relative
_camera_env = HOME / ".camera.env"
# print(ROOT, len(FILE.parents))
_camera_env_old = ROOT / ".env.camera"

# tracker = ROOT / "botsort.yaml"
CAMERAS = (-1, 10, 10, 640, 480)
print(_camera_env)
if os.path.exists(_camera_env):
    with open(_camera_env, 'rt') as file:
        size = file.readline()
        info = size.split(",")
        CAMERAS = (int(info[0]), max(int(info[1]), 0), max(int(info[2]), 0), int(info[3]), int(info[4]))

elif os.path.exists(_camera_env_old):
    with open(_camera_env_old, 'rt') as file:
        size = file.readline()
        info = size.split(",")
        CAMERAS = (int(info[0]), max(int(info[1]), 0), max(int(info[2]), 0), int(info[3]), int(info[4]))

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
    
    print("INIT NEW ONNX CORE AI (DETECTION ONLY)")
    detector = YOLODetONNX()

    camera = cv2.VideoCapture(src)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    colors = [(255, 0, 0), (0, 128, 0), (0, 0, 200),
              (120, 120, 120), (180, 140, 180)]
    len_colr = len(colors)
    tracker = Tracker()
    f = 0

    # Initialize variables for FPS calculation
    prev_time = time.time()
    curr_time = 0
    
    while True:
        ret, im = camera.read()
        f += 1
        if f % 200 == 0:
            with open(_camera_env, 'rt') as file:
                size = file.readline()
                info = size.split(",")
                CAMERAS = (int(info[0]), int(info[1]), int(
                    info[2]), int(info[3]), int(info[4]))
                cv2.destroyAllWindows()
        if ret:
            # im = cv2.flip(im, 1)
            img = im

            # Calculate FPS
            curr_time = time.time()
            time_diff = curr_time - prev_time
            fps = 1 / time_diff if time_diff > 0 else 0
            prev_time = curr_time

            # --- DETECT BẰNG CORE AI MỚI ---
            raw_detections = detector.detect(img)
            
            valid_boxes = []
            detections = []
            
            for det in raw_detections:
                x1, y1, x2, y2 = det["box"]
                score = det["score"]
                idx_class = int(det["class_name"])
                detections.append([x1, y1, x2, y2, idx_class, score])

            # --- TRACKING ---
            tracker.update(img, detections)
            for track in tracker.tracks:
                bbox = track.bbox
                track_id = track.track_id
                print(track_id, "track_id", track.id)
                valid_boxes.append(
                    (bbox, track.id, track.confidence, track_id))
                    
            cv2.rectangle(img, (CAMERAS[1], CAMERAS[2]), (CAMERAS[3],
                                                          CAMERAS[4]), (255, 255, 255), 2, cv2.LINE_AA)
            for box in valid_boxes:
                idx_class = box[1]
                x1, y1, x2, y2 = box[0]
                conf = box[2]
                id = box[3]
                
                print(id, conf)
                color = colors[idx_class % len_colr]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              color, 2, cv2.LINE_AA)
                              
                # In thông tin RVM class + conf + id (GIỐNG HỆT BẢN GỐC)
                cv2.putText(img, str(idx_class) + " | " + str(int(conf * 100) / float(100)) + " | " + str(id),
                            (int(x1) + 10, int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

            # Display FPS on image
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("CAMERA " + str(src) + " | " +
                       str(CAMERAS[3]) + "x" + str(CAMERAS[4]), img)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    camera.release()


def FindCamera():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


if __name__ == "__main__":
    print(CAMERAS)
    # index = CAMERAS[0]
    index = 0
    if index < 0:
        cameras = FindCamera()
        if len(cameras) == 0:
            print("can not open camera")
        print(cameras)
        index = cameras[0]
    open_camera(index)