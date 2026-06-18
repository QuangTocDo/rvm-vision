import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import threading
import queue
import numpy as np
# pyrefly: ignore [missing-import]
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# ── Load .env nếu có thư viện python-dotenv ──────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv là tùy chọn; dùng biến môi trường hệ thống

def setup_logger():
    logger = logging.getLogger("RVM_Vision")
    logger.setLevel(logging.INFO)
    
    log_file = "rvm_vision_system.log"
    handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger()

# ADD GLOBAL ROOT PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from s3_worker import init, sync, unix
from tracker import Tracker

from utils.camera_utils import FindCamera, VideoCapture
from utils.helper_functions import crop_from_box
from utils.hierarchy import evaluate_hierarchical_class, estimate_volume
from utils.roi import ROI
from utils.types import DetectionBox
from utils.result_logger import ResultLogger
from core.state_machine import DetectionStateMachine
from workers.inference import InferenceWorker
from workers.classifier import ClassifierWorker
from collections import deque

HOME = Path.home()
mac_add = "rvm"
__env = HOME / ".env"
_camera_env = HOME / ".camera.env"
CODE = "8.5.0"
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

if os.path.exists(_camera_env):
    CAMERAS = _parse_camera_env(str(_camera_env))
elif os.path.exists(".env.camera"):
    CAMERAS = _parse_camera_env(".env.camera")
    _write_camera_env(str(_camera_env), CAMERAS)
    os.unlink(".env.camera")
else:
    _write_camera_env(str(_camera_env), CAMERAS)

if not mac_add.startswith("r-"):
    init()
    mac_add = unix()
    with open(__env, 'wt') as file:
        file.write(mac_add)

# ── Flask / SocketIO ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, async_mode=None)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ── Globals ────────────────────────────────────────────────────────────────
app_start_time = time.time()
beginTime      = 0
flag_camera    = False
flag_camera_start_time = None

# Thư mục tạm hàng ngày
_dir = "./temp/" + datetime.strftime(datetime.utcnow(), "%Y-%m-%d")
if not os.path.exists(_dir):
    os.makedirs(_dir)

# ── Khởi tạo các tiện ích dùng chung ──────────────────────────────────────
roi       = ROI.from_config(config.ROI_COORDS)
sm        = DetectionStateMachine()   # State machine chính
rlog      = ResultLogger("results.jsonl")
cache_lock = threading.Lock()
camera     = None
worker     = None
cls_worker = None

# ── Transform dict (do UI cập nhật qua SocketIO) ──────────────────────────
_transform_lock = threading.Lock()
_transform_dict = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2}

def transform_id(id_detect):
    with _transform_lock:
        d = _transform_dict
    if d is None:
        return id_detect
    return d.get(str(id_detect), id_detect)


# ── Throttled SocketIO emit ───────────────────────────────────────────────
emit_times: dict = {}
emit_lock  = threading.Lock()

def global_emit(event, data):
    global emit_times
    try:
        with emit_lock:
            lastTime = emit_times.get(event, 0)
            if time.time() - lastTime < 0.5 and event != "result":
                return
            emit_times[event] = time.time()

        with app.test_request_context('/'):
            emit(event, data, broadcast=True, namespace="/")

        logger.info(f"emit {event} {data} {time.time()}")
    except Exception as e:
        logger.error(f"Error in global_emit: {e}", exc_info=True)


# ── S3 Sync background task ────────────────────────────────────────────────
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
            _delta_seconds = datetime(year=__date.year, month=__date.month,
                                      day=__date.day, hour=20, minute=1,
                                      second=0) - __date
            socketio.sleep(_delta_seconds.seconds)
        else:
            socketio.sleep(seconds)


# ── Graceful Shutdown ──────────────────────────────────────────────────────
def _shutdown(signum, frame):
    logger.info(f"Nhận tín hiệu {signum}. Đang tắt hệ thống...")
    global worker, cls_worker, camera
    if worker is not None:
        worker.stop()
        logger.info("InferenceWorker đã dừng.")
    if cls_worker is not None:
        cls_worker.stop()
        logger.info("ClassifierWorker đã dừng.")
    if camera is not None:
        camera.release()
        logger.info("Camera đã được giải phóng.")
    # Flush pending S3 uploads
    try:
        sync(mac_add, datetime.strftime(datetime.utcnow(), "%Y-%m-%d"))
        logger.info("S3 sync hoàn tất.")
    except Exception as e:
        logger.warning(f"S3 sync khi shutdown thất bại: {e}")
    sys.exit(0)

signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)


# ── Main processing loop ───────────────────────────────────────────────────
def run():
    global flag_camera, flag_camera_start_time, camera, worker, cls_worker

    __path = config.YOLO_DET_MODEL_PATH
    logger.info(f"INIT AI CORE WITH HIERARCHICAL ROUTING V2 {__path}")

    # ── Khởi tạo camera ──────────────────────────────────────────────────
    camera_ii = CAMERAS[0]
    if camera_ii < 0:
        camera_ii = FindCamera()
        if camera_ii < 0:
            logger.error("Không thể mở camera")
            return
    camera = VideoCapture(camera_ii)

    # ── Khởi tạo worker threads ───────────────────────────────────────────
    in_queue  = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=2)
    worker    = InferenceWorker(in_queue, out_queue)
    worker.start()

    cls_in_queue  = queue.Queue(maxsize=4)
    cls_out_queue = queue.Queue(maxsize=4)
    cls_worker    = ClassifierWorker(cls_in_queue, cls_out_queue)
    cls_worker.start()

    # ── Per-frame caches ──────────────────────────────────────────────────
    caches_ids = []
    frameCount = 0
    final_result = 0

    volume_cache    : dict = {}
    track_age_vol   : dict = {}
    track_cache     : dict = {}
    track_age_cls   : dict = {}
    track_history   : dict = {}
    triggered_ids   : set  = set()
    roi_sliding_windows: dict = {}
    SLIDING_WINDOW_SIZE = config.SLIDING_WINDOW_ROI

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        # --- Đọc frame từ camera ---
        if camera is not None:
            ret, im = camera.read()
        else:
            ret, im = False, None
        if not ret:
            socketio.sleep(0.1)
            continue

        next_start_time = time.time()

        try:
            processed_frame, valid_boxes, start_time = out_queue.get(timeout=0.01)

            # ── Đọc kết quả phân loại từ ClassifierWorker ────────────────
            while not cls_out_queue.empty():
                try:
                    res_id, res_class, res_score = cls_out_queue.get_nowait()
                    with cache_lock:
                        track_cache[res_id] = (res_class, res_score)
                except queue.Empty:
                    break

            # ── Dọn cache cho các track không còn active ─────────────────
            active_ids = {box[3] for box in valid_boxes}
            with cache_lock:
                for cache_dict in (volume_cache, track_age_vol, track_cache,
                                   track_age_cls, roi_sliding_windows, track_history):
                    for k in list(cache_dict.keys() - active_ids):
                        cache_dict.pop(k, None)
                triggered_ids &= active_ids

            try:
                cls_in_queue.put(("cleanup", active_ids), block=False)
            except queue.Full:
                pass

            # ── Xử lý từng bounding box trong ROI ────────────────────────
            detected_in_roi_this_frame = {_id: False for _id in active_ids}
            classify_tasks = []

            with cache_lock:
                for box in valid_boxes:
                    bbox, idx_class, conf, _id = box
                    if idx_class not in [config.YOLOClass.CAN, config.YOLOClass.PLASTIC,
                                         config.YOLOClass.GLASS, config.YOLOClass.PLASTIC_OTHER]:
                        continue

                    if not roi.contains_center(bbox):
                        continue

                    detected_in_roi_this_frame[_id] = True

                    # 1. Đo thể tích
                    track_age_vol[_id] = track_age_vol.get(_id, 0) + 1
                    if track_age_vol[_id] == 1 or track_age_vol[_id] % config.CLASSIFY_INTERVAL == 0:
                        x1, y1, x2, y2 = map(int, bbox)
                        w_box, h_box = x2 - x1, y2 - y1
                        length_px, diameter_px = max(w_box, h_box), min(w_box, h_box)
                        if length_px < 300:
                            k = config.VOLUME_SCALING_UP
                        elif length_px > 700:
                            k = config.VOLUME_SCALING_DOWN
                        else:
                            k = 1
                        volume_cache[_id] = estimate_volume(
                            length_px, diameter_px, config.PIXEL_TO_CM_RATIO, k)

                    # 2. Nhận diện thương hiệu (nếu volume hợp lệ)
                    current_vol = volume_cache.get(_id, -1)
                    vol_valid = config.MIN_ACCEPTABLE_VOLUME < current_vol < config.MAX_ACCEPTABLE_VOLUME
                    if vol_valid and idx_class in [config.YOLOClass.CAN, config.YOLOClass.PLASTIC,
                                                   config.YOLOClass.GLASS, config.YOLOClass.PLASTIC_OTHER]:
                        track_age_cls[_id] = track_age_cls.get(_id, 0) + 1
                        if track_age_cls[_id] == 1 or track_age_cls[_id] % config.CLASSIFY_INTERVAL == 0:
                            crop = crop_from_box(processed_frame, bbox)
                            if crop is not None and crop.size > 0:
                                db_type = "special" if idx_class == config.YOLOClass.GLASS else "standard"
                                classify_tasks.append((_id, crop, db_type))

                # Cập nhật sliding window ROI cho tất cả active track
                for _id in active_ids:
                    if _id not in roi_sliding_windows:
                        roi_sliding_windows[_id] = deque(maxlen=SLIDING_WINDOW_SIZE)
                    roi_sliding_windows[_id].append(
                        1 if detected_in_roi_this_frame.get(_id, False) else 0)

            # Đẩy classify tasks ngoài lock
            for t_id, crop, db_type in classify_tasks:
                try:
                    cls_in_queue.put(("classify", t_id, crop, db_type), block=False)
                except queue.Full:
                    logger.warning("Classifier Queue đầy, bỏ qua frame phân loại.")

            # ── Kiểm tra tay trong ROI ────────────────────────────────────
            is_hand_in_roi = any(
                box[1] == config.YOLOClass.HAND and roi.contains_center(box[0])
                for box in valid_boxes
            )

            # ── Lọc boxes theo volume ─────────────────────────────────────
            decision_boxes = []
            with cache_lock:
                for box in valid_boxes:
                    bbox, idx_class, conf, _id = box
                    if idx_class == config.YOLOClass.HAND:
                        continue
                    if sm.is_detecting and _id == sm.target_id:
                        decision_boxes.append(box)
                        continue
                    vol = volume_cache.get(_id, -1)
                    if vol != -1 and not (config.MIN_ACCEPTABLE_VOLUME < vol < config.MAX_ACCEPTABLE_VOLUME):
                        continue
                    decision_boxes.append(box)

            len_decision_boxes = len(decision_boxes)

            is_object_in_roi = any(
                roi.contains_center(box[0]) for box in decision_boxes
            )

            # ── Cập nhật quỹ đạo và kiểm tra virtual line crossing ───────
            trigger_this_frame    = False
            triggered_id          = -1
            is_trigger_invalid_vol = False
            invalid_volume_val    = -1.0

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

                    is_above_now   = roi.is_above_virtual_line(cy)
                    is_bbox_ok     = roi.is_bbox_complete(y2)
                    track_age      = track_age_vol.get(_id, 0)

                    if is_above_now and is_bbox_ok and (_id not in triggered_ids) and track_age >= 5:
                        vol = volume_cache.get(_id, -1)
                        if vol != -1:
                            if not (config.MIN_ACCEPTABLE_VOLUME < vol < config.MAX_ACCEPTABLE_VOLUME):
                                is_trigger_invalid_vol = True
                                invalid_volume_val     = vol
                            trigger_this_frame = True
                            triggered_id       = _id

            # ── Xử lý trigger (ARMED + vật vượt vạch) ────────────────────
            if sm.is_armed and trigger_this_frame and not sm.is_detecting and not is_hand_in_roi:
                if is_trigger_invalid_vol:
                    result = sm.reject_immediately(
                        volume=invalid_volume_val, triggered_id=triggered_id,
                        reason="Volume out of bounds")
                    global_emit('result', {
                        'data': result.data, 'model': str(__path), 'ver': CODE,
                        "id": mac_add, "images": result.images,
                        "size": result.size, "item": triggered_id,
                        "volume": float(invalid_volume_val)
                    })
                    rlog.log_result(data=result.data, mac_id=mac_add,
                                    images=result.images, size=result.size,
                                    item=triggered_id, volume=invalid_volume_val,
                                    reason=result.reason)
                    triggered_ids.add(triggered_id)
                else:
                    if sm.trigger(triggered_id):
                        triggered_ids.add(triggered_id)
                        with cache_lock:
                            track_cache.pop(triggered_id, None)
                            track_age_cls[triggered_id] = 0
                        global_emit('auto_trigger', {'triggered': True})

            frameCount += 1
            shape = processed_frame.shape

            if len_decision_boxes > 0:
                if len_decision_boxes > 1 and sm.is_detecting:
                    global_emit('command', 2)
            else:
                global_emit('command', 0)

            # ── Hủy khẩn cấp khi phát hiện tay ───────────────────────────
            if sm.is_detecting and is_hand_in_roi:
                logger.info("Phát hiện tay trong ROI! Hủy detection ngay lập tức!")
                with cache_lock:
                    vol_val = float(volume_cache.get(sm.target_id, -1))
                result = sm.abort("Hand safety abort")
                result.volume = vol_val
                global_emit('result', {
                    'data': result.data, 'model': str(__path), 'ver': CODE,
                    "id": mac_add, "images": result.images, "size": result.size,
                    "item": result.item, "volume": result.volume
                })
                rlog.log_result(data=result.data, mac_id=mac_add,
                                images=result.images, size=result.size,
                                item=result.item, volume=result.volume,
                                reason=result.reason)
                global_emit('command', 0)
                flag_camera = False

            elif sm.is_detecting:
                if frameCount % 4 != 0 or frameCount < 4:
                    pass
                else:
                    if sm.should_finalize():
                        with cache_lock:
                            result = sm.finalize(volume_cache)
                        global_emit('result', {
                            'data': result.data, 'model': str(__path), 'ver': CODE,
                            "id": mac_add, "images": result.images,
                            "size": result.size, "item": result.item,
                            "volume": result.volume
                        })
                        rlog.log_result(data=result.data, mac_id=mac_add,
                                        images=result.images, size=result.size,
                                        item=result.item, volume=result.volume)
                        flag_camera = False
                    else:
                        if len_decision_boxes > 0:
                            box       = decision_boxes[0]
                            x1, y1, x2, y2 = box[0]
                            _id       = box[3]
                            idx_class = box[1]
                            size_val  = max((y2 - y1) / shape[0] * 100,
                                           (x2 - x1) / shape[1] * 100)

                            # Hierarchical Decision Logic V2
                            with cache_lock:
                                final_class, is_unknown_class2 = evaluate_hierarchical_class(
                                    box, volume_cache, track_cache, config.ROI_COORDS)

                            if is_unknown_class2:
                                logger.info("Unknown brand (class 2). Kết thúc detection ngay!")
                                with cache_lock:
                                    vol_val  = float(volume_cache.get(sm.target_id, -1))
                                result = sm.abort("Unknown brand class 2")
                                result.volume = vol_val
                                # Override data sang REJECT
                                global_emit('result', {
                                    'data': config.RVMClass.REJECT, 'model': str(__path), 'ver': CODE,
                                    "id": mac_add, "images": result.images,
                                    "size": result.size, "item": result.item,
                                    "volume": result.volume
                                })
                                rlog.log_result(data=config.RVMClass.REJECT, mac_id=mac_add,
                                                images=result.images, size=result.size,
                                                item=result.item, volume=result.volume,
                                                reason=result.reason)
                                global_emit('command', 0)
                                flag_camera = False
                            else:
                                sm.add_sample(transform_id(final_class), size_val)
                        else:
                            global_emit('command', 0)

            elif flag_camera:
                if frameCount % 3 == 0 and len_decision_boxes > 0:
                    _id = decision_boxes[0][3]
                    if _id < 1 or (_id not in caches_ids):
                        final_result += 1
                    if final_result > 4:
                        global_emit('detect', {
                            'data': True, 'model': str(__path), 'ver': CODE, "id": mac_add})
                        final_result = 0
                        flag_camera  = False
                        caches_ids.append(_id)
                        frameCount   = 0
                elif frameCount % 6 == 0:
                    final_result = max(0, final_result - 1)
            else:
                final_result = 0
                if len(caches_ids) > 500:
                    caches_ids = []
                if frameCount % 1000 == 0:
                    worker.request_tracker_reset()

        except queue.Empty:
            pass

        try:
            in_queue.put((im, sm.is_detecting, next_start_time), block=False)
        except queue.Full:
            pass

        if frameCount > 99999:
            frameCount = 0
        socketio.sleep(0.01)


# ── SocketIO events ────────────────────────────────────────────────────────
@socketio.on('connect')
def connect(data):
    logger.info(f"Client connected with data: {data}")


@socketio.event
def my_event(message):
    logger.info(f"my_event message: {message}")
    emit('my_response', {'data': message["data"]}, broadcast=True)


@socketio.event
def run_detect(data):
    global flag_camera, flag_camera_start_time, beginTime
    global _transform_dict

    logger.info(f"run_detect: {data} | time since begin: {time.time() - beginTime:.4f}s")
    try:
        with open("ui_payload_log.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} | Received UI Payload: {data}\n")
    except Exception as e:
        logger.error(f"Failed to log UI payload: {e}")

    if "transform" in data:
        with _transform_lock:
            _transform_dict = data["transform"]
        logger.info(f"Updated transform dict: {_transform_dict} | transform(4)={transform_id(4)}")

    if "detext" in data:
        if data["detext"]:
            if sm.arm():
                logger.info("Armed for detection.")
        else:
            sm.cancel()
            logger.info("Detection cancelled by user.")

    if "camera" in data:
        flag_camera = data["camera"]
        if flag_camera:
            flag_camera_start_time = time.time()
        else:
            flag_camera_start_time = None


# ── HTTP routes ────────────────────────────────────────────────────────────
@app.route('/camera')
def camera_view():
    return render_template('index.html')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint — dùng cho monitoring, load balancer, uptime robot."""
    _worker_fps  = worker.fps  if worker  else 0.0
    _cls_fps     = cls_worker.fps if cls_worker else 0.0
    return jsonify({
        "status":      "ok",
        "version":     CODE,
        "mac_id":      mac_add,
        "uptime_sec":  round(time.time() - app_start_time, 1),
        "camera_open": camera is not None,
        "state":       sm.state.name,
        "yolo_fps":    round(_worker_fps,  1),
        "cls_fps":     round(_cls_fps,     1),
        "yolo_latency_ms": round(worker.latency  if worker  else 0.0, 1),
        "cls_latency_ms":  round(cls_worker.latency if cls_worker else 0.0, 1),
    })


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"OPEN CAMERA | mac: {mac_add} | config: {CAMERAS} | "
                f"{datetime.strftime(datetime.utcnow(), '%Y-%m-%d %X%Z')}")
    socketio.start_background_task(target=run)
    socketio.start_background_task(target=sync_dir)
    logger.info("RUNS")
    socketio.run(app)
