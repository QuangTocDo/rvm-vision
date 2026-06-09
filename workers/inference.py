import threading
import queue
import time
import cv2
import config
from tracker import Tracker
from ultralytics import YOLO

class InferenceWorker(threading.Thread):
    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True

        self.__path = config.YOLO_DET_MODEL_PATH
        self.detector = YOLO(self.__path)
        self.tracker = Tracker()
        
        self.ai_active = True
        self._stop_event = threading.Event()
        self.reset_tracker_flag = False
        
        # Biến đo FPS và Latency
        self.prev_time = time.time()
        self.fps = 0.0
        self.latency = 0.0

    def stop(self):
        self._stop_event.set()

    def request_tracker_reset(self):
        self.reset_tracker_flag = True

    def run(self):
        # We access CAMERAS as a global in the parent script, but we can load from config or fallback
        # Let's import CAMERAS from __main__ or use a fallback
        import __main__
        cameras = getattr(__main__, "CAMERAS", (-1, 0, 0, 640, 480))
        
        while not self._stop_event.is_set():
            try:
                data = self.in_queue.get(timeout=0.1)
                if data is None:
                    break
                frame_curr, current_detext, start_time = data
                
                if self.reset_tracker_flag:
                    self.tracker.reset()
                    self.reset_tracker_flag = False
                
                # Tính FPS luồng YOLO
                current_time = time.time()
                time_diff = current_time - self.prev_time
                self.prev_time = current_time
                if time_diff > 0 and time_diff < 1.0:
                    instant_fps = 1.0 / time_diff
                    alpha = config.FPS_ALPHA_YOLO
                    self.fps = alpha * instant_fps + (1.0 - alpha) * self.fps
                
                start_proc = time.time()
                frame_curr = cv2.flip(frame_curr, 1)
                
                valid_boxes = []
                
                # --- 1. CHẠY YOLO (.PT) ---
                results = self.detector(frame_curr, conf=config.YOLO_CONF_DET, agnostic_nms=True, iou=config.YOLO_IOU_DET, verbose=False)
                
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
                        
                        if x1 < cameras[1] or y1 < cameras[2]:
                            continue
                        
                        score = float(boxx.conf[0])
                        idx_class = int(boxx.cls[0])
                        
                        # Lọc các lớp chai lọ 0, 1, 2, 4 và lớp bàn tay 5
                        if idx_class not in [0, 1, 2, 4, 5]:
                            continue
                            
                        detections.append([x1, y1, x2, y2, idx_class, score])
                
                # --- 2. TRACKING ---
                self.tracker.update(frame_curr, detections)
                
                for track in self.tracker.tracks:
                    bbox = track.bbox
                    track_id = track.track_id
                    valid_boxes.append((bbox, track.id, track.confidence, track_id))
                
                if len(valid_boxes) > 0:
                    valid_boxes.sort(key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]), reverse=True)
                
                self.latency = (time.time() - start_proc) * 1000.0
                    
                if not self._stop_event.is_set():
                    self.out_queue.put((frame_curr, valid_boxes, start_time))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker Error: {e}")
