import cv2
import queue
import threading
import time
import logging
import config

logger = logging.getLogger("RVM_Vision")

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

class VideoCapture:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.cache_lock = threading.Lock()
        
        self.open_camera_source()
        
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()

    def open_camera_source(self):
        with self.cache_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            
            camera_idx = self.src
            if camera_idx < 0:
                logger.info("Camera source index is negative, scanning dynamically...")
                camera_idx = FindCamera()
                
            if camera_idx >= 0:
                logger.info(f"Opening camera source index: {camera_idx}")
                self.cap = cv2.VideoCapture(camera_idx)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            else:
                logger.error("No valid camera source found.")
                self.cap = None

    def _reader(self):
        while self.running:
            with self.cache_lock:
                cap_ok = self.cap is not None and self.cap.isOpened()
            
            if not cap_ok:
                time.sleep(2.0)
                self.open_camera_source()
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Camera frame read failed. Reopening in 2 seconds...")
                time.sleep(2.0)
                self.open_camera_source()
                continue

            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return True, self.q.get(timeout=0.1)
        except queue.Empty:
            return False, None

    def release(self):
        self.running = False
        with self.cache_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

