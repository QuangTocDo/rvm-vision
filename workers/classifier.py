import threading
import queue
import time
import torch
import numpy as np
from collections import deque, Counter
import config
from onnx_v2.triplet_classifier import TripletClassifier

class ClassifierWorker(threading.Thread):
    def __init__(self, in_queue, out_queue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.daemon = True

        db_path = config.DATABASE_EMBEDDING_PC

        self.classifier = TripletClassifier(
            model_path=config.TRIPLET_MODEL_PATH,
            database_path=str(db_path)
        )
        
        # Tải database đặc biệt cho lớp 2 (Thủy tinh/Special)
        special_db_path = config.DATABASE_EMBEDDING_G
        self.special_classifier = TripletClassifier(
            model_path=config.TRIPLET_MODEL_PATH_GLASS,
            database_path=str(special_db_path)
        )

        self.track_embeddings = {}
        self.voting_history = {}
        self.VOTING_WINDOW = config.VOTING_WINDOW_CLASSIFIER
        self._stop_event = threading.Event()
        
        # Biến đo FPS và Latency cho luồng Classifier
        self.prev_time = time.time()
        self.fps = 0.0
        self.latency = 0.0

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
                    
                    # Tính FPS luồng Classifier
                    current_time = time.time()
                    time_diff = current_time - self.prev_time
                    self.prev_time = current_time
                    if time_diff > 0 and time_diff < 2.0:
                        instant_fps = 1.0 / time_diff
                        alpha = config.FPS_ALPHA_CLASSIFIER
                        self.fps = alpha * instant_fps + (1.0 - alpha) * self.fps
                    elif time_diff >= 2.0:
                        self.fps = 0.0

                    start_proc = time.time()
                    try:
                        # Lựa chọn classifier tương ứng dựa trên db_type (Standard vs Special/Glass)
                        active_classifier = self.special_classifier if db_type == "special" else self.classifier
                        
                        embs = active_classifier.extract_batch([crop])
                        if embs.size > 0:
                            emb = embs[0]
                            norm = np.linalg.norm(emb)
                            emb_norm = emb / norm if norm > 0 else emb

                            # Làm mượt embedding qua thời gian (EMA) để chống rung/flickering
                            if _id not in self.track_embeddings:
                                self.track_embeddings[_id] = emb_norm
                            else:
                                alpha = config.EMA_SMOOTHING_ALPHA  # Hệ số làm mượt
                                smoothed = alpha * self.track_embeddings[_id] + (1.0 - alpha) * emb_norm
                                smoothed_norm = np.linalg.norm(smoothed)
                                self.track_embeddings[_id] = smoothed / smoothed_norm if smoothed_norm > 0 else smoothed

                            # Phân loại dựa trên vector đã được làm mượt
                            preds = active_classifier.predict_embeddings([self.track_embeddings[_id]],
                                                                        threshold=config.SIM_THRESHOLD, 
                                                                        margin_thres=config.MARGIN_THRESHOLD,
                                                                        outlier_floor=config.OUTLIER_RADIUS_FLOOR)
                            pred_class, p_score = preds[0]

                            # Đưa qua bộ lọc bỏ phiếu Sliding Window để tối ưu hóa quyết định
                            v_class, v_score = self._get_voted_class(_id, pred_class, p_score)
                            self.out_queue.put((_id, v_class, v_score))
                    except Exception as e:
                        print(f"Classifier Error for ID {_id}: {e}")
                    
                    self.latency = (time.time() - start_proc) * 1000.0

                elif task_type == "cleanup":
                    _, active_ids = task
                    # Dọn dẹp cache của các track không hoạt động
                    for cache_dict in (self.track_embeddings, self.voting_history):
                        inactive_keys = cache_dict.keys() - active_ids
                        for k in inactive_keys:
                            if k in cache_dict:
                                del cache_dict[k]

            except queue.Empty:
                if time.time() - self.prev_time > 1.5:
                    self.fps = 0.0
                    self.latency = 0.0
                continue
            except Exception as e:
                print(f"ClassifierWorker Error: {e}")
