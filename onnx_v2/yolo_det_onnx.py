import cv2
import numpy as np
import onnxruntime as ort
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

class YOLODetONNX:
    def __init__(self,
                 model_path=config.YOLO_DET_MODEL_ONNX_PATH,
                 conf_thresh=config.YOLO_CONF,
                 input_size=config.INPUT_SIZE,
                 num_threads=(4,2)):

        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.input_size = input_size

        # ===== ONNX session =====
        os.environ["OMP_NUM_THREADS"] = str(num_threads[0])
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = num_threads[0]
        sess_options.inter_op_num_threads = num_threads[1]

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )

    # ================= PREPROCESS =================
    def letterbox(self, frame, new_shape=None, color=(114,114,114)):
        if new_shape is None:
            new_shape = self.input_size
        h, w = frame.shape[:2]
        r = min(new_shape[0]/h, new_shape[1]/w)
        new_unpad = (int(round(w*r)), int(round(h*r)))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        img = cv2.resize(frame, new_unpad)
        top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
        left, right = int(round(dw-0.1)), int(round(dw+0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        return img, r, dw, dh

    def preprocess(self, frame):
        img, r, dw, dh = self.letterbox(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[np.newaxis, :]
        return img, r, dw, dh


    # ================= DETECTION =================
    def detect(self, frame):
        img_tensor, r, dw, dh = self.preprocess(frame)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_tensor})
        preds = outputs[0][0]

        detections = []
        h, w = frame.shape[:2]

        for det in preds:
            x1, y1, x2, y2, score, cls_id = det
            if score < self.conf_thresh:
                continue

            # scale back
            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r

            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            detections.append({
                "box": [x1, y1, x2, y2],
                "score": float(score),
                "class_name": str(int(cls_id))
            })

        return detections