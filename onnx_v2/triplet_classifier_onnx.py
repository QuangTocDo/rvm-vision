import cv2
import numpy as np
import onnxruntime as ort
# pyrefly: ignore [missing-import]
import faiss
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

class TripletClassifier:

    def __init__(self,
                 model_path=config.TRIPLET_MODEL_ONNX_PATH,
                 database_path=config.DATABASE_EMBEDDING_ONNX_PATH,
                 device=config.DEVICE):

        # ===== ONNX SESSION =====
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # ===== LOAD DATABASE V2 (Multiple files support) =====
        self.database = {}
        
        # Check if database_path is a directory or a single file
        if os.path.isdir(database_path):
             for file in os.listdir(database_path):
                 if file.endswith(".npy"):
                     brand_name = file.replace(".npy", "")
                     db_part = np.load(os.path.join(database_path, file), allow_pickle=True)
                     # Handle dictionary format or raw array
                     if isinstance(db_part, np.ndarray) and db_part.ndim > 1:
                         self.database[brand_name] = db_part
                     else:
                         db_part = db_part.item()
                         self.database.update(db_part)
        else:
             self.database = np.load(database_path, allow_pickle=True).item()

        # ===== COMPUTE CENTROIDS =====
        self.centroids = {}
        for k, v in self.database.items():
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            c = np.mean(v, axis=0)
            self.centroids[k] = c / np.linalg.norm(c)

        # ===== COMPUTE RADIUS =====
        self.radius = {}
        for k, v in self.database.items():
            v = v / np.linalg.norm(v, axis=1, keepdims=True)

            centroid = np.mean(v, axis=0)
            centroid = centroid / np.linalg.norm(centroid)

            sims = np.dot(v, centroid)
            dists = 1 - sims

            self.radius[k] = np.mean(dists) + 2 * np.std(dists)

        # ===== BUILD FAISS INDEX =====
        self.labels = list(self.centroids.keys())

        self.centroid_matrix = np.array(
            [self.centroids[k] for k in self.labels],
            dtype=np.float32
        )

        faiss.normalize_L2(self.centroid_matrix)

        self.index = faiss.IndexFlatIP(self.centroid_matrix.shape[1])
        self.index.add(self.centroid_matrix)

    # ===== PREPROCESS =====
    @staticmethod
    def preprocess(img):
        img = cv2.resize(img, config.IMAGE_SIZE)
        img = img[:, :, ::-1] / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    # ===== EMBEDDING =====
    def extract_batch(self, crops):
        imgs = [self.preprocess(c) for c in crops]
        imgs = np.concatenate(imgs, axis=0)

        outputs = self.session.run(None, {self.input_name: imgs})
        embs = outputs[0]  # (B, 128)

        # normalize embedding
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        return embs.astype(np.float32)

    # ===== PREDICT =====
    def predict_batch(self,
                      crops,
                      threshold=config.SIM_THRESHOLD,
                      margin_thresh=0.03):

        embs = self.extract_batch(crops)

        # normalize cho FAISS
        faiss.normalize_L2(embs)

        # search top2 để tính margin
        D, I = self.index.search(embs, k=2)

        outputs = []

        for i in range(len(embs)):

            top1_idx = I[i][0]
            top1_score = D[i][0]
            top1_brand = self.labels[top1_idx]

            if I.shape[1] > 1:
                top2_score = D[i][1]
            else:
                top2_score = 0

            margin = top1_score - top2_score

            # # ===== ƯU TIÊN NO BRAND =====
            # if top1_brand == "no_brand":
            #     outputs.append(("no_brand", float(top1_score)))
            #     continue

            # ===== FILTER 1 =====
            if top1_score < threshold:
                outputs.append(("unknown", float(top1_score)))
                continue

            # ===== FILTER 2 =====
            if margin < margin_thresh:
                outputs.append(("unknown", float(top1_score)))
                continue

            # ===== FILTER 3 =====
            centroid = self.centroids[top1_brand]
            dist = 1 - np.dot(embs[i], centroid)

            if dist > self.radius[top1_brand]:
                outputs.append(("unknown", float(top1_score)))
                continue

            # ===== PASS =====
            outputs.append((top1_brand, float(top1_score)))

        return outputs