import torch
import cv2
import numpy as np
from .triplet_embedding import EmbeddingNet
import config

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class TripletClassifier:

    def __init__(self, model_path, database_path, device=config.DEVICE):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 1. Load embedding model
        self.model = EmbeddingNet(128).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 2. Load database (.npz)
        data = np.load(database_path)
        self.database = {k: data[k] for k in data.files}

        # 3. Tính toán Centroid & Radius dùng phân vị (Percentile 95%) & K-Means (Multi-Centroid)
        self.centroids = {}
        self.radius = {}
        
        self.all_embeddings = []
        self.all_labels = []

        for k, v in self.database.items():
            # Chuẩn hóa L2 các mẫu đặc trưng
            v_norm = v / np.linalg.norm(v, axis=1, keepdims=True)
            self.all_embeddings.append(v_norm)
            self.all_labels.extend([k] * len(v_norm))
            # Multi-Centroid sử dụng KMeans (Nếu có sklearn)
            if HAS_SKLEARN:
                try:
                    n_clusters = min(2, len(v_norm))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(v_norm)
                    # Chuẩn hóa L2 các tâm cụm
                    centers = kmeans.cluster_centers_
                    centers_norm = centers / np.linalg.norm(centers, axis=1, keepdims=True)
                    self.centroids[k] = centers_norm
                except Exception:
                    c = np.mean(v_norm, axis=0)
                    self.centroids[k] = np.expand_dims(c / np.linalg.norm(c), axis=0)
            else:
                # Fallback: Chỉ dùng 1 centroid toàn cục nhưng biểu diễn dưới dạng 2D array để đồng nhất logic
                c = np.mean(v_norm, axis=0)
                self.centroids[k] = np.expand_dims(c / np.linalg.norm(c), axis=0)

            # Tính toán bán kính Outlier dùng Percentile 95%
            # Đo khoảng cách từ tất cả các điểm của hãng tới tâm cụm gần nhất của hãng đó
            brand_centroids = self.centroids[k]  # Shape: (n_clusters, 128)
            sims = np.dot(v_norm, brand_centroids.T)  # Shape: (len(v_norm), n_clusters)
            max_sims = np.max(sims, axis=1)
            dists = 1.0 - max_sims
            self.radius[k] = float(np.percentile(dists, 95.0))

        # Concatenate tất cả để hỗ trợ k-NN retrieval nếu cần
        if len(self.all_embeddings) > 0:
            self.all_embeddings = np.concatenate(self.all_embeddings, axis=0).astype(np.float32)
        else:
            self.all_embeddings = np.empty((0, 128), dtype=np.float32)

    # ===== Kiểm tra ảnh bị nhòe (Motion Blur) =====
    def is_blurry(self, img, threshold=90.0):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            return fm < threshold
        except Exception:
            return False

    # ===== Tiền xử lý ảnh dạng NumPy =====
    @staticmethod
    def preprocess_numpy(img):
        img = cv2.resize(img, config.IMAGE_SIZE)
        img = img[:, :, ::-1] / 255.0  # BGR sang RGB và chuẩn hóa về [0, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC sang CHW
        return img

    # ===== Trích xuất đặc trưng tối ưu hóa Host-to-Device Copy =====
    def extract_batch(self, crops):
        if not crops:
            return np.empty((0, 128), dtype=np.float32)
            
        processed_imgs = [self.preprocess_numpy(c) for c in crops]
        batch_np = np.stack(processed_imgs, axis=0).astype(np.float32)

        # Đẩy cả batch lên GPU/MPS duy nhất 1 lần
        imgs_tensor = torch.from_numpy(batch_np).to(self.device)

        with torch.no_grad():
            embs = self.model(imgs_tensor).cpu().numpy()

        return embs

    # ===== Phân loại dựa trên vector Embedding (Đã làm mượt qua tracking) =====
    def predict_embeddings(self, embs, threshold=config.SIM_THRESHOLD, margin_thres=0.03, outlier_floor=config.OUTLIER_RADIUS_FLOOR):
        outputs = []
        for emb in embs:
            # L2 normalize để đảm bảo độ dài vector bằng 1
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb_norm = emb / norm
            else:
                emb_norm = emb

            # Tính độ tương đồng với tâm gần nhất của mỗi hãng (Multi-Centroid)
            scores = []
            for k, brand_centroids in self.centroids.items():
                brand_scores = np.dot(brand_centroids, emb_norm)
                max_score = float(np.max(brand_scores))
                scores.append((k, max_score))

            # Sắp xếp điểm số giảm dần
            scores.sort(key=lambda x: x[1], reverse=True)

            top1_brand, top1_score = scores[0]
            top2_score = scores[1][1] if len(scores) > 1 else 0.0
            margin = top1_score - top2_score

            # BỘ LỌC 1: Ngưỡng tự tin tối thiểu (Threshold)
            if top1_score < threshold:
                outputs.append(("unknown", top1_score))
                continue

            # BỘ LỌC 2: Đạt biên phân biệt giữa Top 1 và Top 2 (Margin)
            if margin < margin_thres:
                outputs.append(("unknown", top1_score))
                continue

            # BỘ LỌC 3: Nằm trong bán kính mẫu cho phép của hãng (Adaptive Floor Outlier Radius)
            dist = 1.0 - top1_score
            radius_limit = max(self.radius.get(top1_brand, 0.0), outlier_floor)
            if dist > radius_limit:
                outputs.append(("unknown", top1_score))
                continue

            # Thỏa mãn mọi bộ lọc
            outputs.append((top1_brand, top1_score))

        return outputs

    # ===== Dự đoán theo lô ảnh trực tiếp (Hỗ trợ lọc mờ) =====
    def predict_batch(self, crops, threshold=config.SIM_THRESHOLD, margin_thres=0.03, outlier_floor=config.OUTLIER_RADIUS_FLOOR):
        if not crops:
            return []

        # Khởi tạo danh sách kết quả với kích thước bằng crops
        outputs = [None] * len(crops)
        valid_crops = []
        valid_indices = []

        # Lọc ảnh nhòe / mờ do chuyển động trước khi trích xuất embedding
        for idx, crop in enumerate(crops):
            if self.is_blurry(crop):
                outputs[idx] = ("unknown", 0.0)
            else:
                valid_crops.append(crop)
                valid_indices.append(idx)

        # Chạy inference và phân loại cho các ảnh rõ nét
        if valid_crops:
            embs = self.extract_batch(valid_crops)
            preds = self.predict_embeddings(embs, threshold=threshold, margin_thres=margin_thres, outlier_floor=outlier_floor)
            for idx, pred in zip(valid_indices, preds):
                outputs[idx] = pred

        return outputs

    # Giữ lại predict_batch_v2 để đảm bảo tương thích ngược hoàn hảo
    def predict_batch_v2(self, crops, threshold=config.SIM_THRESHOLD, margin_thres=0.03, outlier_floor=config.OUTLIER_RADIUS_FLOOR):
        return self.predict_batch(crops, threshold=threshold, margin_thres=margin_thres, outlier_floor=outlier_floor)