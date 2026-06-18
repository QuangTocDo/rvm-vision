import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Thêm thư mục gốc vào PATH để Python tìm thấy các import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from onnx_v2.triplet_classifier import TripletClassifier

# ===== CẤU HÌNH ĐƯỜNG DẪN TEST =====
IMAGE_DIR = "/Users/trannhutquang/PycharmProjects/PlasticCanCls/rvm-vision/videos"


def main():
    print("=" * 60)
    print("CHƯƠNG TRÌNH KIỂM THỬ ĐỘ TƯƠNG ĐỒNG COSINE (BỎ QUA BỘ LỌC 2 & 3)")
    print("=" * 60)
    
    # 1. Khởi tạo TripletClassifier từ model và database embedding
    db_path = getattr(config, "DATABASE_EMBEDDING_G", None)
    if db_path is None or not os.path.exists(db_path):
        db_path = getattr(config, "DATABASE_EMBEDDING_ONNX_PATH", "")

    print(f"Đang tải Triplet Model: {config.TRIPLET_MODEL_PATH}")
    print(f"Đang tải Database: {db_path}")
    
    try:
        classifier = TripletClassifier(
            model_path=config.TRIPLET_MODEL_PATH,
            database_path=str(db_path)
        )
        print("Tải dữ liệu thành công!\n")
    except Exception as e:
        print(f"Lỗi khởi tạo TripletClassifier: {e}")
        return

    # 2. Khởi tạo & Huấn luyện One-Class SVM cho từng brand trong Database
    from sklearn.svm import OneClassSVM
    oc_svms = {}
    print("Đang huấn luyện các mô hình One-Class SVM Anomaly Detector (Jittered Soft RBF)...")
    for brand, embeddings in classifier.database.items():
        # L2 Normalize
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Áp dụng Jittering: Thêm nhiễu Gaussian cực nhỏ (std = 1e-4) để phá vỡ tính suy biến ma trận,
        # giúp thư viện libsvm giải bài toán tối ưu ổn định số học mà không làm mất đặc trưng gốc.
        np.random.seed(42) # Cố định seed để kết quả tái lập được
        noise = np.random.normal(0, 1e-4, embeddings_norm.shape)
        embeddings_norm_jittered = embeddings_norm + noise
        
        # Sử dụng gamma=0.01 kết hợp nu=0.005 để tạo biên mềm rộng rãi
        svm = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.005)
        svm.fit(embeddings_norm_jittered)
        oc_svms[brand] = svm
    print("Huấn luyện One-Class SVM thành công!\n")

    if not os.path.exists(IMAGE_DIR):
        print(f"Thư mục ảnh test không tồn tại: {IMAGE_DIR}")
        return

    # Lấy danh sách ảnh trong thư mục test
    img_names = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not img_names:
        print(f"Không tìm thấy ảnh test nào trong thư mục: {IMAGE_DIR}")
        return

    print(f"Tìm thấy {len(img_names)} ảnh. Tiến hành quét độ tương đồng...")
    print("-" * 80)

    for idx, img_name in enumerate(img_names):
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Không thể đọc ảnh: {img_name}")
            continue

        # --- BƯỚC A: TRÍCH XUẤT EMBEDDING ---
        # Hàm extract_batch đã bao gồm tiền xử lý (Resize 224x224, chuyển RGB, CHW, chia 255.0)
        embs = classifier.extract_batch([img])
        if embs.size == 0:
            print(f"[{img_name}] Lỗi không trích xuất được đặc trưng.")
            continue
        
        # --- BƯỚC B: CHUẨN HÓA L2 & PHÂN LOẠI ---
        # Gọi directly predict_embeddings để đồng nhất logic với pipeline thực tế
        preds = classifier.predict_embeddings(embs, threshold=config.SIM_THRESHOLD, margin_thres=config.MARGIN_THRESHOLD)
        final_decision, _ = preds[0]

        # Tính toán các chỉ số chi tiết cho báo cáo HUD
        emb = embs[0]
        norm = np.linalg.norm(emb)
        emb_norm = emb / norm if norm > 0 else emb

        brand_scores = []
        for brand, brand_centroids in classifier.centroids.items():
            similarities = np.dot(brand_centroids, emb_norm)
            max_sim = float(np.max(similarities))
            brand_scores.append((brand, max_sim))

        brand_scores.sort(key=lambda x: x[1], reverse=True)

        top1_brand, top1_score = brand_scores[0]
        top2_brand, top2_score = brand_scores[1] if len(brand_scores) > 1 else ("N/A", 0.0)
        
        margin = top1_score - top2_score
        dist = 1.0 - top1_score
        
        calculated_radius = classifier.radius.get(top1_brand, 0.0)
        # Sử dụng Adaptive Floor Outlier Radius
        radius_limit = max(calculated_radius, config.OUTLIER_RADIUS_FLOOR)

        # Kiểm tra điều kiện vượt qua bộ lọc hình học (Threshold + Margin + Radius)
        pass_threshold = top1_score >= config.SIM_THRESHOLD
        pass_margin = margin >= config.MARGIN_THRESHOLD
        pass_radius = dist <= radius_limit

        status_sim = "PASS" if pass_threshold else f"FAIL (Yêu cầu >= {config.SIM_THRESHOLD:.2f})"
        status_margin = "PASS" if pass_margin else f"FAIL (Yêu cầu >= {config.MARGIN_THRESHOLD:.2f})"
        status_radius = f"PASS (Sàn: {config.OUTLIER_RADIUS_FLOOR:.4f}, Tính: {calculated_radius:.4f})" if pass_radius else f"FAIL (Vượt giới hạn {radius_limit:.4f})"

        # --- BƯỚC C: KIỂM TRA BẰNG ONE-CLASS SVM (ANOMALY DETECTION) ---
        svm_model = oc_svms.get(top1_brand)
        if svm_model is not None:
            # predict() trả về 1 cho Inlier (Normal), -1 cho Outlier (Anomaly)
            svm_pred = svm_model.predict([emb_norm])[0]
            pass_svm = (svm_pred == 1)
        else:
            pass_svm = False
            
        status_svm = "🟢 PASS (ĐỒNG NHẤT / INLIER)" if pass_svm else "🔴 FAIL (BẤT THƯỜNG / ANOMALY)"

        # Kết luận cuối cùng kết hợp One-Class SVM thay cho Radius thủ công
        final_decision_svm = "UNKNOWN"
        if pass_threshold and pass_margin and pass_svm:
            final_decision_svm = top1_brand

        print(f"\n[{idx + 1}] 📷 FILE: {img_name}")
        print(f"    ⭐ TOP 1 DỰ ĐOÁN (Chỉ dựa trên Cosine): {top1_brand.upper()} với Score: {top1_score:.4f} (Threshold: {status_sim})")
        print(f"    🥈 TOP 2 DỰ ĐOÁN: {top2_brand.upper()} với Score: {top2_score:.4f}")
        print(f"    📐 Margin (Top1 - Top2): {margin:.4f} (Margin Check: {status_margin})")
        print(f"    🎯 Lọc thô (Outlier Distance): {dist:.4f} (Radius Limit: {radius_limit:.4f}) (Radius Check: {status_radius})")
        print(f"    🧠 Lọc tinh (One-Class SVM   ): {status_svm}")
        print(f"    🛡️  KẾT LUẬN CŨ (Radius Lọc thô): {final_decision.upper()}")
        print(f"    🏆 KẾT LUẬN MỚI (SVM Lọc tinh ): {final_decision_svm.upper()}")
        print("    📊 Bảng điểm độ tương đồng chi tiết đối với tất cả các hãng:")
        
        for rank, (brand, score) in enumerate(brand_scores[:5]):
            print(f"       Rank {rank + 1}: {brand:<15} => Cosine Similarity: {score:.4f}")
            
        print("-" * 80)

if __name__ == "__main__":
    main()
