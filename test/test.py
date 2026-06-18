import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Tải cơ sở dữ liệu đặc trưng của hãng A
data = np.load("/Users/trannhutquang/PycharmProjects/PlasticCanCls/best_embeddings/brand_database_g.npz")
v_norm = data["lanh"]  # Lấy ví dụ hãng Aquafina
v_norm = v_norm / np.linalg.norm(v_norm, axis=1, keepdims=True)

# Thử nghiệm K từ 2 đến 6
best_k = 2
best_score = -1

for k in range(2, 7):
    if len(v_norm) > k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(v_norm)
        score = silhouette_score(v_norm, labels)
        print(f"Số lượng cụm K = {k} | Điểm Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k

print(f"==> Số lượng cụm tối ưu nhất cho hãng này là K = {best_k} với điểm số: {best_score:.4f}")
