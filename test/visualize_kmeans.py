import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Thêm thư mục cha vào sys.path để import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config

def plot_brand_clusters(database_path):
    print(f"Loading database from: {database_path}")
    data = np.load(database_path)
    brands = data.files
    num_brands = len(brands)
    
    if num_brands == 0:
        print("Database is empty!")
        return
        
    # Tính toán kích thước lưới đồ thị (grid size)
    cols = 3
    rows = int(np.ceil(num_brands / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
    if num_brands == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, brand in enumerate(brands):
        ax = axes[idx]
        v = data[brand]
        
        # 1. Chuẩn hóa L2
        v_norm = v / np.linalg.norm(v, axis=1, keepdims=True)
        
        # 2. Định nghĩa số cụm giống TripletClassifier
        n_clusters = min(2, len(v_norm))
        
        # 3. Phân cụm KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(v_norm)
        centroids = kmeans.cluster_centers_
        
        # 4. Giảm chiều dữ liệu xuống 2D dùng PCA để trực quan hóa
        pca = PCA(n_components=2)
        # Gộp cả các điểm mẫu và tâm cụm lại để giảm chiều đồng bộ
        combined = np.vstack([v_norm, centroids])
        combined_2d = pca.fit_transform(combined)
        
        v_2d = combined_2d[:len(v_norm)]
        centroids_2d = combined_2d[len(v_norm):]
        
        # 5. Vẽ các điểm mẫu theo cụm
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Màu sắc phân biệt
        for c_id in range(n_clusters):
            class_mask = (cluster_labels == c_id)
            ax.scatter(
                v_2d[class_mask, 0], v_2d[class_mask, 1],
                s=40, color=colors[c_id % len(colors)], alpha=0.7,
                label=f"Cluster {c_id}"
            )
            
        # 6. Vẽ các tâm cụm (Centroids) làm mốc
        ax.scatter(
            centroids_2d[:, 0], centroids_2d[:, 1],
            s=200, c='black', marker='X', edgecolors='white', linewidths=1.5,
            label='Centroids'
        )
        
        ax.set_title(f"Brand: {brand} (K={n_clusters})", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=8)
        
    # Ẩn các ô đồ thị trống nếu có
    for idx in range(num_brands, len(axes)):
        fig.delaxes(axes[idx])
        
    plt.suptitle("Tối ưu hóa Phân cụm K-Means cho từng Brand riêng biệt (2D PCA Projection)", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Lưu đồ thị ra thư mục dự án
    output_path = "brand_kmeans_clusters.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSuccessfully saved cluster plot to: {os.path.abspath(output_path)}")
    plt.show()

if __name__ == "__main__":
    db_path = config.DATABASE_EMBEDDING_G
    plot_brand_clusters(db_path)
