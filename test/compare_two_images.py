import os
import sys
import argparse

# Thiết lập in tiếng Việt có dấu chuẩn đẹp trên Terminal
if sys.platform.startswith('win'):
    import lzma  # Tránh lỗi trên một số môi trường Windows cũ
    os.system('chcp 65001 > nul')

# Chỉ định cấu hình chạy trên GPU số 1 để tránh lỗi tràn bộ nhớ (OOM) trên các GPU khác.
# LƯU Ý: Phải đặt biến môi trường này trước khi import torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    import torch
    import torch.nn as nn
    import cv2
    import numpy as np
    from torchvision import models, transforms
except ImportError:
    print("❌ Lỗi: Thiếu các thư viện cần thiết.")
    print("👉 Vui lòng chạy lệnh: pip install torch torchvision opencv-python numpy\n")
    sys.exit(1)

def preprocess_image(img_path):
    """
    Đọc và tiền xử lý ảnh theo chuẩn ImageNet để đưa vào mạng CNN (MobileNetV3)
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {img_path}")
        
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh (định dạng hỏng hoặc sai đường dẫn): {img_path}")
        
    # Chuyển BGR sang RGB giống hệt lúc huấn luyện PyTorch
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Chuỗi chuyển đổi ảnh theo chuẩn ImageNet
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = preprocess(img_rgb)
    # Thêm chiều Batch (batch dimension) -> [1, 3, 224, 224]
    return tensor.unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Đánh giá và so sánh nhãn chai Thủy tinh bằng mô hình Deep Learning Binary Classifier.")
    parser.add_argument("image1", nargs="?", help="Đường dẫn tới ảnh thứ nhất")
    parser.add_argument("image2", nargs="?", help="Đường dẫn tới ảnh thứ hai")
    parser.add_argument("--model_weights", type=str, default="best_binary_classifier.pth", help="Đường dẫn tới file trọng số .pth")
    args = parser.parse_args()

    img1_path = args.image1
    img2_path = args.image2
    weights_path = args.model_weights

    print("=" * 85)
    print("     SO SÁNH VÀ ĐÁNH GIÁ CHAI THỦY TINH BẰNG MÔ HÌNH BINARY CNN CLASSIFIER")
    print("=" * 85)

    if not img1_path or not img2_path:
        print("\n💡 Cách chạy nhanh: python3 compare_two_images.py <ảnh_1> <ảnh_2>")
        if not img1_path:
            img1_path = input("👉 Nhập đường dẫn ảnh 1 (Ảnh mẫu nhãn chuẩn): ").strip().strip('"').strip("'")
        if not img2_path:
            img2_path = input("👉 Nhập đường dẫn ảnh 2 (Ảnh cần kiểm tra): ").strip().strip('"').strip("'")

    # 1. Đọc và tiền xử lý ảnh
    print("\n[1/3] Đang tải và tiền xử lý dữ liệu ảnh đầu vào...")
    try:
        input1 = preprocess_image(img1_path)
        input2 = preprocess_image(img2_path)
        print(f"      + Ảnh 1: {os.path.basename(img1_path)} => Kích thước tensor: {list(input1.shape)}")
        print(f"      + Ảnh 2: {os.path.basename(img2_path)} => Kích thước tensor: {list(input2.shape)}")
    except Exception as e:
        print(f"❌ Lỗi đọc ảnh: {e}")
        return

    # 2. Khởi tạo mô hình
    print("\n[2/3] Đang xây dựng cấu trúc mô hình MobileNetV3 Small...")
    # Tải cấu trúc MobileNetV3 Small mặc định
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2) # Lớp đầu ra nhị phân (2 classes: 0 - Lanh, 1 - Other)

    # Nạp trọng số huấn luyện
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    if os.path.exists(weights_path):
        print(f"      => Phát hiện tệp trọng số huấn luyện: '{weights_path}'")
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print("      => Nạp trọng số PyTorch thành công!")
        except Exception as e:
            print(f"❌ Lỗi nạp trọng số: {e}. Đang tiếp tục chạy với trọng số ngẫu nhiên.")
    else:
        print(f"⚠️  Cảnh báo: Không tìm thấy file trọng số '{weights_path}'.")
        print("💡 Gợi ý: Hãy chạy script 'train_binary_classifier.py' trước để tối ưu hóa mô hình.")
        print("      * Đang nạp mô hình MobileNetV3 mặc định (ImageNet) để chạy kiểm thử giả lập...")
        # Dự phòng bằng cách nạp pre-trained weights từ ImageNet để chạy demo được ngay
        try:
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_ftrs, 2)
            print("      => Nạp weights demo thành công!")
        except Exception as e:
            print(f"⚠️  Không tải được pre-trained demo. Sử dụng weights khởi tạo ngẫu nhiên.")

    model = model.to(device)
    model.eval() # Chuyển sang Evaluation mode (tắt BatchNorm/Dropout)

    # 3. Chạy suy luận (Inference)
    print("\n[3/3] Đang thực hiện nhận diện bằng mô hình học sâu...")
    
    input1 = input1.to(device)
    input2 = input2.to(device)
    
    with torch.no_grad():
        out1 = model(input1)
        out2 = model(input2)
        
        # Tính xác suất Softmax cho từng lớp
        prob1 = torch.softmax(out1, dim=1)[0]
        prob2 = torch.softmax(out2, dim=1)[0]

    # Lấy thông tin lớp
    # Lớp 0 đại diện cho LANH, Lớp 1 đại diện cho OTHER
    p_lanh1 = float(prob1[0])
    p_other1 = float(prob1[1])
    class_pred1 = int(torch.argmax(prob1))

    p_lanh2 = float(prob2[0])
    p_other2 = float(prob2[1])
    class_pred2 = int(torch.argmax(prob2))

    labels_map = {0: "LANH (Chuẩn)", 1: "OTHER (Chai lạ / Dị vật)"}

    # In báo cáo chi tiết cho từng ảnh
    print("\n" + "=" * 85)
    print("📊 KẾT QUẢ PHÂN TÍCH NHẬN DIỆN CHI TIẾT:")
    print("=" * 85)
    print(f"   📷 ẢNH 1: {os.path.basename(img1_path)}")
    print(f"      + Dự đoán nhãn: {labels_map[class_pred1].upper()}")
    print(f"      + Xác suất LANH (Chuẩn)   : {p_lanh1 * 100:.2f}%")
    print(f"      + Xác suất OTHER (Chai lạ): {p_other1 * 100:.2f}%")
    print("-" * 85)
    print(f"   📷 ẢNH 2: {os.path.basename(img2_path)}")
    print(f"      + Dự đoán nhãn: {labels_map[class_pred2].upper()}")
    print(f"      + Xác suất LANH (Chuẩn)   : {p_lanh2 * 100:.2f}%")
    print(f"      + Xác suất OTHER (Chai lạ): {p_other2 * 100:.2f}%")
    print("-" * 85)

    # Tiêu chí quyết định MATCH (Đồng nhất):
    # Cả hai ảnh đều phải được phân loại là LANH (class 0) với độ tự tin (Confidence) >= 85% (0.85)
    CONFIDENCE_THRESHOLD = 0.85
    is_match = (class_pred1 == 0 and p_lanh1 >= CONFIDENCE_THRESHOLD) and \
               (class_pred2 == 0 and p_lanh2 >= CONFIDENCE_THRESHOLD)

    print(f"   ⚙️ Ngưỡng tự tin tối thiểu để chấp nhận chai Lanh: {CONFIDENCE_THRESHOLD * 100:.1f}%")
    print("-" * 85)
    
    if is_match:
        print(f"   🟢 KẾT LUẬN: ĐỒNG NHẤT (MATCH)")
        print(f"      => Cả hai ảnh đều là vỏ chai thủy tinh 'LANH' chuẩn, đủ điều kiện thu hồi!")
    else:
        print(f"   🔴 KẾT LUẬN: KHÁC BIỆT / BẤT THƯỜNG (DIFFERENT)")
        print(f"      => Phát hiện chai thủy tinh lạ (Tiger, Heineken) hoặc dị vật trong buồng nhận diện.")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()
