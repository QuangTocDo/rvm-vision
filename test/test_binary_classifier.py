import os
import sys
import argparse

# Chỉ định cấu hình chạy độc quyền trên GPU số 1 để tránh lỗi tràn bộ nhớ (OOM)
# LƯU Ý: Phải đặt trước khi import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

try:
    import torch
    import torch.nn as nn
    import cv2
    import numpy as np
    from torchvision import models, transforms
except ImportError:
    print("❌ Lỗi: Thiếu các thư viện cần thiết.")
    print("👉 Vui lòng cài đặt: pip install torch torchvision opencv-python numpy\n")
    sys.exit(1)

# Cấu hình mặc định
DEFAULT_IMAGE_DIR = "/Users/trannhutquang/PycharmProjects/PlasticCanCls/rvm-vision/videos"
DEFAULT_MODEL_PATH = "/Users/trannhutquang/PycharmProjects/PlasticCanCls/best_weights/best_binary_classifier.pth"

def preprocess_image(img):
    """
    Tiền xử lý ảnh BGR OpenCV sang tensor chuẩn ImageNet
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = preprocess(img_rgb)
    return tensor.unsqueeze(0) # Thêm chiều batch dimension

def main():
    parser = argparse.ArgumentParser(description="Chương trình quét và kiểm thử thư mục ảnh bằng Mô hình CNN Nhị phân PyTorch")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR, help="Thư mục chứa ảnh cần quét test")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Đường dẫn file trọng số .pth")
    parser.add_argument("--conf_thres", type=float, default=0.85, help="Ngưỡng tự tin chấp nhận chai Lanh")
    args = parser.parse_args()

    print("=" * 80)
    print("        CHƯƠNG TRÌNH QUÉT KIỂM THỬ THƯ MỤC ẢNH BẰNG MÔ HÌNH CNN NHỊ PHÂN")
    print("=" * 80)
    print(f"📁 Thư mục test  : {args.image_dir}")
    print(f"🛡️  File trọng số : {args.model_path}")
    print(f"⚙️ Ngưỡng tự tin : {args.conf_thres * 100:.1f}%")
    
    # 1. Nạp mô hình
    if not os.path.exists(args.model_path):
        print(f"\n❌ Lỗi: Không tìm thấy file trọng số '{args.model_path}'!")
        print("👉 Vui lòng huấn luyện mô hình trước hoặc đặt đúng file 'best_binary_classifier.pth' vào thư mục.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"💻 Thiết bị chạy  : {device} (Đã cô lập chạy trên GPU số 1 nếu có CUDA)")

    print("\n[1/3] Đang khởi tạo mô hình và nạp trọng số...")
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("      => Nạp mô hình thành công!")
    except Exception as e:
        print(f"❌ Lỗi nạp trọng số: {e}")
        return

    # 2. Quét thư mục ảnh
    if not os.path.exists(args.image_dir):
        print(f"❌ Lỗi: Thư mục ảnh test không tồn tại: {args.image_dir}")
        return

    img_names = [f for f in os.listdir(args.image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not img_names:
        print(f"❌ Không tìm thấy file ảnh .jpg/.png nào trong thư mục: {args.image_dir}")
        return

    print(f"\n[2/3] Đã phát hiện {len(img_names)} ảnh test. Bắt đầu quét kiểm thử...")
    print("-" * 90)
    
    total_tested = 0
    lanh_count = 0
    other_count = 0
    
    # Mảng ánh xạ nhãn
    labels_map = {0: "LANH (Chuẩn)", 1: "OTHER (Chai lạ / Dị vật)"}

    # Chạy qua từng ảnh
    for idx, img_name in enumerate(sorted(img_names)):
        img_path = os.path.join(args.image_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️  Không thể đọc ảnh: {img_name}")
            continue

        total_tested += 1
        
        # Tiền xử lý và đẩy lên thiết bị
        input_tensor = preprocess_image(img).to(device)
        
        # Suy luận
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            
        p_lanh = float(probs[0])
        p_other = float(probs[1])
        class_pred = int(torch.argmax(probs))
        
        # Tiêu chí quyết định
        is_lanh_match = (class_pred == 0 and p_lanh >= args.conf_thres)
        
        if is_lanh_match:
            lanh_count += 1
            status_str = "🟢 [LANH CHUẨN]"
            detail_str = f"Lanh: {p_lanh*100:.2f}%"
        else:
            other_count += 1
            status_str = "🔴 [CHAI LẠ / DỊ VẬT]"
            if class_pred == 0:
                detail_str = f"Lanh: {p_lanh*100:.2f}% (Bị loại do dưới ngưỡng {args.conf_thres*100:.1f}%)"
            else:
                detail_str = f"Other: {p_other*100:.2f}% (Lanh: {p_lanh*100:.2f}%)"

        print(f"[{idx+1:<3}] 📷 FILE: {img_name:<40} | {status_str:<25} | {detail_str}")

    # 3. Tổng kết báo cáo
    print("-" * 90)
    print("\n[3/3] BÁO CÁO KẾT QUẢ TỔNG QUAN:")
    print("=" * 80)
    print(f"   📊 Tổng số ảnh đã quét     : {total_tested} ảnh")
    print(f"   🟢 Số chai LANH chấp nhận  : {lanh_count} ({lanh_count/total_tested*100:.1f}%)")
    print(f"   🔴 Số chai LẠ / RÁC bị loại : {other_count} ({other_count/total_tested*100:.1f}%)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
