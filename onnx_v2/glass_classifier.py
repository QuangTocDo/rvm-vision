import os
import torch
import torch.nn as nn
import cv2
from torchvision import models, transforms

class GlassClassifier:
    """
    Mô-đun độc lập phục vụ việc phân loại nhị phân chai Thủy tinh Lanh bằng PyTorch CNN (MobileNetV3).
    Tách biệt hoàn toàn phần tiền xử lý, cấu trúc mạng, nạp trọng số và suy luận giúp code chính tinh giản.
    """
    def __init__(self, model_path, device=None, conf_thres=0.85):
        """
        Khởi tạo và tải mô hình PyTorch nhị phân.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        else:
            self.device = device
            
        self.conf_thres = conf_thres
        
        # 1. Khởi tạo cấu trúc mạng MobileNetV3 Small nhị phân
        self.model = models.mobilenet_v3_small(weights=None)
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, 2) # 2 classes: 0 - Lanh, 1 - Other
        
        # 2. Nạp trọng số huấn luyện từ tệp tin .pth
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"      => [GlassClassifier] Nạp trọng số PyTorch thành công từ: {model_path}")
            except Exception as e:
                print(f"❌ [GlassClassifier] Lỗi nạp trọng số: {e}")
        else:
            print(f"⚠️ [GlassClassifier] Cảnh báo: Không tìm thấy tệp trọng số tại: {model_path}")
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 3. Định nghĩa chuỗi tiền xử lý ảnh chuẩn ImageNet cho PyTorch
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, crop):
        """
        Thực hiện tiền xử lý và suy luận nhị phân từ ảnh cắt ra từ camera.
        Đầu vào: crop (BGR OpenCV numpy array)
        Trả về: tuple (pred_class, score)
            - pred_class: "lanh" hoặc "unknown"
            - score: độ tin cậy phần trăm trong khoảng [0.0, 1.0]
        """
        try:
            if crop is None or crop.size == 0:
                return "unknown", 0.0
                
            # Chuyển BGR sang RGB giống hệt lúc huấn luyện PyTorch
            img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            input_tensor = self.preprocess(img_rgb).unsqueeze(0).to(self.device)
            
            # Chạy suy luận không tính đạo hàm để tối ưu tốc độ và bộ nhớ
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                
            p_lanh = float(probs[0])
            class_pred = int(torch.argmax(probs))
            
            # Áp dụng ngưỡng tự tin để đưa ra quyết định
            is_lanh = (class_pred == 0 and p_lanh >= self.conf_thres)
            
            if is_lanh:
                return "lanh", p_lanh
            else:
                # Nếu là Other hoặc dưới ngưỡng tự tin, kết luận là unknown (chai lạ)
                p_score = float(probs[class_pred])
                return "unknown", p_score
                
        except Exception as e:
            print(f"❌ [GlassClassifier] Lỗi trong quá trình suy luận: {e}")
            return "unknown", 0.0
