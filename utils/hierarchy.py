import config
import math
def evaluate_hierarchical_class(box, volume_cache, track_cache, roi_coords):
    """
    Đánh giá phân lớp theo logic phân tầng (Hierarchical Decision Logic V2).
    Trả về: (final_class, is_unknown_class2)
    """
    bbox, idx_class, conf, _id = box
    x1, y1, x2, y2 = map(int, bbox)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
    final_class = config.RVMClass.REJECT 
    is_unknown_class2 = False
    
    if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
        current_vol = volume_cache.get(_id, -1)
        # Bước 1: Kiểm tra Thể tích
        if config.MIN_ACCEPTABLE_VOLUME < current_vol < config.MAX_ACCEPTABLE_VOLUME:
            if idx_class == config.YOLOClass.CAN:
                final_class = config.RVMClass.CAN  # Lon
            elif idx_class == config.YOLOClass.PLASTIC:
                brand, b_score = track_cache.get(_id, ("unknown", 0.0))
                if brand.lower() == "aquafina":
                    final_class = config.RVMClass.AQUAFINA
                else:
                    final_class = config.RVMClass.STANDARD_PLASTIC
            elif idx_class == config.YOLOClass.GLASS:
                if _id in track_cache:
                    brand, b_score = track_cache.get(_id, ("unknown", 0.0))
                    if brand.lower() == "lanh":
                        final_class = config.RVMClass.STANDARD_PLASTIC  
                    else:
                        is_unknown_class2 = True
                else:
                    # Gán tạm nhãn STANDARD_PLASTIC trong khi luồng ClassifierWorker tính toán dự đoán
                    final_class = config.RVMClass.STANDARD_PLASTIC
            elif idx_class == config.YOLOClass.METAL_OTHER:
                final_class = config.RVMClass.METAL_OTHER
            elif idx_class == config.YOLOClass.PLASTIC_OTHER:
                final_class = config.RVMClass.PLASTIC_OTHER
    
    return final_class, is_unknown_class2

def estimate_volume(width, height, scale=config.PIXEL_TO_CM_RATIO, k=1.0):
    length_cm = width * scale
    diameter_cm = height * scale
    r = diameter_cm / 2
    volume = math.pi * (r ** 2) * length_cm * k
    return volume
