import cv2

def draw_premium_hud(display_img, status_text, status_color, smoothed_fps, yolo_fps, yolo_latency, cls_status, is_hand_in_roi, roi_coords):
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
    
    # Vẽ vùng ROI màu vàng
    cv2.rectangle(display_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
    cv2.putText(display_img, "ROI AREA", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Vẽ vạch ảo màu đỏ ở 1/3 phía dưới ROI làm vạch kích hoạt (Trigger Line) nằm ngang
    virtual_line_y = roi_y2 - (roi_y2 - roi_y1) // 3
    cv2.line(display_img, (roi_x1, virtual_line_y), (roi_x2, virtual_line_y), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(display_img, "TRIGGER LINE", (roi_x1 + 10, virtual_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Vẽ HUD nền mờ phía trên bên trái
    overlay = display_img.copy()
    cv2.rectangle(overlay, (5, 5), (290, 105), (0, 0, 0), -1)
    alpha = 0.55  
    cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)
    
    # Điền chữ các thông số FPS, trạng thái
    cv2.putText(display_img, f"Status: {status_text}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
    cv2.putText(display_img, f"Main GUI FPS: {int(smoothed_fps)}", (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(display_img, f"YOLO Det FPS: {int(yolo_fps)} ({yolo_latency:.1f}ms)", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1, cv2.LINE_AA)
    cv2.putText(display_img, f"Classifier: {cls_status}", (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Hiển thị cảnh báo bàn tay
    if is_hand_in_roi:
        cv2.putText(display_img, "WARNING: HAND IN ROI!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def draw_predictions(display_img, valid_boxes, volume_cache, track_cache, colors):
    for box in valid_boxes:
        bbox, idx_class, conf, _id = box
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[idx_class % len(colors)]
        
        vol = volume_cache.get(_id, -1)
        brand, b_score = track_cache.get(_id, ("N/A", 0.0))

        # Vẽ hình chữ nhật bounding box
        cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_img, f"{idx_class}|{conf:.2f}|{_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Vẽ thông tin thương hiệu
        text_color = (0, 255, 0) if brand.lower() == "lanh" else (255, 255, 0)
        cv2.putText(display_img, f"{brand.upper()}|{b_score:.2f}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        # Vẽ thể tích ml
        if vol > 0:
            cv2.putText(display_img, f"{vol:.0f}ml", (x1, y1 - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
