import cv2
import math
import numpy as np
def draw_detections(frame, detections, fps):
    for det in detections:
        x1, y1, x2, y2 = det["box"]

        base_label = f'{det["class_name"]} {det["score"]:.2f}'

        if "brand" in det and "brand_conf" in det:
            brand_label = f'{det["brand"]} {det["brand_conf"]:.2f}'
            label = base_label + " | " + brand_label
        else:
            label = base_label + " | unknow "

        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0), 2)

        cv2.putText(frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0), 2)

    # ===== Draw FPS =====
    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)

    return frame

# def estimate_volume(width, height, scale=0.03, k=1.0):
#     """
#     width, height: pixel (lấy từ OBB)
#     scale: cm / pixel
#     k: hệ số hiệu chỉnh (do chai không phải hình trụ hoàn hảo)
#     """
#     # ===== convert sang cm =====
#     w_cm = width * scale
#     h_cm = height * scale
#
#     # ===== bán kính =====
#     r = w_cm / 2
#
#     # ===== thể tích hình trụ =====
#     volume = math.pi * (r ** 2) * h_cm * k
#
#     return volume


def estimate_volume(width, height, scale=0.001, k=1.0):
    # width từ OBB đang là cạnh dài (chiều dài chai)
    # height từ OBB đang là cạnh ngắn (đường kính chai)

    # ===== convert sang cm =====
    length_cm = width * scale
    diameter_cm = height * scale

    # ===== bán kính =====
    r = diameter_cm / 2  # Bán kính phải lấy từ cạnh ngắn

    # ===== thể tích hình trụ =====
    volume = math.pi * (r ** 2) * length_cm * k  # Chiều cao hình trụ lấy từ cạnh dài

    return volume
def crop_from_box(frame, box, padding=0, auto_rotate=True):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # ===== Thêm padding =====
    x1 -= padding
    y1 -= padding
    x2 += padding
    y2 += padding

    # ===== Clamp để không vượt khỏi ảnh =====
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # ===== Check box hợp lệ =====
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]

    # ===== Auto rotate nếu nằm ngang =====
    if auto_rotate:
        ch, cw = crop.shape[:2]
        if cw > ch:  # nằm ngang
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return crop

def fix_orientation(crop):

    h, w = crop.shape[:2]

    if w > h:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return crop
def crop_from_obb(frame, obb):
    pts = obb["poly"]

    w = int(obb["width"])
    h = int(obb["height"])

    dst = np.array([
        [0, h-1],
        [0, 0],
        [w-1, 0],
        [w-1, h-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts.astype("float32"), dst)
    warped = cv2.warpPerspective(frame, M, (w, h))

    return warped


