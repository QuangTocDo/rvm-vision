"""
utils/types.py
==============
Định nghĩa các kiểu dữ liệu chung được dùng xuyên suốt hệ thống.

Thay thế tuple vô danh (bbox, idx_class, conf, _id) bằng
dataclass rõ ràng, có type hints đầy đủ.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class DetectionBox:
    """
    Đại diện cho một bounding box đã được tracking.

    Attributes:
        bbox      : Tọa độ (x1, y1, x2, y2) dạng pixel.
        class_id  : Index của class do YOLO phân loại (YOLOClass.*).
        confidence: Độ tin cậy của YOLO [0.0, 1.0].
        track_id  : ID tracking duy nhất do DeepSORT gán.
    """
    bbox: BBox
    class_id: int
    confidence: float
    track_id: int

    # ── Thuộc tính tính toán ────────────────────────────────────────────────

    @property
    def x1(self) -> int:
        return int(self.bbox[0])

    @property
    def y1(self) -> int:
        return int(self.bbox[1])

    @property
    def x2(self) -> int:
        return int(self.bbox[2])

    @property
    def y2(self) -> int:
        return int(self.bbox[3])

    @property
    def center(self) -> Tuple[int, int]:
        """Tâm (cx, cy) của bounding box."""
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    # ── Chuyển đổi tương thích ngược ────────────────────────────────────────

    def to_tuple(self):
        """Chuyển thành tuple cũ (bbox, class_id, confidence, track_id)."""
        return (self.bbox, self.class_id, self.confidence, self.track_id)

    @classmethod
    def from_tuple(cls, t) -> "DetectionBox":
        """Tạo từ tuple cũ (bbox, idx_class, conf, _id)."""
        bbox, class_id, confidence, track_id = t
        return cls(bbox=tuple(map(int, bbox)), class_id=int(class_id),
                   confidence=float(confidence), track_id=int(track_id))
