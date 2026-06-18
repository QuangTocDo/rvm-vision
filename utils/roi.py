"""
utils/roi.py
============
Tiện ích quản lý vùng quan tâm (Region Of Interest — ROI).

Thay thế 6+ đoạn code lặp lại kiểm tra toạ độ tâm trong index.py
bằng một dataclass rõ ràng, tái sử dụng được.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ROI:
    """
    Vùng quan tâm hình chữ nhật.

    Attributes:
        x1, y1: góc trái trên
        x2, y2: góc phải dưới
    """
    x1: int
    y1: int
    x2: int
    y2: int

    # ── Thuộc tính tính toán ────────────────────────────────────────────────

    @property
    def virtual_line_y(self) -> int:
        """Vạch ảo nằm ở 1/3 phía dưới ROI — dùng làm trigger line."""
        return self.y2 - (self.y2 - self.y1) // 3

    @property
    def coords(self) -> Tuple[int, int, int, int]:
        """Trả về (x1, y1, x2, y2) dạng tuple tương thích code cũ."""
        return (self.x1, self.y1, self.x2, self.y2)

    # ── Kiểm tra vị trí ──────────────────────────────────────────────────────

    def contains_center(self, bbox) -> bool:
        """
        Kiểm tra tâm của bounding box có nằm trong ROI không.

        Args:
            bbox: iterable có 4 phần tử (x1, y1, x2, y2) — pixel coords.

        Returns:
            True nếu tâm (cx, cy) nằm bên trong ROI.
        """
        x1, y1, x2, y2 = map(int, bbox)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return self.x1 < cx < self.x2 and self.y1 < cy < self.y2

    def center_of(self, bbox) -> Tuple[int, int]:
        """Trả về (cx, cy) của một bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def is_above_virtual_line(self, cy: int) -> bool:
        """Kiểm tra điểm cy có nằm phía trên (hoặc đúng) virtual_line_y không."""
        return cy <= self.virtual_line_y

    def is_bbox_complete(self, y2: int, margin: int = 15) -> bool:
        """
        Kiểm tra bbox có nằm hoàn toàn trên cạnh dưới ROI không
        (tức là vật thể chưa bị cắt bởi cạnh dưới frame).
        """
        return y2 < self.y2 + margin

    @classmethod
    def from_config(cls, roi_coords: Tuple[int, int, int, int]) -> "ROI":
        """Tạo ROI từ tuple config (roi_x1, roi_y1, roi_x2, roi_y2)."""
        return cls(x1=roi_coords[0], y1=roi_coords[1],
                   x2=roi_coords[2], y2=roi_coords[3])
