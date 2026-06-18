"""
utils/result_logger.py
======================
Ghi kết quả phân loại ra file dạng JSON Lines (.jsonl) thay vì
chuỗi văn bản thô — dễ parse, query, và phân tích về sau.

Mỗi dòng là một JSON object độc lập:
    {"timestamp": "...", "data": 0, "mac_id": "r-xxx", ...}
"""

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger("RVM_Vision")


class ResultLogger:
    """
    Thread-safe logger ghi kết quả phân loại dạng JSON Lines.

    Usage:
        rlog = ResultLogger("results.jsonl")
        rlog.log_result(data=0, mac_id="r-abc", images=[], size=0, item=1,
                        volume=330.5, reason="OK")
    """

    def __init__(self, log_path: str = "results.jsonl") -> None:
        self.path = Path(log_path)
        self._lock = threading.Lock()
        logger.info(f"[ResultLogger] Ghi kết quả vào: {self.path.resolve()}")

    # ── Public API ────────────────────────────────────────────────────────────

    def log_result(
        self,
        data: int,
        mac_id: str,
        images: List[str],
        size: float,
        item: int,
        volume: float,
        reason: str = "",
    ) -> None:
        """
        Ghi một kết quả phân loại.

        Args:
            data   : Mã class kết quả (RVMClass.* hoặc -1 khi hủy).
            mac_id : Mã định danh thiết bị.
            images : Danh sách đường dẫn ảnh mẫu.
            size   : Kích thước tương đối (phần trăm frame).
            item   : Track ID của vật thể.
            volume : Thể tích ước tính (ml).
            reason : Lý do bổ sung (e.g. "Hand safety abort", "Volume OOB").
        """
        entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "data": data,
            "mac_id": mac_id,
            "images": images,
            "size": round(float(size), 2),
            "item": item,
            "volume": round(float(volume), 1),
            "reason": reason,
        }
        self._write(entry)

    # ── Private ───────────────────────────────────────────────────────────────

    def _write(self, entry: dict) -> None:
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error(f"[ResultLogger] Không thể ghi log: {exc}")
