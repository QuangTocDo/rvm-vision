"""
core/state_machine.py
=====================
DetectionStateMachine — quản lý vòng đời phân loại vật thể.

Trích xuất toàn bộ state-management logic ra khỏi hàm run() khổng lồ
trong index.py, giúp mỗi class chỉ có một nhiệm vụ duy nhất.

Vòng đời trạng thái:
    IDLE  ──arm()──>  ARMED  ──trigger()──>  DETECTING
      ^                  |                       |
      |    cancel()       |       finalize()      |
      +<─────────────────+<──────────────────────+
      |                                          |
      +<──────────── abort() ────────────────────+
"""

import logging
import threading
import time
from enum import Enum, auto
from typing import List, Optional

import config

logger = logging.getLogger("RVM_Vision")


# ── Enum trạng thái ──────────────────────────────────────────────────────────

class State(Enum):
    IDLE      = auto()   # Chờ lệnh từ UI
    ARMED     = auto()   # UI đã ra lệnh, chờ vật vượt virtual line
    DETECTING = auto()   # Đang thu thập mẫu phân loại


# ── Kết quả phân loại ────────────────────────────────────────────────────────

class DetectionResult:
    """Kết quả phân loại hoàn chỉnh, truyền ra ngoài khi finalize."""
    __slots__ = ("data", "volume", "size", "images", "item", "reason")

    def __init__(self, data: int, volume: float, size: float,
                 images: List[str], item: int, reason: str = "") -> None:
        self.data    = data
        self.volume  = volume
        self.size    = size
        self.images  = images
        self.item    = item
        self.reason  = reason


# ── State Machine ─────────────────────────────────────────────────────────────

class DetectionStateMachine:
    """
    Thread-safe state machine cho vòng đời phân loại vật thể.

    Mọi thay đổi trạng thái đều được bảo vệ bằng một RLock duy nhất.
    Phương thức công khai trả về True nếu chuyển trạng thái thành công,
    False nếu điều kiện tiên quyết không đúng.
    """

    def __init__(self) -> None:
        self._state     = State.IDLE
        self._lock      = threading.RLock()

        # ── Accumulators (chỉ hợp lệ khi state == DETECTING) ──────────────
        self.target_id  : int         = -1
        self.begin_time : float       = 0.0
        self.calc_ids   : List[int]   = []
        self.images     : List[str]   = []
        self.sizes      : List[float] = []

        # Thời điểm bắt đầu ARMED (để debug / timeout nếu cần)
        self.arm_time   : Optional[float] = None

    # ── Truy vấn trạng thái ─────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == State.IDLE

    @property
    def is_armed(self) -> bool:
        return self._state == State.ARMED

    @property
    def is_detecting(self) -> bool:
        return self._state == State.DETECTING

    def elapsed(self) -> float:
        """Thời gian đã trôi qua kể từ khi bắt đầu DETECTING (giây)."""
        if self._state != State.DETECTING:
            return 0.0
        return time.time() - self.begin_time

    def samples_collected(self) -> int:
        return len(self.calc_ids)

    # ── Chuyển trạng thái ───────────────────────────────────────────────────

    def arm(self) -> bool:
        """IDLE → ARMED. Trả về True nếu thành công."""
        with self._lock:
            if self._state != State.IDLE:
                return False
            self._state   = State.ARMED
            self.arm_time = time.time()
            logger.info("StateMachine: IDLE → ARMED")
            return True

    def trigger(self, target_id: int) -> bool:
        """ARMED → DETECTING. Trả về True nếu thành công."""
        with self._lock:
            if self._state != State.ARMED:
                return False
            self._reset_accumulators()
            self._state     = State.DETECTING
            self.target_id  = target_id   # set AFTER reset
            self.begin_time = time.time()
            logger.info(f"StateMachine: ARMED → DETECTING (target_id={target_id})")
            return True

    def cancel(self) -> None:
        """Hủy từ bất kỳ trạng thái nào → IDLE (lệnh từ UI)."""
        with self._lock:
            if self._state != State.IDLE:
                logger.info(f"StateMachine: {self._state.name} → IDLE (user cancel)")
            self._to_idle()

    def abort(self, reason: str) -> DetectionResult:
        """
        Hủy khẩn cấp trong khi DETECTING (e.g. phát hiện tay) → IDLE.

        Trả về DetectionResult với data=-1 để caller emit kết quả hủy.
        """
        with self._lock:
            # Capture state BEFORE resetting
            vol  = 0.0
            size = self._avg_sizes()
            imgs = list(self.images)
            item = self.target_id           # save before _to_idle clears it
            logger.warning(f"StateMachine: DETECTING → IDLE (abort: {reason})")
            self._to_idle()
            return DetectionResult(data=-1, volume=vol, size=size,
                                   images=imgs, item=item, reason=reason)

    def reject_immediately(self, volume: float, triggered_id: int,
                           reason: str = "Volume out of bounds") -> DetectionResult:
        """
        Từ chối ngay khi ARMED nhưng volume không hợp lệ → IDLE.

        Trả về DetectionResult với data=REJECT.
        """
        with self._lock:
            self._to_idle()
            logger.info(f"StateMachine: ARMED → IDLE (reject: {reason}, "
                        f"volume={volume:.1f}, id={triggered_id})")
            return DetectionResult(data=config.RVMClass.REJECT, volume=volume,
                                   size=0.0, images=[], item=triggered_id,
                                   reason=reason)

    # ── Thu thập mẫu (chỉ khi DETECTING) ────────────────────────────────────

    def add_sample(self, class_id: int, size: float) -> None:
        """Thêm một mẫu phân loại vào bộ đệm. Bỏ qua nếu không DETECTING."""
        with self._lock:
            if self._state != State.DETECTING:
                return
            self.calc_ids.append(class_id)
            self.sizes.append(size)

    def should_finalize(self) -> bool:
        """True nếu đã đủ mẫu hoặc timeout."""
        with self._lock:
            if self._state != State.DETECTING:
                return False
            enough_samples = len(self.calc_ids) >= config.MAX_DECISION_SAMPLES
            timed_out      = self.elapsed() > config.DETECTION_TIMEOUT
            return enough_samples or timed_out

    def finalize(self, volume_cache: dict) -> DetectionResult:
        """
        Kết thúc DETECTING → IDLE.

        Tính toán kết quả bỏ phiếu đa số (majority vote) và trả về.
        """
        from collections import Counter

        with self._lock:
            # Capture state BEFORE resetting
            if self.calc_ids:
                counts = Counter(self.calc_ids)
                avg = counts.most_common(1)[0][0]
            else:
                avg = -1

            vol  = float(volume_cache.get(self.target_id, -1))
            size = self._avg_sizes()
            imgs = list(self.images)
            item = self.target_id
            n_samples = len(self.calc_ids)

            logger.info(f"StateMachine: DETECTING → IDLE (finalize: data={avg}, "
                        f"volume={vol:.1f}, samples={n_samples})")
            self._to_idle()               # reset AFTER capturing everything
            return DetectionResult(data=avg, volume=vol, size=size,
                                   images=imgs, item=item)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _to_idle(self) -> None:
        """Chuyển về IDLE và reset tất cả accumulators."""
        self._state    = State.IDLE
        self.arm_time  = None
        self.target_id = -1
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        """Chỉ reset các bộ tích lũy mẫu — KHÔNG đụng đến target_id."""
        self.begin_time = 0.0
        self.calc_ids   = []
        self.images     = []
        self.sizes      = []

    def _avg_sizes(self) -> float:
        if not self.sizes:
            return 0.0
        return sum(self.sizes) / len(self.sizes)
