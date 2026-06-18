import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
_model_env = "~/.model.env"
print(ROOT)
__weights = ROOT / "alta_weights/aquafina/v18/best.pt"

def get_model_path():
    local_model = None
    if os.path.exists(_model_env):
        with open(_model_env,'rt') as file:
            local_model = file.readline()
    if local_model is None:
        return __weights
    if os.path.exists(local_model):
        return local_model
    return __weights

# ===== TÍCH HỢP TỪ CONFIG.PY GỐC CỦA DỰ ÁN =====
# Đảm bảo đường dẫn tham chiếu ngược ra ngoài thư mục gốc
PROJECT_ROOT = FILE.parents[1]

# ===== PATHS =====
WEIGHTS_DIR = PROJECT_ROOT / "best_weights"
DATABASE_DIR = PROJECT_ROOT / "best_embeddings"

# WEIGHTS_DIR = PROJECT_ROOT / "rvm-ai/alta_weights/weights"
# DATABASE_DIR = PROJECT_ROOT / "rvm-ai/alta_weights/database"

YOLO_DET_MODEL_ONNX_PATH = WEIGHTS_DIR / "best_13thg5.onnx"
YOLO_DET_MODEL_PATH  = WEIGHTS_DIR / "best_12thg6.pt"
YOLO_OBB_MODEL_ONNX_PATH = WEIGHTS_DIR / "best_obb26thg3.onnx"
TRIPLET_MODEL_ONNX_PATH= WEIGHTS_DIR / "model.onnx"
TRIPLET_MODEL_PATH= WEIGHTS_DIR / "mbnv2_embedding19thg3.pth"
TRIPLET_MODEL_PATH_GLASS= WEIGHTS_DIR / "embedding_model_g.pth"
GLASS_MODEL_PATH = WEIGHTS_DIR / "best_binary_classifier.pth"

DATABASE_EMBEDDING_PATH = DATABASE_DIR / "brand_database.npz"
DATABASE_EMBEDDING_PC = DATABASE_DIR / "brand_database_pc.npz"
DATABASE_EMBEDDING_G = DATABASE_DIR / "brand_database_g.npz"

# ===== YOLO CONFIG =====
# YOLO_CONF_OBB = 0.7
YOLO_CONF_DET = 0.8 
YOLO_IOU_DET = 0.5

# ===== COSINE SIMILARITY =====
SIM_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.06
OUTLIER_RADIUS_FLOOR = 0.15  # Ngưỡng sàn tối thiểu cho bán kính Outlier (tương đương Cosine Sim >= 0.85)
# ===== DEVICE =====
DEVICE = "mps"

# ===== CAMERA =====
CAMERA_INDEX = 0

# ===== FPS SMOOTHING =====
FPS_ALPHA = 0.5

# ===== DISPLAY =====
WINDOW_NAME = "Result"
FONT_SCALE = 0.6
FPS_FONT_SCALE = 0.8
IMAGE_SIZE = (224,224)
INPUT_SIZE = (320, 320)
ALPHA = 0.7
CLASSIFY_INTERVAL = 5
# CONF_THRESHOLD_CLASSIFY = 0.6
GLASS_CONF_THRESHOLD = 0.85

# ===== LABELS & COLORS =====
CLASS_NAMES = ['Can', 'Plastic']
BRAND_COLORS = {
    "Aquafina": (0, 255, 0),
    "unknown": (0, 0, 255)
}

# ===== VOLUME ESTIMATION =====
PIXEL_TO_CM_RATIO = 0.05
VOLUME_SCALING_UP = 1.18
VOLUME_SCALING_DOWN = 0.99
MIN_ACCEPTABLE_VOLUME = 180
MAX_ACCEPTABLE_VOLUME = 550

# ===== SLIDING WINDOW & VOTING =====
VOTING_WINDOW_CLASSIFIER = 5      # Cửa sổ vote của ClassifierWorker (VOTING_WINDOW)
SLIDING_WINDOW_ROI = 10           # Cửa sổ trượt lưu lịch sử phát hiện trong ROI (SLIDING_WINDOW_SIZE)
MIN_SAMPLES_ROI_CHECK = 5         # Số lượng mẫu tối thiểu để tính mật độ ROI
ROI_STABILITY_THRESHOLD = 0.75    # Ngưỡng mật độ (> 75%) để coi là vật thể ổn định trong ROI

# ===== INTERVALS & LIMITS =====
DETECTION_TIMEOUT = 5             # Thời gian tối đa thu thập mẫu (giây) (beginTime - endTime > 5)
MAX_DECISION_SAMPLES = 20         # Số lượng mẫu calc_ids tối đa cần thu thập để đưa ra kết luận
EMA_SMOOTHING_ALPHA = 0.6         # Hệ số EMA làm mượt embedding

# ===== ROI COORDINATES =====
ROI_COORDS = (200, 25, 475, 400)  # (roi_x1, roi_y1, roi_x2, roi_y2)

# ===== CAMERA RESOLUTION =====
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ===== FPS SMOOTHING ALPHAS =====
FPS_ALPHA_YOLO = 0.1
FPS_ALPHA_CLASSIFIER = 0.2
FPS_ALPHA_GUI = 0.1


# ===== YOLO CLASS DEFINITIONS =====
class YOLOClass:
    CAN = 0
    PLASTIC = 1
    GLASS = 2
    METAL_OTHER = 3
    PLASTIC_OTHER = 4
    HAND = 5

# ===== RVM OUTPUT CLASS DEFINITIONS =====
class RVMClass:
    AQUAFINA = 0
    STANDARD_PLASTIC = 1
    CAN = 2
    METAL_OTHER = 3
    PLASTIC_OTHER = 4
    REJECT = 7
