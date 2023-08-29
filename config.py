import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
_model_env = "~/.model.env"
print(ROOT)
__weights = ROOT / "alta_weights/aquafina/v17/best.pt"

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
