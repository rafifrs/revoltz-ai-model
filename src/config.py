from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "hnei"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_MODEL_PATH = MODELS_DIR / "cell_matcher_v1.pkl"
DEFAULT_SNAPSHOT_CYCLE = 100

COMPATIBILITY_IR_THRESHOLD = 0.005
COMPATIBILITY_OCV_THRESHOLD = 0.05
IQR_MULTIPLIER = 1.5
MODEL_VERSION = "cell_matcher_v1"

