# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# Paths
DATA_DIR        = Path(os.getenv("MALTWIN_DATA_DIR", BASE_DIR / "data/malimg"))
PROCESSED_DIR   = BASE_DIR / "data/processed"
MODEL_DIR       = Path(os.getenv("MALTWIN_MODEL_DIR", BASE_DIR / "models"))
CHECKPOINT_DIR  = MODEL_DIR / "checkpoints"
LOG_DIR         = Path(os.getenv("MALTWIN_LOG_DIR", BASE_DIR / "logs"))
REPORTS_DIR     = Path(os.getenv("MALTWIN_REPORTS_DIR", BASE_DIR / "reports"))
MITRE_JSON_PATH = BASE_DIR / "data/mitre_ics_mapping.json"
DB_PATH         = LOG_DIR / "maltwin.db"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pt"

# Image settings
IMG_SIZE        = int(os.getenv("MALTWIN_IMG_SIZE", 128))   # 128x128 per SRS FR3.2

# Training hyperparameters
BATCH_SIZE      = int(os.getenv("MALTWIN_BATCH_SIZE", 32))
EPOCHS          = int(os.getenv("MALTWIN_EPOCHS", 30))
LR              = float(os.getenv("MALTWIN_LR", 0.001))
NUM_WORKERS     = int(os.getenv("MALTWIN_NUM_WORKERS", 4))
TRAIN_RATIO     = 0.7
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15   # must sum to 1.0

# Device
import torch
_device_env = os.getenv("MALTWIN_DEVICE", "auto")
if _device_env == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(_device_env)

# Binary upload limits
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per SRS FR3.1
ACCEPTED_EXTENSIONS = {".exe", ".dll", ".elf", ""}  # "" covers extensionless ELF

# Confidence thresholds for UI color coding (SRS FR5.2)
CONFIDENCE_GREEN = 0.80
CONFIDENCE_AMBER = 0.50
