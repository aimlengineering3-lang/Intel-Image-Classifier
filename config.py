from pathlib import Path
import torch

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"

TRAIN_DIR = DATASET_DIR / "seg_train"

TEST_DIR = DATASET_DIR / "seg_test"

PRED_DIR = DATASET_DIR / "seg_pred"

# ==================================================
# OUTPUT DIRECTORIES
# ==================================================

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "outputs"

PLOT_DIR = OUTPUT_DIR / "plots"

PREDICTION_DIR = OUTPUT_DIR / "predictions"

# Create folders automatically if they don't exist
MODEL_DIR.mkdir(exist_ok=True)

OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_DIR.mkdir(exist_ok=True)

PREDICTION_DIR.mkdir(exist_ok=True)

# ==================================================
# TRAINING SETTINGS
# ==================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

NUM_CLASSES = 6

EPOCHS = 10

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==================================================
# MODEL
# ==================================================

MODEL_NAME = "ResNet18"

MODEL_PATH = MODEL_DIR / "best_model.pth"

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

# ==================================================
# SCHEDULER
# ==================================================

STEP_SIZE = 3

GAMMA = 0.1