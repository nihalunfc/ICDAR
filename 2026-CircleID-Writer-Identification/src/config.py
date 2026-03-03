# --- Configuration and Hyperparameters ---
import torch

# Paths (Kaggle Specific)
DATA_PATH = "/kaggle/input/competitions/icdar-2026-circleid-writer-identification"
TRAIN_CSV = f"{DATA_PATH}/train.csv"
TEST_CSV = f"{DATA_PATH}/test.csv"
IMAGE_ROOT = DATA_PATH  # Paths in CSV are relative to this

# Model Settings
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# Open-Set Recognition Setting
# If the model confidence is below this, we label the writer as -1
CONFIDENCE_THRESHOLD = 0.5 

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
