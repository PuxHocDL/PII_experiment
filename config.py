import os
import torch
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN", "")
TARGET_REPO = os.getenv("TARGET_REPO", "")


# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
RUN_QUICK_TEST = True
DEBUG_MODE = False
QUICK_SAMPLE_SIZE = 100
DEBUG_SAMPLE_SIZE = 10
MAX_SEQ_LENGTH = 512
MAX_SPAN_WIDTH = 30
SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')