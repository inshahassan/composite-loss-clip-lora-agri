"""
Configuration for AgroOnto-VLM lambda sweep.
Paths are set for local Windows GPU execution.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PLANTVILLAGE_DIR = os.path.join(DATA_DIR, "plantvillage", "PlantVillage")
DEEPWEEDS_IMG_DIR = os.path.join(DATA_DIR, "deepweeds", "images")
DEEPWEEDS_LABEL_DIR = os.path.join(DATA_DIR, "deepweeds", "labels")
IP102_IMG_DIR = os.path.join(DATA_DIR, "ip102", "IP102_YOLOv5", "images")
IP102_LABEL_DIR = os.path.join(DATA_DIR, "ip102", "IP102_YOLOv5", "labels")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Training ───────────────────────────────────────────────────────────
EPOCHS = 7
BATCH_SIZE = 16          # RTX 3050 4GB — safe with fp16
NUM_WORKERS = 0
SEED = 42
CLIP_MODEL = "ViT-B/32"
IMAGE_SIZE = 224

# ── LoRA ───────────────────────────────────────────────────────────────
LORA_RANK = 8
LORA_ALPHA = 16
LORA_BLOCKS = [8, 9, 10, 11]   # last 4 transformer blocks

# ── Optimiser ──────────────────────────────────────────────────────────
LR_LORA = 1e-4
LR_HEAD = 1e-3
LR_DELTA = 1e-3
LR_LOGIT_SCALE = 1e-3
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# ── Loss weights (baseline) ───────────────────────────────────────────
LAMBDA_CE = 0.75
LAMBDA_UNIFORM = 0.10
LAMBDA_ONTO = 0.15        # swept during experiment
LAMBDA_PROTO = 0.05       # graph regulariser

# ── Dataset metadata ──────────────────────────────────────────────────
DATASETS = {
    "PlantVillage": {
        "num_classes": 38,
        "holdout_every_n": None,    # hold out 2 classes for open-set
        "holdout_count": 2,
    },
    "DeepWeeds": {
        "num_classes": 9,
        "holdout_every_n": None,
        "holdout_count": 2,
    },
    "IP102": {
        "num_classes": 102,
        "holdout_every_n": 10,      # every 10th class
        "holdout_count": None,
    },
}

# ── Lambda sweep values ────────────────────────────────────────────────
SWEEP_LAMBDAS = [0.025, 0.05, 0.075, 0.10, 0.125]
