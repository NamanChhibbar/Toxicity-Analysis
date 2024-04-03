"""
Contains training loop configurations.
"""

# Data paths
DATA_DIR = f"Sample-Data"
DATA_FILES = [
    "sample.csv"
]

# Model to use
# Should be a valid Hugging Face checkpoint
MODEL = "distilbert/distilbert-base-uncased"

# Data pre-processing parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.5
MAX_TOKENS = 120
SHUFFLE = True

# Training parameters
EPOCHS = 10
BATCH_SIZE = 32
INIT_LR = 1e-3
SCH_GAMMA = 0.1
SCH_STEP = 5

# Print settings
FLT_PREC = 4
WHITE_SPACE = 100




#################### DO NOT CHANGE ####################


import os

PROJECT_DIR = os.path.dirname(__file__)

MODEL_DIR = f"{PROJECT_DIR}/Models/{os.path.basename(MODEL)}"
PLOT_PATH = f"{PROJECT_DIR}/Performance-Plots/{os.path.basename(MODEL)}.jpg"

DATA_PATHS = [f"{PROJECT_DIR}/{DATA_DIR}/{file}" for file in DATA_FILES]


#######################################################
