"""
Contains training loop configurations.
"""

# Path to directory containing data files relative to project directory
DATA_DIR = f"Sample-Data"

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
DATA_DIR = f"{PROJECT_DIR}/{DATA_DIR}"

MODEL_DIR = f"{PROJECT_DIR}/Models/{os.path.basename(MODEL)}"
PLOT_PATH = f"{PROJECT_DIR}/Performance-Plots/{os.path.basename(MODEL)}.jpg"

DATA_PATHS = [f"{DATA_DIR}/{file}" for file in os.listdir(DATA_DIR)]


#######################################################
