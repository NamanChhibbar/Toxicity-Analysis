"""
Contains training loop configurations.
"""

# Directory containing data files
DATA_DIR = "Sample-Data"

# Directory where models will be saved
MODEL_DIR = "Models"

# Directory where plots will be saved
PLOT_DIR = "Plots"

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

MODEL_DIR = f"{MODEL_DIR}/{os.path.basename(MODEL)}"

PLOT_PATH = f"{PLOT_DIR}/{os.path.basename(MODEL)}.jpg"

DATA_PATHS = [
    f"{DATA_DIR}/{file}"
    for file in os.listdir(DATA_DIR)
    if os.path.isfile(f"{DATA_DIR}/{file}")
]


#######################################################
