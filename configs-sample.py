import os

PROJECT_DIR = os.path.dirname(__file__)
MODEL_DIR = f"{PROJECT_DIR}/distilbert"

# Data paths
DATA_DIR = f"{PROJECT_DIR}/Sample-Data"
DATA_FILES = [
    "sample.csv"
]

# Print settings
FLT_PREC = 4
WHITE_SPACE = 100

# Data processing parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.5
SENT_MAXLEN = 120
SHUFFLE = True

# Training parameters
EPOCHS = 10
BATCH_SIZE = 32
INIT_LR = 1e-3
SCH_GAMMA = 0.1
SCH_STEP = 5
