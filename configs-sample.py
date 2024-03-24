import os

PROJECT_DIR = os.path.dirname(__file__)
MODEL_DIR = f"{PROJECT_DIR}/distilbert"

# Data paths
DATA_DIR = f"{PROJECT_DIR}/Sample-Data"
DATA_FILES = [
    "sample.csv"
]

# Float precision for printing
FLT_PREC = 4

# Data processing parameters
TRAIN_TEST_RATIO = 0.8
VAL_TEST_RATIO = 0.5
SENT_MAXLEN = 120

# Training parameters
EPOCHS = 40
BATCH_SIZE = 64
INIT_LR = 0.1
SCH_STEP = 8
SCH_GAMMA = 0.2
