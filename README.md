# Toxicity Analysis

## Overview

This project focuses on analysing the toxicity and severity of toxic comments.

## Setup for model training

### Installing dependencies

Using a python virtual environment is highly recommended. Create one using the following command:

```zsh
python3 -m venv .venv
```

Activate the virtual environment using:

```zsh
source .venv/bin/activate
```

To install the required Python packages, run the following command:

```zsh
pip3 install -r requirements.txt
```

### Training configurations

Before training the model, ensure that data file paths, model parameters, and other configurations are set up correctly in `configs.py`. You can use the provided sample configuration file [configs-sample.py](configs-sample.py) as a template. Rename it to `configs.py` and update `DATA_DIR` accordingly.

## Training the model

The main training loop is implemented in [train.py](train.py). Execute it as a Python script to begin training. The model will be fetched from Hugging Face based on the configurations specified in [configs.py](configs.py) and will be saved in the model directory. If a model is already saved, it will be loaded from the directory itself. After training, the model will overwrite the pre-trained model. Additionally, the code will generate a graph visualizing the model performance on training and validation sets over epochs, saved in `Performance-Plots/`.

## Getting toxicity score on text

You can obtain the toxicity score for your text using the `toxicity_score` function provided in [utils.py](utils.py).
