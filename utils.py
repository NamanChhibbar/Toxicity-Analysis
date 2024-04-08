"""
Contains utility functions for data and model loading, data pre-processing, model training and testing.
"""

import numpy as np, pandas as pd, torch, os, warnings
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification as Model, AutoTokenizer as Tokenizer
from sklearn.metrics import f1_score, accuracy_score
from time import perf_counter

warnings.filterwarnings("ignore")

def get_device():
    """
    ## Returns
    `torch.device`: cuda or mps device if available, else cpu
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_data(data_paths, shuffle=False):
    """
    Loads and shuffles data in a numpy array. Data should be in csv or xlsx format with a
    "text" column containing the text and "toxicity_label" column containing the label.

    ## Parameters
    `data_paths`: List of strings containing path to data files
    `shuffle`: Boolean indicating whether to shuffle data

    ## Returns
    `np.ndarray` of shape `(None,)`
    """
    all_texts, all_labels = [], []
    for path in data_paths:
        _, extension = os.path.splitext(path)
        match extension:
            case ".csv": df = pd.read_csv(path)
            case ".xlsx": df = pd.read_excel(path)
            case _: continue
        filter = pd.notnull(df["text"]) & pd.notnull(df["toxicity_label"])
        texts = df["text"][filter].tolist()
        labels = df["toxicity_label"][filter].tolist()
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"File loaded: {path}")
    print()
    data = np.stack((all_texts, all_labels), axis=1)
    print(f"Data loaded of shape {data.shape}\n")
    return np.random.permutation(data) if shuffle else data

def load_model(model, model_dir):
    """
    Loads a Hugging Face model and tokenizer from the given checkpoint or a local directory.

    ## Parameters
    `model`: Hugging Face checkpoint to download the model, if not found
    `model_dir`: Path to local directory where model and tokenizer are saved. Model and
    tokenizer should be saved as `{model_dir}/Model` and `{model_dir}/Tokenizer`

    ## Returns
    `model, tokenizer` of type `transformers.AutoModelForSequenceClassification` and
    `transformers.AutoTokenizer`
    """
    tokenizer_path = f"{model_dir}/tokenizer"
    model_path = f"{model_dir}/model"
    if not os.path.exists(model_dir):
        Tokenizer.from_pretrained(model).save_pretrained(tokenizer_path)
        Model.from_pretrained(model).save_pretrained(model_path)
        print(f"\n{model} model and tokenizer saved in directory {model_dir}\n")

    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    model = Model.from_pretrained(model_path)
    print(f"Model loaded from {model_dir}\n")
    return model, tokenizer

def train_val_test_split(data, train_ratio, val_ratio):
    """
    Splits the data into train, validation, and test sets.

    ## Parameters
    `data`: Data to split
    `train_ratio`: Fraction of train data in the whole data
    `val_ratio`: Fraction of validation data in validation + test data

    ## Returns
    `train_data, val_data, test_data`
    """
    train_split = int(len(data) * train_ratio)
    train_data = data[:train_split]
    test_data = data[train_split:]
    val_split = int(val_ratio * len(test_data))
    val_data = test_data[:val_split]
    test_data = test_data[val_split:]
    return train_data, val_data, test_data

def tokenize_and_batch(data, tokenizer, max_tokens, batch_size=None):
    """
    Tokenizes and batches data.

    ## Parameters
    `data`: Data to be tokenized and batched, should be of shape `(None, None)`, texts in first column and labels in second
    `tokenizer`: Tokenizer to tokenize texts
    `max_tokens`: Maximum tokens in tokenized sentence
    `batch_size`: Size of batches, only 1 batch created if set to `None`
    """
    texts, labels = data[:, 0], data[:, 1].astype(np.float64)
    tokenized = tokenizer(list(texts), padding=True, truncation=True, max_length=max_tokens, return_tensors="pt")
    dataset = TensorDataset(tokenized["input_ids"], torch.tensor(labels).to(torch.int64), tokenized["attention_mask"])
    if batch_size:
        return DataLoader(dataset, batch_size=batch_size)
    return dataset.tensors

def train_and_test(
        train_data, val_data, test_data, model, tokenizer,
        max_tokens, optimizer, scheduler=None,
        batch_size=32, epochs=10, flt_prec=4, white_space=100
):
    """
    Trains and validates a Hugging Face model. Train, validation, and test data should
    be of the shape `(None, None)` with texts in first column and labels in second.

    ## Parameters
    `train_data`: Data to be used for training
    `val_data`: Data to be used for validation
    `test_data`: Data to be used for testing
    `model`: Hugging Face model to be trained
    `tokenizer`: Appropriate tokenizer for `model`
    `max_tokens`: Maximum tokens in tokenized sentence
    `optimizer`: Optimizer to use for `model`
    `scheduler`: Scheduler to use for `optimizer`
    `batch_size`: Batch size for training data
    `epochs`: Number of epochs to train for
    `flt_prec`: Floating point precision for stdout
    `white_space`: Number of white spaces to print for clearing stdout

    ## Returns
    `train_loss`: List containing average training set loss over an epoch
    `val_metrics`: Dictionary containing average validation loss, accuracy, and f1 score over an epoch
    `test_metrics`: Dictionary containing test loss, accuracy, and f1 score
    """
    device = get_device()

    train_data = tokenize_and_batch(train_data, tokenizer, max_tokens, batch_size)
    val_data = tokenize_and_batch(val_data, tokenizer, max_tokens)
    test_data = tokenize_and_batch(test_data, tokenizer, max_tokens)
    batches = len(train_data)
    train_loss = []
    val_metrics = {"loss": [], "accuracy": [], "f1": []}
    model.to(device)

    for epoch in range(epochs):
        model.train(True)
        epoch_loss = 0
        for batch, (inp, labels, attention_mask) in enumerate(train_data):

            inp = inp.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            start_time = perf_counter()
            loss = model(input_ids=inp, labels=labels, attention_mask=attention_mask).loss
            loss.backward()
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad()
            time = (perf_counter() - start_time) * 1000
            epoch_loss += loss.item()

            time_remaining = (time * ((epochs - epoch) * batches - batch - 1)) // 1000
            seconds = time_remaining % 60
            minutes = (time_remaining // 60) % 60
            hours = time_remaining // 3600

            print("\r", " " * white_space, end="\r")
            if batch + 1 == batches: print(
                f"Epoch {epoch + 1} average loss: {round(epoch_loss / epochs, flt_prec)}"
            )
            else: print(
                f"Epoch: {epoch + 1}/{epochs} "
                f"Batch: {batch + 1}/{batches} "
                f"Time: {round(time, flt_prec)} ms/batch "
                f"Loss: {round(loss.item(), flt_prec)} "
                f"Time remaining: {hours}h {minutes}m {seconds}s",
                end="\t\t"
            )
        train_loss.append(epoch_loss / batches)

        val_loss, val_accuracy, val_f1 = test_model(val_data, model, device)
        val_metrics["loss"].append(val_loss)
        val_metrics["accuracy"].append(val_accuracy)
        val_metrics["f1"].append(val_f1)
        print(
            f"Validation loss: {round(val_loss, flt_prec)}\n"
            f"Validation accuracy: {round(val_accuracy, flt_prec)}\n"
            f"Validation F1 = {round(val_f1, flt_prec)}\n"
        )
    test_loss, test_accuracy, test_f1 = test_model(test_data, model, device)
    test_metrics = {"loss": test_loss, "accuracy": test_accuracy, "f1": test_f1}

    return train_loss, val_metrics, test_metrics

def test_model(test_data, model, device=torch.device("cpu")):
    """
    Tests model on testing data using metrics loss, accuracy, and f1 score.

    ## Parameters
    `test_data`: Testing data to be used
    `model`: Model to be tested
    `device`: PyTorch device to use

    ## Returns
    `loss, accuracy, f1_score`
    """
    test_inp, test_labels, test_mask = test_data
    test_inp = test_inp.to(device)
    test_labels = test_labels.to(device)
    test_mask = test_mask.to(device)
    model.train(False)
    with torch.no_grad():
        output = model(input_ids=test_inp, labels=test_labels, attention_mask=test_mask)
    test_labels = test_labels.cpu()
    predictions = torch.argmax(output.logits, dim=-1).cpu()
    return output.loss.item(), accuracy_score(test_labels, predictions), f1_score(test_labels, predictions)

def toxicity_score(text, model, tokenizer, max_tokens):
    """
    Returns the toxicity score of a single sentence.

    ## Parameters
    `text`: Sentence
    `model`: Model to use
    `tokenizer`: Tokenizer associated to `model`
    `max_tokens`: Maximum tokens in tokenized sentence

    ## Returns
    `float` between 0 and 1
    """
    tokenized = tokenizer([text], max_length=max_tokens, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokenized)
    return torch.nn.functional.softmax(output.logits)[0, 1].item()
