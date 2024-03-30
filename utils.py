import numpy as np, pandas as pd, torch, os, warnings
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertForSequenceClassification as Model, DistilBertTokenizer as Tokenizer
from sklearn.metrics import f1_score, accuracy_score
from time import perf_counter

from configs import FLT_PREC, WHITE_SPACE

warnings.filterwarnings("ignore")

def load_data(csv_paths, shuffle=False):
    all_texts, all_labels = [], []
    for path in csv_paths:
        df = pd.read_csv(path)
        filter = pd.notnull(df["manual_label"])
        texts = df["text"][filter].tolist()
        labels = df["manual_label"][filter].tolist()
        all_texts.extend(texts)
        all_labels.extend(labels)
        print(f"File loaded: {path}")
    print()
    data = np.stack((all_texts, all_labels), axis=1)
    return np.random.permutation(data) if shuffle else data

def load_model(model_dir):
    tokenizer_path = f"{model_dir}/tokenizer"
    model_path = f"{model_dir}/model"
    if not os.path.exists(model_dir):
        Tokenizer.from_pretrained("distilbert-base-uncased").save_pretrained(tokenizer_path)
        Model.from_pretrained("distilbert-base-uncased").save_pretrained(model_path)
        print(f"\nDistilbert model and tokenizer saved in directory {model_dir}\n")

    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    model = Model.from_pretrained(model_path)
    print(f"Model loaded from {model_dir}\n")
    return model, tokenizer

def train_test_split(data, train_ratio, val_ratio):
    train_split = int(len(data) * train_ratio)
    train_data = data[:train_split]
    test_data = data[train_split:]
    val_split = int(val_ratio * len(test_data))
    val_data = test_data[:val_split]
    test_data = test_data[val_split:]
    return train_data, val_data, test_data

def tokenize_and_batch(data, tokenizer, sent_maxlen, batch_size=None):
    texts, labels = data[:, 0], data[:, 1].astype(np.float64)
    tokenized = tokenizer(list(texts), padding=True, truncation=True, max_length=sent_maxlen, return_tensors="pt")
    dataset = TensorDataset(tokenized["input_ids"], torch.tensor(labels).to(torch.int64), tokenized["attention_mask"])
    if not batch_size:
        return dataset.tensors
    return DataLoader(dataset, batch_size=batch_size)

def train_and_validate(
        train_data, val_data, model, tokenizer,
        sent_maxlen, optimizer, scheduler=None,
        batch_size=32, epochs=10
):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_data = tokenize_and_batch(train_data, tokenizer, sent_maxlen, batch_size)
    val_data = tokenize_and_batch(val_data, tokenizer, sent_maxlen)
    batches = len(train_data)
    train_loss, val_loss, val_accuracy, val_f1 = [], [], [], []
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

            print("\r", " " * WHITE_SPACE, end="\r")
            if batch + 1 == batches: print(
                f"Epoch {epoch + 1} average loss: {round(epoch_loss / epochs, FLT_PREC)}"
            )
            else: print(
                f"Epoch: {epoch + 1}/{epochs} "
                f"Batch: {batch + 1}/{batches} "
                f"Time: {round(time, FLT_PREC)} ms/batch "
                f"Loss: {round(loss.item(), FLT_PREC)} "
                f"Time remaining: {hours}h {minutes}m {seconds}s",
                end="\t\t"
            )
        train_loss.append(epoch_loss / batches)

        val_loss_, val_accuracy_, val_f1_ = test_model(val_data, model, device)
        val_loss.append(val_loss_)
        val_accuracy.append(val_accuracy_)
        val_f1.append(val_f1_)
        print(
            f"Validation loss: {round(val_loss_, FLT_PREC)}\n"
            f"Validation accuracy: {round(val_accuracy_, FLT_PREC)}\n"
            f"Validation F1 = {round(val_f1_, FLT_PREC)}\n"
        )

    return train_loss, val_loss, val_accuracy, val_f1

def test_model(test_data, model, device, tokenizer=None, sent_maxlen=None):
    if tokenizer:
        test_data = tokenize_and_batch(test_data, tokenizer, sent_maxlen)
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

def toxicity_score(text, model, tokenizer, sent_maxlen):
    tokenized = tokenizer([text], max_length=sent_maxlen, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokenized)
    return torch.nn.functional.softmax(output.logits)[0, 1].item()
