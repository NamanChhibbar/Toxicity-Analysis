import numpy as np, pandas as pd, torch, os
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
# from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from time import perf_counter

from configs import FLT_PREC, PROJECT_DIR, MODEL_PATH, TOKENIZER_PATH

def load_data(csv_paths):
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
    return np.random.permutation(data)

def load_model(model_path, tokenizer_path, device):
    if not os.path.exists(f"{PROJECT_DIR}/distilbert"):
        DistilBertTokenizer.from_pretrained("distilbert-base-uncased").save_pretrained(TOKENIZER_PATH)
        DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased").save_pretrained(MODEL_PATH)
        print(f"\nDistilbert model and tokenizer saved in directory {PROJECT_DIR}/distilbert\n")

    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    print(f"Model loaded from {MODEL_PATH}\n")
    return model, tokenizer

def train_test_split(data, train_test_ratio, val_test_ratio):
    train_test_split = int(len(data) * train_test_ratio)
    train_data = data[:train_test_split]
    test_data = data[train_test_split:]
    return train_data, test_data

def tokenize_and_batch(data, tokenizer, tokenizer_maxlen, batch_size=None, shuffle=True):
    texts, labels = data[:, 0], data[:, 1].astype(np.float64)
    tokenized = tokenizer(list(texts), padding=True, truncation=True, max_length=tokenizer_maxlen, return_tensors="pt")
    dataset = TensorDataset(tokenized["input_ids"], torch.tensor(labels).to(torch.int64), tokenized["attention_mask"])
    if batch_size:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset

def train_and_validate(
        train_data, val_data, model, tokenizer,
        tokenizer_maxlen, optimizer, scheduler, device,
        batch_size=32, epochs=20, shuffle=True
):

    train_set = tokenize_and_batch(train_data, tokenizer, tokenizer_maxlen, batch_size, shuffle=shuffle)
    val_inp, val_labels, val_mask = tokenize_and_batch(val_data, tokenizer, tokenizer_maxlen, shuffle=shuffle).tensors
    batches = len(train_set)

    train_loss, val_loss, val_accuracy, val_f1 = [], [], [], []
    for epoch in range(epochs):
        model.train(True)
        epoch_loss = 0
        for batch, (inp, labels, attention_mask) in enumerate(train_set):

            inp = inp.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            start_time = perf_counter()
            loss = model(input_ids=inp, labels=labels, attention_mask=attention_mask).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            time = (perf_counter() - start_time) * 1000

            print(
                f"\rEpoch: {epoch + 1}/{epochs} "
                f"Batch: {batch + 1}/{batches} "
                f"Time: {round(time, FLT_PREC)} ms "
                f"Loss: {round(loss.item(), FLT_PREC)}",
                end="\t\t"
            )
            epoch_loss += loss.item()
        train_loss.append(epoch_loss / batches)

        model.train(False)
        with torch.no_grad():
            output = model(input_ids=val_inp, labels=val_labels, attention_mask=val_mask)
        logits = output.logits
        predictions = torch.argmax(logits, dim=-1)
        val_loss.append(output.loss.item())
        val_accuracy.append(accuracy_score(val_labels, predictions))
        val_f1.append(f1_score(val_labels, predictions))
        print()
    print()

    return train_loss, val_loss, val_accuracy, val_f1

def toxicity_score(text, model, tokenizer, sent_maxlen):
    tokenized = tokenizer([text], max_length=sent_maxlen, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokenized)
    return torch.argmax(output.logits).item()
