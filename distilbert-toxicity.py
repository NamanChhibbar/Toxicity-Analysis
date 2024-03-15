import numpy as np, pandas as pd, torch, warnings, matplotlib.pyplot as plt, os
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from time import perf_counter

warnings.filterwarnings("ignore")

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROJECT_DIR = "/Users/naman/Desktop/Code/Projects/ToxicityAnalysis"
    DATA_DIR = f"{PROJECT_DIR}/Labeled_data"
    DATA_FILES = [
        "1coviddetoxify0.5.csv",
        "Andhbhaktafterdetoxify.csv",
        "GobackModifinallabels1.csv",
        "Gyanvapiafterdetoxifyandcorrectedlabels.csv",
        "indianmusafterdetoxify-1.csv",
        "indianmusafterdetoxify.csv"
    ]

    MODEL_CHECKPOINT = f"{PROJECT_DIR}/pretrained-model" if os.path.exists("pretrained-model") else "distilbert-base-uncased"
    EPOCHS = 20
    BATCH_SIZE = 128
    SPLITS = 5
    SENTENCE_MAXLEN = 80
    INIT_LR = 1e-1
    SCH_STEP = 5
    SCH_GAMMA = 0.1

    csv_paths = [f"{DATA_DIR}/{file}" for file in DATA_FILES]

    all_texts, all_labels = load_data(csv_paths)

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_CHECKPOINT)
    model.to(device)
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCH_STEP, gamma=SCH_GAMMA, verbose=True)

    test_losses, accuracy_scores, f1_scores = train_and_test(
        all_texts, all_labels,
        model, tokenizer, SENTENCE_MAXLEN, optimizer, scheduler,
        EPOCHS, device, BATCH_SIZE, SPLITS
    )

    model.save_pretrained(f"{PROJECT_DIR}/pretrained-model")

    x = range(len(test_losses))
    plt.plot(x, test_losses, label="Test loss")
    plt.plot(x, accuracy_scores, label="Accuracy")
    plt.plot(x, f1_scores, label="F1 score")
    plt.legend()
    plt.show()

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
    # zipped_texts_labels = list(zip(all_texts, all_labels))
    # sorted_texts_labels = sorted(zipped_texts_labels, key=lambda x: len(x[0]))
    # all_texts = [text for text, _ in sorted_texts_labels]
    # all_labels = [label for _, label in sorted_texts_labels]
    return np.array(all_texts), np.array(all_labels, dtype=np.int64)

def tokenize_and_batch(texts, labels, tokenizer, tokenizer_maxlen, batch_size=None, shuffle=True):
    tokenized = tokenizer(list(texts), padding=True, truncation=True, max_length=tokenizer_maxlen, return_tensors="pt")
    dataset = TensorDataset(tokenized["input_ids"], torch.tensor(labels), tokenized["attention_mask"])
    batch_size = batch_size if batch_size else len(dataset)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset

def train_and_test(all_texts, all_labels, model, tokenizer, tokenizer_maxlen, optimizer, scheduler, epochs, device, batch_size=32, splits=5, shuffle=True):
    test_losses, accuracy_scores, f1_scores = [], [], []

    kfold = StratifiedKFold(n_splits=splits)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(all_texts, all_labels)):

        train_texts = all_texts[train_idx]
        train_labels = all_labels[train_idx]
        test_texts = all_texts[test_idx]
        test_labels = all_labels[test_idx]
        train_set = tokenize_and_batch(train_texts, train_labels, tokenizer, tokenizer_maxlen, batch_size, shuffle=shuffle)
        test_set = tokenize_and_batch(test_texts, test_labels, tokenizer, tokenizer_maxlen, shuffle=shuffle)
        batches = len(train_set)

        model.train(True)
        for epoch in range(epochs):
            for batch, (inp, labels, attention_mask) in enumerate(train_set):

                start_time = perf_counter()
                inp = inp.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)
                model_out = model(inp, labels=labels, attention_mask=attention_mask)
                loss = model_out.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                print(
                    f"\rFold: {fold + 1}/{splits} "
                    f"Epoch: {epoch + 1}/{epochs} "
                    f"Batch: {batch + 1}/{batches} "
                    f"Time: {(perf_counter() - start_time) * 1000} ms "
                    f"Loss: {loss.item()}",
                    end=""
                )
            print()
        
        print()
        model.train(False)
        test_inp, true_labels, test_mask = test_set
        with torch.no_grad():
            output = model(test_inp, attention_mask=test_mask)
        logits = output.logits
        predictions = torch.argmax(logits, dim=-1)
        test_loss = output.loss.item()
        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        print(
            f"Metrics for fold {fold + 1}/{splits}",
            f"Model loss: {test_loss}"
            f"Model accuracy: {accuracy}",
            f"F1 score: {f1}",
            sep="\n"
        )
        test_losses.append(test_loss)
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)
    
    return test_losses, accuracy_scores, f1_scores

if __name__ == "__main__":
    main()
    exit(0)
