import torch, matplotlib.pyplot as plt

from configs import (
    PROJECT_DIR, DATA_DIR, DATA_FILES, MODEL_DIR,
    TRAIN_RATIO, VAL_RATIO, SENT_MAXLEN, SHUFFLE,
    EPOCHS, BATCH_SIZE, INIT_LR, SCH_STEP, SCH_GAMMA
)
from utils import load_data, load_model, train_test_split, train_and_validate, test_model

def main():
    print()

    csv_paths = [f"{DATA_DIR}/{file}" for file in DATA_FILES]
    data = load_data(csv_paths, shuffle=SHUFFLE)
    train_data, val_data, test_data = train_test_split(data, TRAIN_RATIO, VAL_RATIO)

    model, tokenizer = load_model(MODEL_DIR)

    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCH_STEP, gamma=SCH_GAMMA, verbose=True)

    train_loss, val_loss, val_acc, val_f1 = train_and_validate(
        train_data, val_data, model, tokenizer,
        SENT_MAXLEN, optimizer, scheduler,
        BATCH_SIZE, EPOCHS
    )

    x = range(len(train_loss))
    plt.plot(x, train_loss, label="Train loss")
    plt.plot(x, val_loss, label="Validation loss")
    plt.plot(x, val_acc, label="Validation accuracy")
    plt.plot(x, val_f1, label="Validation F1")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(f"{PROJECT_DIR}/model-performance.jpg")
    print(f"Performance plot saved as {PROJECT_DIR}/model-performance.jpg")

    model.save_pretrained(f"{MODEL_DIR}/model")
    print(f"Model trained and saved in {MODEL_DIR}\n")

    test_loss, test_accuracy, test_f1 = test_model(test_data, model, tokenizer, SENT_MAXLEN)
    print(f"Performance metrics on test set:\nTest loss = {test_loss}\nTest accuracy = {test_accuracy}\nTest F1 score = {test_f1}")

if __name__ == "__main__":
    main()
    exit(0)
