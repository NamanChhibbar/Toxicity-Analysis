import torch, matplotlib.pyplot as plt

from configs import (
    PROJECT_DIR, DATA_DIR, DATA_FILES, MODEL_DIR,
    TRAIN_RATIO, VAL_RATIO, SENT_MAXLEN, SHUFFLE,
    EPOCHS, BATCH_SIZE, INIT_LR, SCH_STEP, SCH_GAMMA
)
from utils import load_data, load_model, train_test_split, train_and_validate

def main():
    print()

    csv_paths = [f"{DATA_DIR}/{file}" for file in DATA_FILES]
    data = load_data(csv_paths, shuffle=SHUFFLE)
    train_data, val_data, test_data = train_test_split(data, TRAIN_RATIO, VAL_RATIO)

    model, tokenizer = load_model(MODEL_DIR)

    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCH_STEP, gamma=SCH_GAMMA, verbose=True)

    train_loss, val_metrics, test_metrics = train_and_validate(
        train_data, val_data, test_data, model, tokenizer,
        SENT_MAXLEN, optimizer, scheduler,
        BATCH_SIZE, EPOCHS
    )

    x = range(len(train_loss))
    plt.plot(x, train_loss, label="Train loss")
    plt.plot(x, val_metrics["loss"], label="Validation loss")
    plt.plot(x, val_metrics["accuracy"], label="Validation accuracy")
    plt.plot(x, val_metrics["f1"], label="Validation F1")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(f"{PROJECT_DIR}/model-performance.jpg")
    print(f"Performance plot saved as {PROJECT_DIR}/model-performance.jpg")

    model.save_pretrained(f"{MODEL_DIR}/model")
    print(f"Model trained and saved in {MODEL_DIR}\n")

    print(
        "Performance metrics on test set:\n"
        f"Test loss = {test_metrics["loss"]}\n"
        f"Test accuracy = {test_metrics["accuracy"]}\n"
        f"Test F1 score = {test_metrics["f1"]}\n"
    )

if __name__ == "__main__":
    main()
    exit(0)
