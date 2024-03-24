import torch, matplotlib.pyplot as plt

from configs import *
from utils import load_data, load_model, train_test_split, train_and_validate

def main():
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_paths = [f"{DATA_DIR}/{file}" for file in DATA_FILES]
    data = load_data(csv_paths)
    train_data, test_data = train_test_split(data, TRAIN_TEST_RATIO, VAL_TEST_RATIO)

    model, tokenizer = load_model(MODEL_DIR, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCH_STEP, gamma=SCH_GAMMA, verbose=True)

    train_loss, val_loss, val_acc, val_f1 = train_and_validate(
        train_data, test_data, model, tokenizer,
        SENT_MAXLEN, optimizer, scheduler, device,
        BATCH_SIZE, EPOCHS
    )

    model.save_pretrained(f"{MODEL_DIR}/model")
    print(f"Model trained and saved in {MODEL_DIR}")

    x = range(len(train_loss))
    plt.plot(x, train_loss, label="Train loss")
    plt.plot(x, val_loss, label="Validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{PROJECT_DIR}/train-validation-loss.jpg")

if __name__ == "__main__":
    main()
    exit(0)
