import torch, matplotlib.pyplot as plt

from configs import (
    DATA_PATHS, MODEL, MODEL_DIR, PLOT_PATH,
    TRAIN_RATIO, VAL_RATIO, MAX_TOKENS, SHUFFLE,
    EPOCHS, BATCH_SIZE, INIT_LR, SCH_STEP, SCH_GAMMA,
    FLT_PREC, WHITE_SPACE
)
from utils import load_data, load_model, train_val_test_split, train_and_test

def main():
    print()

    # Load data
    data = load_data(DATA_PATHS, SHUFFLE)
    train_data, val_data, test_data = train_val_test_split(data, TRAIN_RATIO, VAL_RATIO)

    # Load model and tokenizer
    model, tokenizer = load_model(MODEL, MODEL_DIR)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCH_STEP, gamma=SCH_GAMMA, verbose=True)

    # Training loop
    train_loss, val_metrics, test_metrics = train_and_test(
        train_data, val_data, test_data, model, tokenizer,
        MAX_TOKENS, optimizer, scheduler,
        BATCH_SIZE, EPOCHS, FLT_PREC, WHITE_SPACE
    )

    # Plot metrics
    x = range(len(train_loss))
    plt.plot(x, train_loss, label="Train loss")
    plt.plot(x, val_metrics["loss"], label="Validation loss")
    plt.plot(x, val_metrics["accuracy"], label="Validation accuracy")
    plt.plot(x, val_metrics["f1"], label="Validation F1")
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(PLOT_PATH)
    print(f"Performance plot saved as {PLOT_PATH}")

    # Save fine-tuned model
    model.save_pretrained(f"{MODEL_DIR}/model")
    print(f"{MODEL} model trained and saved in {MODEL_DIR}\n")

    # Print test metrics
    print(
        "Performance metrics on test set:\n"
        f"Test loss = {round(test_metrics["loss"], FLT_PREC)}\n"
        f"Test accuracy = {round(test_metrics["accuracy"], FLT_PREC)}\n"
        f"Test F1 score = {round(test_metrics["f1"], FLT_PREC)}\n"
    )

if __name__ == "__main__":
    main()
    exit(0)
