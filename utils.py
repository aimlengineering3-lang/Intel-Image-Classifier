import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import config


def plot_history(history):

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss Graph
    plt.figure(figsize=(8,5))

    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")

    plt.legend()

    plt.savefig(config.PLOT_DIR / "loss_curve.png")

    plt.close()


    # Accuracy Graph
    plt.figure(figsize=(8,5))

    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")

    plt.legend()

    plt.savefig(config.PLOT_DIR / "accuracy_curve.png")

    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(xticks_rotation=45)

    plt.savefig(config.PLOT_DIR / "confusion_matrix.png")

    plt.close()


def print_classification_report(y_true, y_pred, class_names):

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )