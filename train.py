print("1")
import torch

print("2")
import torch.nn as nn

print("3")
import torch.optim as optim

print("4")
import config

print("5")
from dataset import get_dataloaders, class_names

print("6")
from model import build_model

print("7")
from engine import train_one_epoch, validate

print("8")
from utils import (
    plot_history,
    save_confusion_matrix,
    print_classification_report
)

print("Imports completed successfully")


import torch
import torch.nn as nn
import torch.optim as optim

import config

from dataset import get_dataloaders, class_names
from model import build_model
from engine import train_one_epoch, validate
from utils import (
    plot_history,
    save_confusion_matrix,
    print_classification_report
)


def main():

    print("=" * 70)
    print("        Intel Image Classifier Training")
    print("=" * 70)

    # =====================================
    # Load Dataset
    # =====================================

    train_loader, test_loader = get_dataloaders()

    print(f"Training Images : {len(train_loader.dataset)}")
    print(f"Testing Images  : {len(test_loader.dataset)}")
    print(f"Classes         : {class_names}")

    # =====================================
    # Build Model
    # =====================================

    model = build_model()

    print("\nModel Loaded Successfully")

    # =====================================
    # Loss Function
    # =====================================

    criterion = nn.CrossEntropyLoss()

    # =====================================
    # Optimizer
    # =====================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )

    # =====================================
    # Learning Rate Scheduler
    # =====================================

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.STEP_SIZE,
        gamma=config.GAMMA
    )

    # =====================================
    # Training History
    # =====================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    best_accuracy = 0.0

    best_labels = None
    best_predictions = None

    print("\nStarting Training...\n")

    # =====================================
    # Epoch Loop
    # =====================================

    for epoch in range(config.EPOCHS):

        print("=" * 70)
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        print("=" * 70)

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config.DEVICE
        )

        val_loss, val_acc, y_true, y_pred = validate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=config.DEVICE
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print("\nEpoch Summary")
        print("-" * 50)

        print(f"Train Loss          : {train_loss:.4f}")
        print(f"Training Accuracy   : {train_acc:.2f}%")
        print(f"Validation Loss     : {val_loss:.4f}")
        print(f"Validation Accuracy : {val_acc:.2f}%")
        print(f"Learning Rate       : {current_lr:.6f}")

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            best_labels = y_true
            best_predictions = y_pred

            torch.save(
                model.state_dict(),
                config.MODEL_PATH
            )

            print("\nBest Model Updated and Saved Successfully!")

            print()

    # =====================================
    # Training Completed
    # =====================================

    print("=" * 70)
    print("Training Completed Successfully")
    print("=" * 70)

    print(f"\nBest Validation Accuracy : {best_accuracy:.2f}%")

    # =====================================
    # Generate Graphs
    # =====================================

    print("\nGenerating Training Graphs...")

    plot_history(history)

    # =====================================
    # Generate Confusion Matrix
    # =====================================

    print("Generating Confusion Matrix...")

    save_confusion_matrix(
        best_labels,
        best_predictions,
        class_names
    )

    # =====================================
    # Print Classification Report
    # =====================================

    print("\nClassification Report")
    print("=" * 70)

    print_classification_report(
        best_labels,
        best_predictions,
        class_names
    )

    # =====================================
    # Final Summary
    # =====================================

    print("\n" + "=" * 70)
    print("Project Outputs")
    print("=" * 70)

    print(f"Best Model Saved At : {config.MODEL_PATH}")
    print(f"Plots Saved At      : {config.PLOT_DIR}")

    print("\nGenerated Files")

    print("✔ best_model.pth")
    print("✔ loss_curve.png")
    print("✔ accuracy_curve.png")
    print("✔ confusion_matrix.png")

    print("\nIntel Image Classifier Training Finished Successfully!")



if __name__ == "__main__":
    main()