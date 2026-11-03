# IntelVision AI — Terrain Image Classifier

A deep learning project that classifies natural and urban scene photographs into one of six categories using transfer learning with a pretrained ResNet18 network — wrapped in an interactive Streamlit dashboard.

🔗 **Live Demo:** [intel-image-classifier-estrderc6qb9tfjfwv2pcf.streamlit.app](https://intel-image-classifier-estrderc6qb9tfjfwv2pcf.streamlit.app/)

> This is a learning / portfolio project, not a production system. It was built to practice the full computer vision pipeline — from raw data to a usable interface — while following clean, professional project structure.

---

## Overview

IntelVision AI takes a photograph and predicts which of six terrain categories it belongs to, along with a confidence score and the full probability breakdown across all classes:

| Class | Description |
|---|---|
| 🏙 Buildings | Urban structures, facades, dense city blocks |
| 🌲 Forest | Dense canopy cover, woodland texture |
| 🧊 Glacier | Compacted ice fields, high-altitude snowpack |
| ⛰ Mountain | Elevated rock formations, ridgelines, peaks |
| 🌊 Sea | Open water, coastline, seascape |
| 🛣 Street | Paved thoroughfare, road markings, signage |

The model is trained on the **Intel Image Classification** dataset (~25,000 labeled images), originally released as part of an Intel / Analytics Vidhya challenge.

---

## How It Works

The project uses **transfer learning**: instead of training a convolutional neural network from scratch, it starts from a ResNet18 model pretrained on ImageNet (1.2M images, 1000 classes), which already understands general visual features like edges, textures, and shapes.

1. **Freeze the pretrained backbone** — all original convolutional layers are kept frozen so their learned features aren't disturbed.
2. **Replace the final layer** — the last fully-connected layer is swapped from a 1000-class output to a 6-class output matching this dataset.
3. **Train only the new layer** — the classifier head is trained from scratch on the Intel Image dataset while the backbone stays fixed.

This approach converges fast, needs far less data and compute than training from scratch, and carries a lower risk of overfitting.

### Pipeline

```
ImageFolder → Transforms (resize, normalize, augment) → DataLoader
         → ResNet18 (frozen backbone) → Fully Connected (512 → 6)
         → Softmax → Class + Confidence
```

---

## Project Structure

```
├── app.py              # Streamlit dashboard (prediction UI + analytics)
├── config.py            # Paths, hyperparameters, device, class names
├── dataset.py            # Dataset loading, transforms, DataLoaders
├── model.py              # ResNet18-based classifier definition
├── engine.py              # Training and validation loops
├── train.py                # Main training script
├── predict.py              # Standalone CLI prediction script
├── utils.py                # Plotting, confusion matrix, classification report
├── test_dataset.py          # Sanity check for data loading
├── test_model.py             # Sanity check for model forward pass
├── requirements.txt           # Python dependencies
├── models/                     # Saved model checkpoints (best_model.pth)
└── outputs/
    └── plots/                   # Training curves & confusion matrix
```

---

## Model Details

- **Architecture:** ResNet18 (pretrained on ImageNet, fine-tuned final layer)
- **Input size:** 224 × 224 × 3
- **Classes:** 6
- **Loss function:** CrossEntropyLoss
- **Optimizer:** Adam
- **LR Scheduler:** StepLR
- **Augmentation:** Random horizontal flip, random rotation
- **Normalization:** ImageNet mean/std — `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`

Training produces:
- `best_model.pth` — best-performing checkpoint on validation accuracy
- `loss_curve.png` / `accuracy_curve.png` — training history
- `confusion_matrix.png` — per-class prediction breakdown
- A full classification report (precision, recall, F1-score)

---

## Dashboard Features

Built with Streamlit, the dashboard includes:

- **Image Prediction** — upload a photo and get a live classification with confidence score and full class probability chart
- **Training Analytics** — view the actual accuracy curve, loss curve, and confusion matrix from the training run
- **Model Information** — architecture breakdown and inference pipeline explanation
- **About Dataset** — overview of the six classes and dataset source

---

## Running Locally

```bash
# Clone the repo
git clone <repo-url>
cd <repo-folder>

# Install dependencies
pip install -r requirements.txt

# (Optional) Train the model from scratch
python train.py

# Launch the dashboard
streamlit run app.py
```

> Note: `dataset/` and `models/best_model.pth` are not included in this repo. To train, place the Intel Image Classification dataset in `dataset/seg_train` and `dataset/seg_test` following the `ImageFolder` structure. Without a trained checkpoint, the dashboard runs in demo mode.

---

## Why I Built This

This project was built to strengthen hands-on, end-to-end computer vision skills — dataset handling, transfer learning, model evaluation, and shipping a working interface rather than just a notebook. It reflects how I approach structuring real projects: modular code, clear separation of concerns, and documented results.

---

## Tech Stack

`PyTorch` · `Torchvision` · `Streamlit` · `Plotly` · `scikit-learn` · `Pillow`
