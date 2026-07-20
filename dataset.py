from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import config


# =====================================
# Training Image Transformations
# =====================================

train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# Testing Image Transformations
# =====================================

test_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================
# Datasets
# =====================================

train_dataset = datasets.ImageFolder(
    root=config.TRAIN_DIR,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    root=config.TEST_DIR,
    transform=test_transform
)


# =====================================
# DataLoaders
# =====================================

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS
)


# =====================================
# Class Information
# =====================================

class_names = train_dataset.classes

num_classes = len(class_names)


# =====================================
# Function
# =====================================

def get_dataloaders():
    return train_loader, test_loader

