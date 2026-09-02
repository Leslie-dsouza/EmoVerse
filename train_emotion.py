import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# -----------------------------
# Configuration
# -----------------------------

TRAIN_DIR = "train"
MODEL_DIR = "models"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 0.0001

PATIENCE = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# -----------------------------
# Training transforms
# -----------------------------

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load dataset
# -----------------------------

full_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transforms
)

print("Classes:", full_dataset.classes)
print("Total images:", len(full_dataset))

# -----------------------------
# Validation split
# -----------------------------

validation_size = int(0.15 * len(full_dataset))
training_size = len(full_dataset) - validation_size

train_dataset, validation_dataset = random_split(
    full_dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training split:", len(train_dataset))
print("Validation split:", len(validation_dataset))

# -----------------------------
# Data loaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# Model
# -----------------------------

weights = models.MobileNet_V3_Small_Weights.DEFAULT

model = models.mobilenet_v3_small(
    weights=weights
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    7
)

model = model.to(DEVICE)

# -----------------------------
# Class weights
# -----------------------------

class_counts = torch.zeros(7)

for _, label in full_dataset.samples:
    class_counts[label] += 1

class_weights = 1.0 / class_counts

class_weights = (
    class_weights / class_weights.sum() * 7
)

class_weights = class_weights.to(DEVICE)

print("Class weights:", class_weights)

# -----------------------------
# Loss
# -----------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)

# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

# -----------------------------
# Learning-rate scheduler
# -----------------------------

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)

# -----------------------------
# Best model settings
# -----------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

best_model_path = os.path.join(
    MODEL_DIR,
    "emotion_model_v3.pth"
)

best_validation_accuracy = 0.0
epochs_without_improvement = 0

# -----------------------------
# Training loop
# -----------------------------

for epoch in range(EPOCHS):

    # =========================
    # Training
    # =========================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        100 * correct / total
    )

    # =========================
    # Validation
    # =========================

    model.eval()

    validation_correct = 0
    validation_total = 0
    validation_loss = 0.0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            validation_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            validation_total += labels.size(0)

            validation_correct += (
                predicted == labels
            ).sum().item()

    validation_accuracy = (
        100 *
        validation_correct /
        validation_total
    )

    average_train_loss = (
        running_loss /
        len(train_loader)
    )

    average_validation_loss = (
        validation_loss /
        len(validation_loader)
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print()
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{average_validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{validation_accuracy:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.6f}"
    )

    # -------------------------
    # Scheduler
    # -------------------------

    scheduler.step(
        validation_accuracy
    )

    # -------------------------
    # Save best model
    # -------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "✓ Best V3 model saved!"
        )

    else:

        epochs_without_improvement += 1

        print(
            f"No improvement "
            f"({epochs_without_improvement}/"
            f"{PATIENCE})"
        )

    # -------------------------
    # Early stopping
    # -------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            "Early stopping triggered."
        )

        break

# -----------------------------
# Finished
# -----------------------------

print()
print("==============================")
print("MODEL V3 TRAINING COMPLETE")
print("==============================")

print(
    f"Best Validation Accuracy: "
    f"{best_validation_accuracy:.2f}%"
)

print(
    f"Model saved to: "
    f"{best_model_path}"
)