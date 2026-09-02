import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# -----------------------------
# Configuration
# -----------------------------

TEST_DIR = "test"
MODEL_PATH = "models/emotion_model_v3.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# -----------------------------
# Preprocessing
# -----------------------------

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Dataset
# -----------------------------

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transforms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

classes = test_dataset.classes

print("Classes:", classes)
print("Test images:", len(test_dataset))

# -----------------------------
# Load model
# -----------------------------

model = models.mobilenet_v3_small(
    weights=None
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    7
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("Emotion model loaded successfully!")

# -----------------------------
# Predictions
# -----------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

# -----------------------------
# Classification report
# -----------------------------

print()
print("========================================")
print("EMOVERSE MODEL V3 - EVALUATION")
print("========================================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4,
        zero_division=0
    )
)

# -----------------------------
# Confusion matrix
# -----------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:")
print(cm)

# -----------------------------
# Display confusion matrix
# -----------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(
    xticks_rotation=45
)

plt.title(
  "EmoVerse - Model V3 Confusion Matrix"
)

plt.tight_layout()
plt.show()