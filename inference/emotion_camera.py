import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import glob

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "models/emotion_model_v3.pth"

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# ==========================================
# Load Emotion Model
# ==========================================

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

# ==========================================
# Image preprocessing
# ==========================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Find YuNet model
# ==========================================

yunet_files = glob.glob(
    "models/*yunet*.onnx"
)

if len(yunet_files) == 0:
    print("ERROR: YuNet model not found.")
    print("Put the YuNet .onnx file inside models/")
    exit()

YUNET_PATH = yunet_files[0]

print("Using YuNet:", YUNET_PATH)

# ==========================================
# Create YuNet detector
# ==========================================

detector = cv2.FaceDetectorYN.create(
    YUNET_PATH,
    "",
    (320, 240),
    0.7,
    0.3,
    5000
)

# ==========================================
# Start camera
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access camera.")
    exit()

print("Camera connected successfully!")
print("Press Q to quit.")

# ==========================================
# Real-time loop
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    # Update YuNet input size
    detector.setInputSize(
        (width, height)
    )

    # Detect faces
    _, faces = detector.detect(frame)

    face_count = 0

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4].astype(int)

            # Make sure coordinates stay inside frame
            x = max(0, x)
            y = max(0, y)

            w = min(w, width - x)
            h = min(h, height - y)

            if w <= 0 or h <= 0:
                continue

            # Crop face
            face_crop = frame[
                y:y+h,
                x:x+w
            ]

            if face_crop.size == 0:
                continue

            # Convert BGR -> RGB
            face_rgb = cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2RGB
            )

            # Convert to PIL image
            from PIL import Image

            face_image = Image.fromarray(
                face_rgb
            )

            # Preprocess
            input_tensor = transform(
                face_image
            ).unsqueeze(0).to(DEVICE)

            # Predict emotion
            with torch.no_grad():

                output = model(
                    input_tensor
                )

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                confidence, prediction = torch.max(
                    probabilities,
                    1
                )

            emotion = EMOTIONS[
                prediction.item()
            ]

            confidence_value = (
                confidence.item() * 100
            )

            face_count += 1

            # ==================================
            # Draw face rectangle
            # ==================================

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

            # ==================================
            # Draw emotion label
            # ==================================

            label = (
                f"{emotion.upper()} "
                f"{confidence_value:.1f}%"
            )

            cv2.putText(
                frame,
                label,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    # ==========================================
    # Face counter
    # ==========================================

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # ==========================================
    # Display
    # ==========================================

    cv2.imshow(
        "EmoVerse - Real-Time Emotion Detection",
        frame
    )

    # Quit with Q
    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):
        break

# ==========================================
# Cleanup
# ==========================================

camera.release()
cv2.destroyAllWindows()

print("EmoVerse camera stopped.")
