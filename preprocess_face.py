import cv2
import numpy as np

MODEL = "models/face_detection_yunet_2023mar.onnx"

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access the camera.")
    exit()

# YuNet face detector
detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("EmoVerse Face Preprocessing Started!")
print("Press Q to close.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    if faces is not None:

        for i, face in enumerate(faces):

            x, y, w, h = face[:4].astype(int)

            # Keep coordinates inside image
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            # Crop face
            face_crop = frame[y:y+h, x:x+w]

            if face_crop.size == 0:
                continue

            # Resize face to 224 x 224
            face_resized = cv2.resize(
                face_crop,
                (224, 224)
            )

            # Convert BGR → RGB
            face_rgb = cv2.cvtColor(
                face_resized,
                cv2.COLOR_BGR2RGB
            )

            # Normalize pixel values
            face_normalized = face_rgb.astype(
                np.float32
            ) / 255.0

            # Draw detection box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Person {i + 1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Show processed face
            display_face = cv2.cvtColor(
                face_normalized,
                cv2.COLOR_RGB2BGR
            )

            display_face = (
                display_face * 255
            ).astype(np.uint8)

            cv2.imshow(
                f"Processed Face {i + 1}",
                display_face
            )

    cv2.imshow(
        "EmoVerse - Preprocessing",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()