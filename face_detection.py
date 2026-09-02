import cv2

MODEL = "models/face_detection_yunet_2023mar.onnx"

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access the camera.")
    exit()

# Create YuNet face detector
detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("EmoVerse YuNet Face Detection Started!")
print("Press Q to close.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    height, width = frame.shape[:2]

    # Tell detector the current camera resolution
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    face_count = 0 if faces is None else len(faces)

    # Draw detected faces
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            confidence = face[-1]

            cv2.putText(
                frame,
                f"Face {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Faces detected: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("EmoVerse - YuNet Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()