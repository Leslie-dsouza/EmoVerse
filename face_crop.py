import cv2

MODEL = "models/face_detection_yunet_2023mar.onnx"

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access the camera.")
    exit()

# Create YuNet detector
detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (320, 320),
    0.6,
    0.3,
    5000
)

print("EmoVerse Face Cropping Started!")
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

            # Face bounding box
            x, y, w, h = face[:4].astype(int)

            # Make sure coordinates stay inside the image
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            # Crop the face
            face_crop = frame[y:y+h, x:x+w]

            # Draw rectangle on original frame
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Label each person
            cv2.putText(
                frame,
                f"Person {i + 1}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Display the cropped face
            if face_crop.size > 0:
                cv2.imshow(
                    f"Face {i + 1}",
                    face_crop
                )

    # Display original camera
    cv2.imshow(
        "EmoVerse - Face Cropping",
        frame
    )

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()