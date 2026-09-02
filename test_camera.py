import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not access the camera.")
    exit()

print("Camera connected successfully!")
print("Press Q to close the camera.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    cv2.imshow("EmoVerse - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
