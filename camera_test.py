
import cv2

CAMERA_SOURCE = 0

camera = cv2.VideoCapture(CAMERA_SOURCE)

if not camera.isOpened():
    print("Camera did not open. Check your DroidCam connection.")
else:
    print("Camera connected! Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read camera frame.")
            break

        cv2.imshow("SmartSight AI - Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
