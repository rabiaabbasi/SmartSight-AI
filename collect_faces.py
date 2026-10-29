
import cv2
import argparse
import time
from pathlib import Path

from src.face_utils import detect_faces

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    required=True,
    help="Name or ID of the consenting volunteer"
)
parser.add_argument(
    "--camera",
    default="0",
    help="Camera index or DroidCam stream URL"
)
args = parser.parse_args()

person_name = args.name.strip()

# Allow simple folder names only.
if not person_name or any(
    c in person_name for c in r'\/:*?"<>|'
):
    raise ValueError("Please provide a valid folder name.")

save_dir = Path("data") / "faces" / person_name
save_dir.mkdir(parents=True, exist_ok=True)

source = int(args.camera) if args.camera.isdigit() else args.camera
camera = cv2.VideoCapture(source)

if not camera.isOpened():
    raise RuntimeError(
        "Camera did not open. Check camera_test.py first."
    )

saved = 0
max_images = 150
last_saved = 0

print("Look toward the camera and move your head slightly.")
print("Press S to save a face, Q to quit.")

while saved < max_images:
    success, frame = camera.read()

    if not success:
        print("Could not read camera frame.")
        break

    faces = detect_faces(frame)

    for x, y, w, h in faces:
        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

    cv2.putText(
        frame, f"Saved: {saved}/{max_images}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    cv2.imshow("SmartSight AI - Dataset Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("s") and faces is not None:
        now = time.time()

        if now - last_saved < 0.25:
            continue

        if len(faces) == 0:
            print("No face detected. Try better lighting.")
            continue

        # Save the largest detected face.
        x, y, w, h = max(
            faces, key=lambda face: face[2] * face[3]
        )

        face = frame[y:y + h, x:x + w]

        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (64, 64))

        output = save_dir / f"face_{saved:04d}.jpg"

        if cv2.imwrite(str(output), face):
            saved += 1
            last_saved = now
            print(f"Saved: {output}")

camera.release()
cv2.destroyAllWindows()

print(f"Finished. Saved {saved} images in {save_dir}")
