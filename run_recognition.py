
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.face_utils import detect_faces, prepare_face


# -------------------------------
# 1. Load Person Details
# -------------------------------

DETAILS_PATH = Path("data/person_details.json")

if DETAILS_PATH.exists():
    with open(DETAILS_PATH, "r", encoding="utf-8") as file:
        PERSON_DETAILS = json.load(file)
else:
    PERSON_DETAILS = {}
    print("Warning: person_details.json not found. Details will be unavailable.")


# -------------------------------
# 2. Draw Person Information
# -------------------------------

def draw_person_details(frame, name, x, y, w, h):
    details = PERSON_DETAILS.get(name, {})

    age = details.get("age", "Not provided")
    gender = details.get("gender", "Not provided")
    occupation = details.get("occupation", "Not provided")

    lines = [
        f"Name: {name}",
        f"Age: {age}",
        f"Gender: {gender}",
        f"Occupation: {occupation}"
    ]

    panel_width = 260
    line_height = 27
    panel_height = len(lines) * line_height + 15

    frame_height, frame_width = frame.shape[:2]

    # Place panel beside the face where possible
    if x + w + panel_width + 10 < frame_width:
        panel_x = x + w + 10
    else:
        panel_x = max(5, x - panel_width - 10)

    panel_x = min(panel_x, frame_width - panel_width - 5)
    panel_x = max(5, panel_x)

    panel_y = max(
        5,
        min(y, frame_height - panel_height - 5)
    )

    # Dark information panel
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (35, 35, 35),
        -1
    )

    # Green border
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (0, 255, 0),
        1
    )

    # Display information
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (panel_x + 10, panel_y + 25 + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 255, 100),
            1,
            cv2.LINE_AA
        )


# -------------------------------
# 3. Camera Configuration
# -------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--camera",
    default="0",
    help="Camera index or DroidCam stream URL"
)

args = parser.parse_args()


# -------------------------------
# 4. Load Trained ANN Model
# -------------------------------

model_path = Path("models/face_ann.joblib")

if not model_path.exists():
    raise FileNotFoundError(
        "Trained model not found. Run: python train_model.py"
    )

model = joblib.load(model_path)

source = int(args.camera) if args.camera.isdigit() else args.camera

camera = cv2.VideoCapture(source)

if not camera.isOpened():
    raise RuntimeError(
        "Camera did not open. Check your camera index or DroidCam URL."
    )


# -------------------------------
# 5. Recognition Settings
# -------------------------------

UNKNOWN_THRESHOLD = 0.75

print("SmartSight AI is running.")
print("Press Q to quit.")


# -------------------------------
# 6. Live Face Recognition
# -------------------------------

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        faces = detect_faces(frame)

        for x, y, w, h in faces:
            features = prepare_face(frame, x, y, w, h)

            if features is None:
                continue

            probabilities = model.predict_proba(
                features.reshape(1, -1)
            )[0]

            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])

            # Identify registered person or unknown person
            if confidence >= UNKNOWN_THRESHOLD:
                name = str(model.classes_[best_index])
            else:
                name = "Unknown Person"

            label = f"{name} ({confidence:.2f})"

            # Face bounding box
            color = (
                (0, 255, 0)
                if name != "Unknown Person"
                else (0, 0, 255)
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # Name above the face
            cv2.putText(
                frame,
                label,
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

            # Show personal details only for recognized people
            if name != "Unknown Person":
                draw_person_details(
                    frame, name, x, y, w, h
                )

        cv2.imshow(
            "SmartSight AI - Live Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
