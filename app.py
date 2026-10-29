
import json
import threading
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template

from src.face_utils import detect_faces, prepare_face


app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_ann.joblib"
DETAILS_PATH = BASE_DIR / "data" / "person_details.json"

# Configuration
UNKNOWN_THRESHOLD = 0.75
CAMERA_INDEX = 0

# Load the trained ANN model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ANN model not found at: {MODEL_PATH}\n"
        "Please collect face images and run: python train_model.py"
    )

model = joblib.load(MODEL_PATH)

# Load saved person details
if DETAILS_PATH.exists():
    with open(DETAILS_PATH, "r", encoding="utf-8") as file:
        PERSON_DETAILS = json.load(file)
else:
    PERSON_DETAILS = {}

# Shared camera and recognition state
camera = None
camera_lock = threading.Lock()
latest_frame = None
latest_results = []
running = False


def open_camera():
    """Open the laptop camera if it is not already open."""
    global camera

    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(CAMERA_INDEX)

        if not camera.isOpened():
            if camera is not None:
                camera.release()
                camera = None

            raise RuntimeError(
                "Camera could not open. Check camera permissions "
                "and close other apps using the camera."
            )


def process_frame(frame):
    """Detect faces, recognize identities, and draw saved details."""
    results = []
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

        if confidence >= UNKNOWN_THRESHOLD:
            name = str(model.classes_[best_index])
        else:
            name = "Unknown Person"

        color = (
            (0, 200, 0)
            if name != "Unknown Person"
            else (0, 0, 255)
        )

        # Draw the face rectangle and recognition label
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{name} ({confidence:.2f})",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

        details = PERSON_DETAILS.get(name, {})

        # Display details only for recognized identities
        if name != "Unknown Person":
            lines = [
                f"Age: {details.get('age', 'Not provided')}",
                f"Gender: {details.get('gender', 'Not provided')}",
                f"Occupation: {details.get('occupation', 'Not provided')}"
            ]

            panel_width = 250
            line_height = 24
            panel_height = len(lines) * line_height + 15

            frame_h, frame_w = frame.shape[:2]

            # Place the details panel beside the face
            if x + w + panel_width + 10 < frame_w:
                px = x + w + 10
            else:
                px = x - panel_width - 10

            # Keep the panel within the frame
            px = max(5, min(px, max(5, frame_w - panel_width - 5)))
            py = max(5, min(y, max(5, frame_h - panel_height - 5)))

            cv2.rectangle(
                frame,
                (px, py),
                (
                    min(px + panel_width, frame_w - 1),
                    min(py + panel_height, frame_h - 1)
                ),
                (35, 35, 35),
                -1
            )

            for i, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (px + 8, py + 23 + i * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (100, 255, 100),
                    1,
                    cv2.LINE_AA
                )

        results.append({
            "name": name,
            "confidence": round(confidence, 3),
            "details": details if name != "Unknown Person" else {}
        })

    return frame, results


def generate_frames():
    """Capture camera frames and stream them to the browser."""
    global latest_frame, latest_results, running, camera

    try:
        open_camera()
        running = True

        while running:
            with camera_lock:
                if camera is None or not camera.isOpened():
                    break

                success, frame = camera.read()

            if not success:
                time.sleep(0.1)
                continue

            frame, results = process_frame(frame)

            success, buffer = cv2.imencode(".jpg", frame)

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            with camera_lock:
                latest_results = results
                latest_frame = frame_bytes

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

    except Exception as exc:
        print(f"Camera error: {exc}")

    finally:
        running = False

        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None


@app.route("/")
def index():
    """Display the main dashboard."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Provide the live camera stream."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/results")
def api_results():
    """Return the latest face recognition results."""
    with camera_lock:
        results = list(latest_results)
        camera_running = running

    return jsonify({
        "camera_running": camera_running,
        "people": results
    })


@app.route("/health")
def health():
    """Basic health check for deployment."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Local development server
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
