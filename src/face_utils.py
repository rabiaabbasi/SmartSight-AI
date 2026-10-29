
import cv2

# OpenCV's built-in face detector
CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    raise RuntimeError("Could not load the face detector.")


def detect_faces(frame):
    """Return face rectangles detected in a camera frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    return faces


def prepare_face(frame, x, y, w, h):
    """Convert a detected face into a standard ANN input."""
    face = frame[y:y + h, x:x + w]

    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (64, 64))

    # Normalize pixel values to the range 0–1.
    face = face.astype("float32") / 255.0

    # Flatten 64 × 64 pixels into 4096 input features.
    return face.flatten()
