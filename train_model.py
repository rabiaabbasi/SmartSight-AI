
from pathlib import Path

import cv2
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.face_utils import prepare_face

DATA_DIR = Path("data/faces")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X = []
y = []

if not DATA_DIR.exists():
    raise FileNotFoundError(
        "Dataset folder not found. Collect faces first."
    )

for person_dir in sorted(DATA_DIR.iterdir()):
    if not person_dir.is_dir():
        continue

    person_name = person_dir.name
    count = 0

    for image_path in person_dir.glob("*"):
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        # Collection script saved cropped face images.
        face = cv2.resize(image, (64, 64))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        features = gray.astype("float32").flatten() / 255.0

        X.append(features)
        y.append(person_name)
        count += 1

    print(f"{person_name}: {count} images")

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

if len(X) == 0:
    raise ValueError("No usable images were found.")

classes, counts = np.unique(y, return_counts=True)

if len(classes) < 2:
    raise ValueError(
        "Collect images for at least two different people."
    )

if counts.min() < 4:
    raise ValueError(
        "Each person needs at least 4 images for this split."
    )

# Keep the split stratified so each person appears in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Feed-forward Artificial Neural Network.
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    random_state=42,
    early_stopping=True
)

print("\nTraining ANN...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy:.3f}")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions, labels=model.classes_))
print("Class order:", list(model.classes_))

joblib.dump(model, MODEL_DIR / "face_ann.joblib")
joblib.dump(
    list(model.classes_),
    MODEL_DIR / "labels.joblib"
)

print("\nModel saved in models/face_ann.joblib")
