# SmartSight AI

**An AI-powered face recognition and personal information display system using Artificial Neural Networks (ANN).**

SmartSight AI is a computer vision project that uses a live camera feed to detect faces, recognize registered individuals, and display their associated information in real time. The project aims to prototype a smart visual assistant that presents useful information about recognized people.

## Features

- **Real-Time Face Detection:** Detects faces from a live camera feed using OpenCV.
- **ANN-Based Face Recognition:** Uses a Multi-Layer Perceptron (MLP) neural network to classify registered faces.
- **Personal Information Display:** Displays saved details such as name, age, and occupation for recognized individuals.
- **Unknown Person Detection:** Labels faces as `Unknown Person` when the model's confidence falls below a configurable threshold.
- **Live Face Bounding Boxes:** Highlights detected faces with colored boxes and recognition labels.
- **Camera Support:** Supports a local camera and compatible DroidCam stream URLs.
- **Private Dataset Handling:** Keeps collected face images and trained model files out of the Git repository.

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Joblib
- Streamlit (dashboard, if enabled in the project)

## How It Works

1. **Face Detection:** OpenCV detects faces in the camera frame.
2. **Preprocessing:** Detected face images are converted to grayscale, resized to 64 × 64 pixels, normalized, and flattened into feature vectors.
3. **Model Prediction:** A trained ANN model predicts the identity of each detected face.
4. **Identity Matching:** The prediction confidence is compared with a configurable threshold.
5. **Information Retrieval:** For recognized people, the system retrieves their saved details from `data/person_details.json`.
6. **Visual Display:** The person's name and available details are displayed beside the face. Unrecognized faces are labelled `Unknown Person`.

**Important:** The ANN model recognizes identities from its training data. Age, gender information, and occupation are not automatically discovered by the model; these fields are retrieved from the saved details file.

## Project Structure

```text
SmartSight-AI/
├── src/
│   └── face_utils.py
├── data/
│   ├── faces/                 # Local face dataset (not committed)
│   └── person_details.json    # Local person information
├── models/
│   └── face_ann.joblib        # Trained model (not committed)
├── collect_faces.py
├── train_model.py
├── run_recognition.py
├── app.py                     # Dashboard, if configured
├── requirements.txt
├── .gitignore
└── README.md
```

The exact files in your local repository may vary as the project develops.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SmartSight-AI.git
cd SmartSight-AI
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Use a compatible Python and OpenCV version. If your project uses OpenCV 4.x, install the version specified by your working environment or requirements file.

## Usage

### Step 1: Collect face images

Register consenting participants and collect multiple face samples for each person:

```bash
python collect_faces.py
```

Follow the instructions displayed by the collection script. The dataset is stored locally in `data/faces/`.

### Step 2: Train the ANN model

```bash
python train_model.py
```

The training script processes the collected images, trains the neural network, evaluates its performance, and saves the model under `models/`.

### Step 3: Add person details

Create `data/person_details.json` and enter information for registered participants. The JSON keys must exactly match the corresponding folder names and the labels learned by the model.

Example:

```json
{
  "Ali": {
    "age": 22,
    "gender": "Not provided",
    "occupation": "Student"
  }
}
```

Use consented, accurate information. These values are illustrative examples.

### Step 4: Start live recognition

```bash
python run_recognition.py
```

To use another camera index:

```bash
python run_recognition.py --camera 1
```

For a compatible DroidCam stream, provide its stream URL:

```bash
python run_recognition.py --camera "YOUR_CAMERA_STREAM_URL"
```

Press **Q** to close the live recognition window.

### Step 5: Launch the dashboard (optional)

If `app.py` is configured for your current project:

```bash
streamlit run app.py
```

## Privacy and Data Protection

SmartSight AI is an educational prototype. Face images and personal information can be sensitive.

- Obtain participants' informed consent before collecting or processing their face images.
- Keep `data/faces/`, `models/`, and private personal information out of public repositories.
- Do not commit real names, ages, or other identifying information without permission.
- Review `.gitignore` before pushing code to GitHub.
- Remember that `.gitignore` does not remove files that have already been committed.
- Use the system only for appropriate, consent-based purposes.

## Limitations

- Recognition accuracy depends on dataset quality, lighting, camera position, and variations in facial appearance.
- The confidence threshold is configurable and is not a guarantee of identity.
- The model may incorrectly recognize someone or classify a known person as unknown.
- Displayed personal details depend on the local JSON file being correctly configured.
- This prototype is not intended for security-critical identification or decisions about people.

## Future Improvements

- Develop a more robust face embedding and matching pipeline.
- Add a database-backed person management system.
- Improve the dashboard with registration and profile management.
- Add recognition history with appropriate privacy controls.
- Explore an interface inspired by smart glasses and heads-up displays.
- Evaluate model performance on a separate test dataset.

## Project Purpose

SmartSight AI was developed as an educational computer vision and artificial neural network project to explore real-time face detection, identity classification, and contextual information display.

## License

Choose a license before making the repository public. If you want others to reuse your code, consider adding an appropriate open-source license, such as the MIT License.
