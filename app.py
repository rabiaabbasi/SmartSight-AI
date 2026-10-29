
import streamlit as st
import subprocess
import sys
from pathlib import Path

st.set_page_config(
    page_title="SmartSight AI",
    page_icon="👁️",
    layout="wide"
)

st.title("SmartSight AI")
st.caption("ANN-based face recognition prototype")

st.subheader("Project status")

dataset_dir = Path("data/faces")
model_path = Path("models/face_ann.joblib")

people = (
    sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    if dataset_dir.exists()
    else []
)

col1, col2, col3 = st.columns(3)

col1.metric("Registered people", len(people))
col2.metric("ANN model", "Trained" if model_path.exists() else "Not trained")
col3.metric("Dataset folders", len(people))

st.subheader("Registered identities")

if people:
    for person in people:
        st.write(f"• {person}")
else:
    st.info("No dataset found. Run collect_faces.py first.")

st.subheader("Run live recognition")

st.write(
    "The live camera opens in a separate desktop window. "
    "Close it with Q when you finish."
)

if not model_path.exists():
    st.warning("Train the ANN model before starting recognition.")

if st.button("Start live recognition", disabled=not model_path.exists()):
    with st.spinner("Starting SmartSight AI..."):
        try:
            subprocess.Popen(
                [sys.executable, "run_recognition.py"],
                cwd=str(Path(__file__).parent)
            )
            st.success("Recognition process started.")
        except Exception as error:
            st.error(f"Could not start recognition: {error}")

st.caption(
    "Prototype only. Recognition can be incorrect; "
    "do not use it as the sole basis for security decisions."
)
