# ============================================================
# NEUROSCAN AI
# Brain Tumor Detection and Classification Using CNN
# Developed by Gaurav Katre
# ============================================================

import os
import io
import json
import csv
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps
from datetime import datetime
import matplotlib.pyplot as plt


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NeuroScan AI | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. PREMIUM PROFESSIONAL WEBSITE THEME
# ============================================================

st.markdown("""
<style>

/* ==========================================
   MAIN BACKGROUND
========================================== */

.stApp {
    background:
        linear-gradient(
            rgba(235, 246, 240, 0.88),
            rgba(235, 246, 240, 0.93)
        ),
        url("https://images.pexels.com/photos/13176356/pexels-photo-13176356.jpeg?auto=compress&cs=tinysrgb&w=2000");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ==========================================
   MAIN CONTENT WIDTH
========================================== */

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ==========================================
   GLOBAL TEXT
========================================== */

.stApp {
    color: #244b39;
}

.stApp p,
.stApp label,
.stApp li,
.stApp span {
    color: #294b3b;
}

[data-testid="stMarkdownContainer"] {
    color: #294b3b !important;
}

/* ==========================================
   HEADINGS
========================================== */

.stApp h1 {
    color: #103d2b !important;
    font-size: 36px !important;
    font-weight: 800 !important;
}

.stApp h2 {
    color: #174c38 !important;
    font-size: 27px !important;
    font-weight: 750 !important;
}

.stApp h3 {
    color: #216044 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

.stApp h4 {
    color: #216044 !important;
    font-weight: 700 !important;
}

/* ==========================================
   CAPTIONS
========================================== */

[data-testid="stCaptionContainer"] p {
    color: #587263 !important;
    font-size: 12px !important;
}

/* ==========================================
   CONTENT CARDS
========================================== */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.97) !important;
    border: 1px solid #c9dfd0 !important;
    border-radius: 16px !important;
    padding: 16px !important;
    box-shadow: 0 5px 20px rgba(25, 75, 45, 0.07);
}

/* ==========================================
   PRIMARY BUTTONS
========================================== */

.stButton > button {
    background: linear-gradient(
        90deg,
        #176b45,
        #299765
    ) !important;

    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    height: 48px !important;
    font-weight: 700 !important;
    transition: 0.3s ease;
}

.stButton > button p {
    color: #ffffff !important;
}

.stButton > button:hover {
    background: #105536 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(23, 107, 69, 0.2);
}

/* ==========================================
   DOWNLOAD BUTTONS
========================================== */

.stDownloadButton > button {
    background: #e8f5ed !important;
    color: #174c38 !important;
    border: 1px solid #b9d9c4 !important;
    border-radius: 9px !important;
    font-weight: 700 !important;
}

.stDownloadButton > button p {
    color: #174c38 !important;
}

.stDownloadButton > button:hover {
    background: #d2eadb !important;
    border-color: #79b18d !important;
}

/* ==========================================
   FILE UPLOADER
========================================== */

[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1px solid #c8dfd0 !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

[data-testid="stFileUploader"] section {
    background: #f2f9f4 !important;
    border: 2px dashed #78aa89 !important;
    border-radius: 11px !important;
}

[data-testid="stFileUploader"] section * {
    color: #244b35 !important;
}

[data-testid="stFileUploader"] button {
    background: #24784e !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}

[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
}

/* ==========================================
   METRIC CARDS
========================================== */

[data-testid="stMetric"] {
    background: #f0f8f2 !important;
    border: 1px solid #cfe4d5 !important;
    border-radius: 13px !important;
    padding: 16px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #174c38 !important;
}

/* ==========================================
   ALERTS AND MESSAGES
========================================== */

[data-testid="stAlert"] {
    background: #edf7f0 !important;
    border: 1px solid #c5dfcd !important;
    border-radius: 11px !important;
}

[data-testid="stAlert"] p {
    color: #244b35 !important;
}

/* ==========================================
   DATAFRAME / TABLE
========================================== */

[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border: 1px solid #cfe3d5 !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ==========================================
   TABS
========================================== */

.stTabs [data-baseweb="tab"] {
    color: #24543c !important;
    font-weight: 700 !important;
}

.stTabs [aria-selected="true"] {
    color: #176b45 !important;
    border-bottom-color: #299765 !important;
}

/* ==========================================
   INPUT FIELDS
========================================== */

.stTextInput input,
.stNumberInput input,
.stSelectbox [data-baseweb="select"],
.stTextArea textarea {
    background: #ffffff !important;
    color: #174c38 !important;
    border: 1px solid #c9dfd0 !important;
    border-radius: 9px !important;
}

/* ==========================================
   PROGRESS BAR
========================================== */

.stProgress > div > div > div {
    background-color: #269663 !important;
}

/* ==========================================
   SIDEBAR
========================================== */

[data-testid="stSidebar"] {
    background: #f5faf6 !important;
    border-right: 1px solid #d6e7db;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #244b35 !important;
}

/* ==========================================
   DIVIDERS
========================================== */

hr {
    border-color: #d2e5d8 !important;
}

/* ==========================================
   FOOTER
========================================== */

footer {
    visibility: hidden;
}

/* ==========================================
   REMOVE UNNECESSARY TOP SPACE
========================================== */

header[data-testid="stHeader"] {
    background: transparent;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_PATHS = [
    "brain_tumor_model.keras",
    os.path.join("models", "brain_tumor_model.keras")
]

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMAGE_SIZE = (224, 224)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_trained_model():

    for model_path in MODEL_PATHS:

        if os.path.exists(model_path):

            loaded_model = tf.keras.models.load_model(
                model_path,
                compile=False
            )

            return loaded_model, model_path

    return None, None


model, loaded_model_path = load_trained_model()


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    image = ImageOps.exif_transpose(image)

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# 6. SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 NeuroScan AI")

    st.caption("BRAIN MRI CLASSIFICATION PLATFORM")

    st.divider()

    st.subheader("Navigation")

    st.write("🏠 Dashboard")
    st.write("🖼️ MRI Image Analysis")
    st.write("📊 Model Performance")
    st.write("📁 Prediction History")
    st.write("ℹ️ About Project")

    st.divider()

    st.subheader("Model Status")

    if model is not None:

        st.success("CNN Model Loaded")

    else:

        st.error("Model Not Found")

    st.caption("Input Resolution: 224 × 224")

    st.caption("Output Categories: 4")


# ============================================================
# 8. HEADER
# ============================================================

st.title("🧠 NeuroScan AI")

st.caption(
    "INTELLIGENT BRAIN MRI IMAGE CLASSIFICATION PLATFORM"
)

st.divider()

st.subheader("Artificial Intelligence | Deep Learning")

st.header("Brain MRI Classification")

st.write(
    """
    An educational brain MRI image classification platform
    powered by Convolutional Neural Networks.

    Upload an MRI image to explore the model's predicted
    category and class-wise prediction scores.
    """
)

st.success("● CNN Powered Image Classification Platform")


# ============================================================
# 9. PROJECT STATISTICS
# ============================================================

st.header("📊 Project Statistics")

st.write(
    "An overview of the dataset and deep learning architecture."
)

stat1, stat2, stat3, stat4 = st.columns(4)

with stat1:

    with st.container(border=True):

        st.metric(
            "Total MRI Images",
            "7,200"
        )

        st.caption("5,600 Training + 1,600 Testing")

with stat2:

    with st.container(border=True):

        st.metric(
            "Classification Classes",
            "4"
        )

        st.caption("Four MRI image categories")

with stat3:

    with st.container(border=True):

        st.metric(
            "CNN Architecture",
            "3 Layers"
        )

        st.caption("32, 64 and 128 filters")

with stat4:

    with st.container(border=True):

        st.metric(
            "Input Resolution",
            "224 × 224"
        )

        st.caption("RGB Image Processing")


# ============================================================
# 10. PLATFORM OVERVIEW
# ============================================================

st.header("🚀 Platform Overview")

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:

    with st.container(border=True):

        st.subheader("🧠 CNN Model")

        st.write(
            "Deep learning architecture for brain MRI classification."
        )

with col2:

    with st.container(border=True):

        st.subheader("🖼️ Image Processing")

        st.write(
            "Image resizing, RGB conversion and normalization."
        )

with col3:

    with st.container(border=True):

        st.subheader("⚡ AI Prediction")

        st.write(
            "Classification across four image categories."
        )

with col4:

    with st.container(border=True):

        st.subheader("📊 Model Scores")

        st.write(
            "Visual representation of class-wise prediction scores."
        )


# ============================================================
# 11. MRI IMAGE ANALYSIS
# ============================================================

st.divider()

st.header("🔬 MRI Image Analysis")

upload_col, info_col = st.columns(
    [1.25, 0.75],
    gap="large"
)

uploaded_file = None
image = None

with upload_col:

    with st.container(border=True):

        st.subheader("📤 Upload Brain MRI Image")

        uploaded_file = st.file_uploader(
            "Choose an MRI image",
            type=["jpg", "jpeg", "png"],
            help="Upload a JPG, JPEG or PNG image."
        )

        if uploaded_file is not None:

            try:

                image = Image.open(
                    io.BytesIO(uploaded_file.getvalue())
                )

                image = ImageOps.exif_transpose(image)

                st.image(
                    image,
                    caption="Uploaded MRI Image",
                    use_container_width=True
                )

                st.caption(
                    f"Image dimensions: {image.width} × {image.height}"
                )

            except Exception as e:

                st.error(f"Unable to open image: {e}")

        else:

            st.info(
                "Upload a brain MRI image to see its preview."
            )


with info_col:

    with st.container(border=True):

        st.subheader("⚙️ Model Information")

        st.write("**Architecture:** CNN")

        st.write("**Convolutional Layers:** 3")

        st.write("**Filters:** 32, 64, 128")

        st.write("**Input Size:** 224 × 224")

        st.write("**Channels:** RGB")

        st.write("**Output Classes:** 4")

        st.write("**Normalization:** 0–1")

        st.divider()

        st.subheader("Classification Categories")

        for name in CLASS_NAMES:

            st.write(f"🟢 {name}")


# ============================================================
# 12. PREDICTION
# ============================================================

if model is None:

    st.error(
        "Trained model not found. Place "
        "'brain_tumor_model.keras' in the project folder "
        "or inside the models folder."
    )

elif uploaded_file is not None and image is not None:

    if st.button(
        "🚀 ANALYZE MRI IMAGE",
        use_container_width=True
    ):

        try:

            with st.spinner("Analyzing MRI image..."):

                processed_image = preprocess_image(image)

                predictions = model.predict(
                    processed_image,
                    verbose=0
                )

                if (
                    predictions.ndim != 2
                    or predictions.shape[1] != len(CLASS_NAMES)
                ):

                    st.error(
                        "Model output does not match the four classes."
                    )

                    st.stop()

                scores = predictions[0]

                predicted_index = int(
                    np.argmax(scores)
                )

                predicted_class = CLASS_NAMES[predicted_index]

                confidence = float(
                    scores[predicted_index]
                ) * 100

                result = {
                    "filename": uploaded_file.name,
                    "predicted_class": predicted_class,
                    "confidence": confidence,
                    "scores": {
                        CLASS_NAMES[i]: float(scores[i]) * 100
                        for i in range(len(CLASS_NAMES))
                    },
                    "time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }

                st.session_state.latest_prediction = result

                st.session_state.prediction_history.append(
                    result
                )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# ============================================================
# 13. DISPLAY PREDICTION RESULTS
# ============================================================

result = st.session_state.latest_prediction

if result is not None:

    st.divider()

    st.header("🎯 AI Classification Results")

    result_col, score_col = st.columns(
        [1, 1],
        gap="large"
    )

    with result_col:

        with st.container(border=True):

            st.subheader("Predicted Category")

            st.title(
                result["predicted_class"]
            )

            st.metric(
                "Prediction Score",
                f'{result["confidence"]:.2f}%'
            )

            st.caption(
                "This score represents the model's output, "
                "not a medically validated probability."
            )

    with score_col:

        with st.container(border=True):

            st.subheader("📊 Class-wise Prediction Scores")

            score_data = pd.DataFrame({
                "Category": list(result["scores"].keys()),
                "Score (%)": list(result["scores"].values())
            })

            st.bar_chart(
                score_data.set_index("Category"),
                horizontal=True
            )

    # Individual class scores

    st.subheader("Detailed Prediction Scores")

    score_cols = st.columns(4)

    for i, class_name in enumerate(CLASS_NAMES):

        with score_cols[i]:

            st.metric(
                class_name,
                f'{result["scores"][class_name]:.2f}%'
            )

    # DOWNLOAD REPORT

    report_text = f"""
NEUROSCAN AI - MRI CLASSIFICATION REPORT

Filename: {result["filename"]}

Prediction Date: {result["time"]}

Predicted Category: {result["predicted_class"]}

Prediction Score: {result["confidence"]:.2f}%

CLASS-WISE SCORES
"""

    for name, score in result["scores"].items():

        report_text += f"\n{name}: {score:.2f}%"

    report_text += """

IMPORTANT DISCLAIMER:

This is an educational deep learning project.
It is not a clinically validated diagnostic system.
Do not use these predictions for medical diagnosis.
Consult a qualified medical professional.
"""

    st.download_button(
        label="📥 Download Prediction Report",
        data=report_text,
        file_name="NeuroScan_AI_Report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.success("Image classification completed.")

    st.warning(
        "Educational project only. This model is not a "
        "clinically validated diagnostic system. Its predictions "
        "must not be used to diagnose or rule out a brain tumor. "
        "Consult a qualified medical professional."
    )


# ============================================================
# 15. PREDICTION HISTORY
# ============================================================

st.divider()

st.header("📁 Prediction History")

history = st.session_state.prediction_history

if history:

    history_df = pd.DataFrame([
        {
            "Image": item["filename"],
            "Prediction": item["predicted_class"],
            "Score (%)": round(item["confidence"], 2),
            "Date": item["time"]
        }
        for item in history
    ])

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False)

    st.download_button(
        "📥 Download Prediction History",
        data=csv_data,
        file_name="NeuroScan_Prediction_History.csv",
        mime="text/csv"
    )

    if st.button("Clear Prediction History"):

        st.session_state.prediction_history = []

        st.session_state.latest_prediction = None

        st.rerun()

else:

    st.info(
        "Your prediction history will appear here "
        "after you analyze an MRI image."
    )


# ============================================================
# 16. ABOUT THE PROJECT
# ============================================================

st.divider()

st.header("ℹ️ About NeuroScan AI")

about_col1, about_col2 = st.columns(2)

with about_col1:

    with st.container(border=True):

        st.subheader("Project Overview")

        st.write(
            """
            NeuroScan AI is an educational deep learning project
            that uses a Convolutional Neural Network to classify
            brain MRI images into four categories.

            The application demonstrates image preprocessing,
            CNN-based classification and model prediction
            visualization.
            """
        )

with about_col2:

    with st.container(border=True):

        st.subheader("Technologies Used")

        st.write("🐍 Python")

        st.write("🧠 TensorFlow / Keras")

        st.write("📊 NumPy and Pandas")

        st.write("🖼️ Pillow")

        st.write("🌐 Streamlit")

        st.write("📈 Matplotlib")


# ============================================================
# FOOTER
# ============================================================


st.divider()

st.markdown(
    """
    <div style="
        background-color: white;
        border: 2px solid green;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
    ">
        <h2 style="color: #174c38;">🧠 NeuroScan AI</h2>
        <p style="color: #294b39;">Brain Tumor Classification Using CNN</p>
        <p style="color: #174c38;"><b>Developed by Gaurav Katre</b></p>
        <p style="color: #587263;">Machine Learning & Deep Learning Project</p>
        <p style="color: #587263;">Educational and Research Purposes Only</p>
    </div>
    """,
    unsafe_allow_html=True
)