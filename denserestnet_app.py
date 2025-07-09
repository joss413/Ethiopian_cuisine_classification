import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import gdown
import os
from dotenv import load_dotenv
import requests

HF_TOKEN = st.secrets["HF_TOKEN"]
REPO_ID = "Jossi18/Ethiopian_cusine_classification"
# Load flag image
flag_img = Image.open("images/flag.jpg")

# Convert flag image to base64 for inline HTML embedding (better alignment & no path issues)
buffered = BytesIO()
flag_img.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Load models from Google Drive with gdown
@st.cache_resource
def load_models():
    dense_path = "dense_final_model.h5"
    resnet_path = "resnet50v2_final4_model.h5"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Hugging Face direct URLs
    dense_url = f"https://huggingface.co/{REPO_ID}/resolve/main/{dense_path}"
    resnet_url = f"https://huggingface.co/{REPO_ID}/resolve/main/{resnet_path}"

    try:
        if not os.path.exists(dense_path):
            st.info("🔽 Downloading DenseNet model from Hugging Face...")
            with requests.get(dense_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(dense_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        if not os.path.exists(resnet_path):
            st.info("🔽 Downloading ResNet model from Hugging Face...")
            with requests.get(resnet_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                with open(resnet_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

        densenet = tf.keras.models.load_model(dense_path)
        resnet = tf.keras.models.load_model(resnet_path)

        return densenet, resnet

    except Exception as e:
        st.error(f"❌ Failed to load models from Hugging Face: {e}")
        st.stop()

def preprocess_image(img):
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def soft_voting(pred1, pred2):
    return (pred1 + pred2) / 2

# Class labels
CLASS_NAMES = [
    "beyaynetu", "chechebsa", "doro_wat", "firfir", "genfo",
    "kiki1", "kitfo", "shekla_tibs", "shiro_wat", "tihlo", "tire_siga"
]

# Fixed test accuracies for each model
DENSENET_ACC = 83.12
RESNET_ACC = 86.36
ENSEMBLE_ACC = 88.31

# Helper: get top-k predictions as list of (name, prob)
def get_top_k(pred, k=2):
    top_idx = pred.argsort()[-k:][::-1]
    return [(CLASS_NAMES[i], pred[i]) for i in top_idx]

# App UI
st.set_page_config(page_title="DensResNet – Ethiopian Food Classifier", layout="wide")

# Title with flag image side by side using HTML flexbox for perfect vertical alignment
st.markdown(
    f"""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="data:image/jpeg;base64,{img_str}" width="60" style="margin-right: 15px;"/>
        <h1 style="margin: 0;">DensResNet: Ethiopian Cuisine Classifier</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("🖼️ Upload Image")
    with st.expander("Upload your Ethiopian food image here", expanded=True):
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    st.markdown("---")

    with st.expander("🍽️ Models Used", expanded=True):
        st.markdown("""
        - **DenseNet121**  
        - **ResNet50V2**
        """)

    with st.expander("🔗 Ensemble Method", expanded=True):
        st.markdown("Soft Voting")

    with st.expander("📐 Input Size", expanded=True):
        st.markdown("512×512")

# Load models
with st.spinner("Loading models..."):
    densenet, resnet = load_models()

if uploaded_file:
    img = Image.open(uploaded_file)

    # Center and resize uploaded image
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(img, caption="Uploaded Image", width=400)
    st.markdown("</div>", unsafe_allow_html=True)

    processed = preprocess_image(img)

    # Predictions
    pred_dense = densenet.predict(processed)[0]
    pred_res = resnet.predict(processed)[0]
    pred_ensemble = soft_voting(pred_dense, pred_res)

    # Show predictions top-2 for each model
    st.subheader("🔍 DenseNet121 Prediction (Top-2)")
    for name, prob in get_top_k(pred_dense):
        st.write(f"**{name}**: {prob:.2%}")

    st.subheader("🔍 ResNet50V2 Prediction (Top-2)")
    for name, prob in get_top_k(pred_res):
        st.write(f"**{name}**: {prob:.2%}")

    st.subheader("🤝 Ensemble (Soft Voting) Prediction (Top-2)")
    for name, prob in get_top_k(pred_ensemble):
        st.write(f"**{name}**: {prob:.2%}")

    # Model performance table with top-1 confidence of ensemble
    st.subheader("📊 Model Performance and Top-1 Confidence")
    top1_idx = pred_ensemble.argmax()
    st.markdown(f"""
    | Model        | Test Accuracy | Top-1 Confidence |
    |--------------|---------------|------------------|
    | DenseNet121  | {DENSENET_ACC:.2f}%       | {pred_dense[top1_idx]*100:.2f}%          |
    | ResNet50V2   | {RESNET_ACC:.2f}%       | {pred_res[top1_idx]*100:.2f}%          |
    | **Ensemble** | **{ENSEMBLE_ACC:.2f}%**     | **{pred_ensemble[top1_idx]*100:.2f}%**        |
    """)

    # Plot bar charts for each model
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    axs[0].barh(CLASS_NAMES, pred_dense, color="steelblue")
    axs[0].invert_yaxis()
    axs[0].set_title("DenseNet121 Probabilities")
    axs[0].set_xlim(0, 1)
    axs[0].set_xlabel("Probability")

    axs[1].barh(CLASS_NAMES, pred_res, color="darkorange")
    axs[1].invert_yaxis()
    axs[1].set_title("ResNet50V2 Probabilities")
    axs[1].set_xlim(0, 1)
    axs[1].set_xlabel("Probability")

    axs[2].barh(CLASS_NAMES, pred_ensemble, color="teal")
    axs[2].invert_yaxis()
    axs[2].set_title("Ensemble Probabilities")
    axs[2].set_xlim(0, 1)
    axs[2].set_xlabel("Probability")

    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("Upload an image to start classification.")


# Show the 11 dish classes with small images
st.markdown("---")
st.subheader("🍲 Ethiopian Dishes Included in the Classifier")
st.markdown("The classifier is trained on the following 11 traditional Ethiopian dishes:")
cols = st.columns(3)
for i, name in enumerate(CLASS_NAMES):
    img_path = f"images/samples/{name}.jpg"
    with cols[i % 3]:
        st.image(img_path, width=100, caption=name.replace("_", " ").title())

st.markdown("---")
st.markdown("""
**About:** This app demonstrates DensResNet, an ensemble-based deep learning model for classifying 11 traditional Ethiopian dishes by combining the strengths of ResNet50V2 and DenseNet121 architectures.
\n Developed using TensorFlow and Streamlit.

**Author:** Yoseph Negash  
**Contact:** yosephn22@gmail.com  
**Year:** 2025
""")
