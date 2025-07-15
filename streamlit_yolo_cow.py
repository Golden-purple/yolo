import streamlit as st
import gdown
import os
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

st.set_page_config(page_title = "Different Yolo Models", layout = "centered")
st.title("Yolo Models")

MODEL_URLS = {
    "Yolo 8 Nano + Trained manually with limited cow images": "https://drive.google.com/uc?id=1MUadIAXW63vh_w1TPSWjUFVT7sAShxc9",
    "Yolo 11 Nano": "https://drive.google.com/uc?id=1dg-_9_MSHvyN2FgOY4vovRXN9NoaZNCY",
    "Yolo 11 Extra": "https://drive.google.com/uc?id=1NoZTxteebhEPG-gPNYfa-XGDtNA8mw1R",
    "Yolo 8 Nano": "https://drive.google.com/uc?id=1BknHGF2-2NsOzxcw7BIF7MRHhjETNNFU",
    "Yolo 8 Extra": "https://drive.google.com/uc?id=1jb8ZrmSBWz_mRyXu5PPKCSayCZkg3_HK"
}

model_name = st.sidebar.selectbox("Choose a YOLO model", list(MODEL_URLS.keys()))
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

@st.cache_resource
def download_and_load_model(url, name):
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{name}.pt"
    if not os.path.exists(model_path):
        st.info(f"Downloading {name} model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    return YOLO(model_path)

model = download_and_load_model(MODEL_URLS[model_name], model_name)

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        file_path = temp_file.name

    is_video = uploaded_file.type.startswith("video")

    if not is_video:
        image = Image.open(file_path).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        results = model(image, conf=confidence_threshold)
        result_img = results[0].plot()
        st.image(result_img, caption="Detected", use_column_width=True)
    else:
        st.video(file_path)
        cap = cv2.VideoCapture(file_path)

        st.info("Processing video (showing 10 annotated frames)...")
        frame_count = 0
        while cap.isOpened() and frame_count < 10:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=confidence_threshold)
            annotated = results[0].plot()
            st.image(annotated, caption=f"Frame {frame_count + 1}", use_column_width=True)
            frame_count += 1
        cap.release()

