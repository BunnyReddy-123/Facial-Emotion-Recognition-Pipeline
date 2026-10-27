import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from deepface import DeepFace

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊")
st.title("Facial Emotion Recognition using YOLOv8 and DeepFace")
st.write("Take a photo or upload an image. Images are processed in "
         "memory and are not stored.")

@st.cache_resource
def load_detector():
    return YOLO("best.pt")

detector = load_detector()

def analyze(pil_img):
    img = np.array(pil_img.convert("RGB"))
    results = detector(img, conf=0.4, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    for x1, y1, x2, y2 in boxes:
        face = img[max(y1, 0):y2, max(x1, 0):x2]
        if face.size == 0:
            continue
        try:
            out = DeepFace.analyze(face[:, :, ::-1], actions=["emotion"],
                                   detector_backend="skip",
                                   enforce_detection=False, silent=True)
            emotion = out[0]["dominant_emotion"]
        except Exception:
            emotion = "unknown"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return img, len(boxes)

tab1, tab2 = st.tabs(["Camera", "Upload image"])
with tab1:
    shot = st.camera_input("Take a photo")
with tab2:
    upload = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

source = shot or upload
if source is not None:
    result, n = analyze(Image.open(source))
    st.image(result, caption=f"Faces found: {n}")
