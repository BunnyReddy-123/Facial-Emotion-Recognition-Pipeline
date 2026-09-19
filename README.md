# End-to-End Facial Emotion Recognition Pipeline

An intelligent, multi-stage computer vision pipeline that automatically detects human faces using a custom fine-tuned YOLOv8 model and classifies their emotions using DeepFace.

## 🚀 Project Overview
Most standard object detection models are general-purpose. This project demonstrates an end-to-end custom computer vision workflow:
1. **Custom Object Detection:** Fine-tuned `YOLOv8n` on face images to specialize in high-accuracy face localization.
2. **Face Cropping & Pipeline Integration:** Programmatically crops bounding box coordinates from YOLO and passes them securely to DeepFace.
3. **Structured Reporting:** Automatically outputs annotated image visualizations alongside structured Pandas dataframes detailing face coordinates, dominant emotions, and confidence scores.

## 🧠 Model / Architecture
* **Stage 1 (Face Detection):** Utilizes `YOLOv8n` (YOLOv8 nano), a state-of-the-art anchor-free object detection model fine-tuned specifically to predict bounding boxes for human faces.
* **Stage 2 (Emotion Classification):** Integrates **DeepFace** (backed by convolutional neural networks) to analyze cropped face regions and classify emotional states (e.g., Happy, Sad, Angry, Neutral).
* **Pipeline Logic:** OpenCV handles multi-format image decoding, frame management, and rendering visual bounding boxes and labels onto output frames.

## 📊 Dataset & Training
* **Dataset Source:** Custom face dataset annotated for object detection bounding boxes.
* **Model Training:** Fine-tuned using `YOLOv8n` on Google Colab leveraging GPU acceleration.
* **Weights:** The resulting optimal model weights are saved as `best.pt` for direct inference integration.

## 🛠️ Technologies & Libraries Used
* **Python** (Core Programming Language)
* **Ultralytics YOLOv8** (Custom Object Detection & Face Localization)
* **DeepFace** (Facial Emotion Classification)
* **OpenCV & Pillow (PIL)** (Image Processing & Formatting)
* **Pandas & Matplotlib** (Data Reporting & Visualizations)
* **Google Colab** (Training Environment with GPU Acceleration)

## 📈 Results & Outputs
* **Visual Output:** Generates annotated images featuring crisp bounding boxes around detected faces with clear emotion labels and confidence scores rendered via OpenCV.
* **Structured Data:** Outputs clean Pandas DataFrames containing coordinates, dominant emotions, and probabilities for downstream analysis or reporting.

## 🚀 Future Improvements
* **Real-Time Video Stream Inference:** Expand the pipeline to process live webcam feeds or video files frame-by-frame instead of static images.
* **Web Application Deployment:** Wrap the pipeline into an interactive web interface using **Streamlit** or **Gradio** so users can upload images or use a webcam directly from a browser.
* **Model Quantization:** Optimize model weights using ONNX or TensorRT to decrease latency and edge-device inference times.

## ⚙️ How to Run
1. Open the **`FaceEmotion.ipynb`** notebook in Google Colab or your local Jupyter environment.
2. Ensure your `best.pt` file is uploaded in your working directory.
3. Run the cells sequentially to load the model, process images, and generate emotion reports and visualizations.
