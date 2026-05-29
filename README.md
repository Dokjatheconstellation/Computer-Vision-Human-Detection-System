# Advanced Video Analytics Suite: Urdu Deepfake Detection & Real-Time Demographic Profiling

A multi-purpose computer vision and deep learning platform designed for advanced video analysis. This suite contains two core intelligent subsystems: an ongoing, research-driven **Urdu Media Deepfake Detector** tailored to find synthetic manipulations in regional broadcasting, and a real-time **Edge-Based Age & Gender Profiling System** optimized for live camera streams.

---

## 🏗️ Core Subsystems & Frameworks

### 1. Automated Deepfake Detection in Urdu Media
This module targets digital misinformation in regional broadcasting where traditional, western-centric datasets (like FaceForensics++ or DFDC) fail due to distinct localized factors.
* **The Challenges Solved:** Accounts for low-resolution regional feeds, broadcast stream-compression artifacts, unique Urdu lip-sync/phonetic traits, and cultural variations in studio lighting or attire.
* **Pipeline:** Video Ingestion (OpenCV) ➡️ Facial Detection & Alignment (MTCNN) ➡️ Temporal Sequence Chunking ➡️ Dual-Stream Deep Learning Inference Engine (Spatial + Temporal Analysis) ➡️ Authentication Confidence Score.

### 2. Live Age & Gender Detection Camera
A low-latency deployment module that accesses local video feeds to locate human faces and instantaneously classify demographic metrics.
* **Pipeline:** Live Feed Capture ➡️ Haar Cascade Face Localization ➡️ Dual-Stream Caffe Model Inference ➡️ Dynamic Bounding Box UI Rendering with localized age bracket and gender labels.

---

## 🛠️ Tech Stack & Model Architectures

* **Language:** Python 3.10+
* **Deep Learning Frameworks:** PyTorch, TensorFlow/Keras, Caffe Inference Engine
* **Computer Vision Tools:** OpenCV (v4.12.0+), MTCNN, FaceNet, Imutils
* **Data Engineering:** NumPy, Pandas, Scikit-learn
* **Performance Enhancements:** FFmpeg (for rapid media demuxing)
* **Pre-trained Core Weights:** Caffe Models trained on the Adience Dataset (Age/Gender Classification)

---

## 📂 Project Structure

```text
advanced-video-analytics/
├── Deepfake_detection/       # Core files for Urdu Media Analysis Pipeline
│   ├── preprocess.py         # Frame extraction and MTCNN tracking
│   └── inference.py          # Dual-stream deepfake classifier engine
├── Detection_cam/            # Live demographic detection module
│   ├── test.py               # Main real-time camera script
│   ├── age_deploy.prototxt   # Age model layer structure
│   ├── age_net.caffemodel    # Age model weights (~441 MB)
│   ├── gender_deploy.prototxt# Gender model layer structure
│   └── gender_net.caffemodel # Gender model weights (~441 MB)
├── requirements.txt          # Shared dependencies
└── README.md                 # Complete documentation
