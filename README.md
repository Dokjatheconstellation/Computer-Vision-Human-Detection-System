# Computer Vision Human Detection System

A real-time physical security and surveillance application powered by edge-optimized computer vision. This project enhances traditional monitoring systems by automating human detection, reducing the need for constant manual oversight, and providing instantaneous alerting/tracking mechanisms for security applications and restricted area monitoring.

---

## 🏗️ System Workflow

The application operates as a high-throughput, low-latency loop processing video frames frame-by-frame:

1. **Video Stream Ingestion:** Captures live feeds from local webcams, IP security cameras (via RTSP streams), or pre-recorded video files.
2. **Frame Resize & Normalization:** Standardizes resolution inputs to optimize processing throughput for the underlying network without dropping critical image data.
3. **Model Inference:** Passes the frame through a deep learning object detection network to detect human presence and extract bounding box coordinates.
4. **Post-Processing & Non-Maximum Suppression (NMS):** Filters out overlapping redundant bounding boxes, ensuring each individual is counted and highlighted exactly once.
5. **Annotation & Rendering:** Draws bounding boxes, tracking markers, and real-time confidence metrics onto the live UI wrapper.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9+
* **Core Frameworks:** OpenCV (Open Source Computer Vision Library)
* **Inference Models:** YOLO (You Only Look Once) / SSD MobileNet *(Tip: Choose/Keep the one you used)*
* **Data & Math Utilities:** NumPy
* **Hardware Acceleration:** CUDA / cuDNN (Optional for GPU acceleration)

---

## 🌟 Key Features

* **Real-Time Detection & Tracking:** Utilizes highly efficient object detection architectures to maintain stable tracking on human subjects with negligible latency.
* **Smart Alerting Zones:** Engineered logic to draw customizable bounding zones (e.g., restricted red zones); triggers a console or log alert the moment a human boundary intersects with the region.
* **Performance Optimization:** Implemented Non-Maximum Suppression and confidence threshold filtering to drastically minimize false positives from moving background shadows, foliage, or changes in indoor lighting.
* **Dynamic Analytics:** Displays live analytics overlays directly onto the video feed, including the instantaneous total frame person count and model processing times.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9 or higher installed on your host machine.
* A connected webcam or a valid local video file path for testing.

### Installation

1. **Clone the repository:**
```bash
   git clone [https://github.com/YOUR_USERNAME/human-detection-system.git](https://github.com/YOUR_USERNAME/human-detection-system.git)
   cd human-detection-system
