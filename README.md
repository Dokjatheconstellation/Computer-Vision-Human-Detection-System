An edge-optimized, multi-purpose computer vision platform engineered for real-time physical security monitoring and demographic profiling. This intelligence suite combines two core functionalities: automated **Human Presence Tracking & Perimeter Security Alerts** alongside a deep-learning **Age & Gender Inference Pipeline** processing live video feeds.

---

##  System Workflow

The application handles high-throughput, low-latency processing of video frames through a unified sequential loop:

1. **Video Ingestion:** Captures live feeds from local webcams, IP security cameras (via RTSP streams), or pre-recorded video files.
2. **Frame Standardization:** Resizes and normalizes resolution inputs to optimize processing throughput without discarding critical visual features.
3. **Dual-Engine Inference:**
   * **Security Stream:** Passes the frame through an object detection network to map human silhouettes and extract bounding coordinates.
   * **Demographic Stream:** Utilizes a Haar Cascade algorithm to localize faces, which are immediately passed to pre-trained Caffe networks to predict age groups and gender categories.
4. **Post-Processing & Optimization:** Applies Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes and filters low-confidence predictions to eliminate environmental false positives.
5. **Annotation & Rendering:** Dynamic UI wrapper overlays active bounding boxes, threat boundary indicators, real-time person counts, and predictive profiling tags onto the live feed.

---

##  Tech Stack & Requirements

### Core Frameworks & Models
* **Language:** Python 3.12 (Supports 3.9+)
* **Computer Vision Library:** OpenCV (v4.12.0.88)
* **Inference Models:** YOLO / SSD MobileNet (Object Detection) & Caffe Framework Models (Demographic Classification)
* **Face Localizer:** Haar Cascade Classifier
* **Data & Utilities:** NumPy (v2.2.6), Imutils (v0.5.4)
* **Hardware Acceleration:** CUDA / cuDNN (Optional for GPU-backed real-time acceleration)
* **NOTE:** this repository does not contain the dataset for the Project

### Library Specifications
| Package | Version |
| :--- | :--- |
| `opencv-python` | `4.12.0.88` |
| `numpy` | `2.2.6` |
| `imutils` | `0.5.4` |

---

##  Key Features

* **Real-Time Tracking & Perimeter Alert Zones:** Implements efficient object tracking architectures to trace human presence. Includes configurable logic to draw custom "Restricted Zones" that trigger instant console alerts the moment a human boundary intersects the boundary.
* **Intelligent Demographic Profiling:** Automatically extracts and classifies subjects based on:
  * **Age Brackets:** `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`
  * **Gender:** `Male` / `Female`
* **Performance Optimization:** Leverages NMS and confidence-threshold filtering to minimize false detections caused by changing studio lighting, shifting shadows, or foliage movement.
* **Deep Learning Training Base:** Demographic classification engine leverages optimized models trained on the benchmark Adience dataset.

---

##  Project Structure

```text
surveillance-analytics-engine/
├── Detection_cam/
│   ├── test.py                   # Main real-time application script
│   ├── age_deploy.prototxt       # Age model architecture configuration
│   ├── age_net.caffemodel        # Age model weights (~441 MB)
│   ├── gender_deploy.prototxt    # Gender model architecture configuration
│   └── gender_net.caffemodel     # Gender model weights (~441 MB)
├── requirements.txt              # Project dependencies
└── README.md                     # System documentation
