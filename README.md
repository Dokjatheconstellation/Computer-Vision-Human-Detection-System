Age & Gender Detection Camera
A real-time webcam app that detects faces and predicts the age and gender of each person using OpenCV and pre-trained Caffe models.

How It Works

Opens your webcam
Detects faces using Haar Cascade (built into OpenCV)
Passes each face to two pre-trained models — one for age, one for gender
Draws a box around each face and labels it with the result


Project Structure
Detection_cam/
├── test.py                   # main script
├── age_deploy.prototxt       # age model architecture
├── age_net.caffemodel        # age model weights (~441 MB)
├── gender_deploy.prototxt    # gender model architecture
├── gender_net.caffemodel     # gender model weights (~441 MB)
├── requirements.txt
└── README.md

Setup
1. Clone or download the project
2. Create and activate a virtual environment
powershellpython -m venv .venv
.venv\Scripts\activate
3. Install dependencies
powershellpip install -r requirements.txt
4. Download the model files
The .prototxt files (model architecture):

age_deploy.prototxt
gender_deploy.prototxt

The .caffemodel files (model weights) — download via terminal:
powershellcurl -L -o age_net.caffemodel "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/age_net.caffemodel"
curl -L -o gender_net.caffemodel "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/gender_net.caffemodel"

Make sure all 4 files are in the same folder as test.py


Run
powershellpython test.py
Press Q to quit the camera window.

Requirements
PackageVersionopencv-python4.12.0.88numpy2.2.6imutils0.5.4
Python3.12

Age & Gender Labels
Age groups: (0-2) (4-6) (8-12) (15-20) (25-32) (38-43) (48-53) (60-100)
Gender: Male / Female

Notes

Works best with good lighting and a front-facing camera
Accuracy decreases if the face is at an angle or partially covered
Models were trained on the Adience dataset
