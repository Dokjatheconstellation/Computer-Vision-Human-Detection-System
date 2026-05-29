import cv2
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

age_proto = os.path.join(base_dir, "age_deploy.prototxt")
age_model = os.path.join(base_dir, "age_net.caffemodel")
gender_proto = os.path.join(base_dir, "gender_deploy.prototxt")
gender_model = os.path.join(base_dir, "gender_net.caffemodel")

age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

# these are the labels the model was trained on
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# mean values used during training (just copy-pasted from the tutorial)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# using haar cascade to actually detect faces properly
# (my first version just cropped the top of the body box which didnt work well)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_age_gender(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces):
        # draw a box around each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # crop just the face region for the model
        face_crop = frame[y:y + h, x:x + w]

        if face_crop.size == 0:
            continue  # skip if the crop is empty for some reason

        # prepare the image the way the model expects it
        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # put the label above the face box
        label = f"Person {i + 1}: {gender}, Age {age}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


def start_camera():
    # open the default webcam
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        print("error: couldnt open webcam")
        return

    print("press q to quit")

    while True:
        ret, frame = video.read()

        if not ret:
            print("couldnt read frame, exiting")
            break

        # resize to make it run faster
        frame = cv2.resize(frame, (640, 480))

        # run detection on the frame
        output = detect_age_gender(frame)

        cv2.imshow("Age and Gender Detector", output)

        # quit if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()
