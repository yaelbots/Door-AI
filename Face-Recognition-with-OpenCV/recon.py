import pandas as pd
import numpy as np
import cv2 as cv

# Load the mapping of ids to names
id_names = pd.read_csv('id-names.csv')
id_names = id_names[['id', 'name']]

# Initialize the face detector and the face recognizer
faceClassifier = cv.CascadeClassifier('/Users/yaelreyes/Downloads/Door AI new/Face-Recognition-with-OpenCV/Classifiers/haarface.xml')
lbph = cv.face.LBPHFaceRecognizer_create()
lbph.read('/Users/yaelreyes/Downloads/Door AI new/Face-Recognition-with-OpenCV/Classifiers/trainedmode.xml')

# Explicitly use DirectShow API to potentially solve capture issues
camera = cv.VideoCapture(0, cv.CAP_DSHOW)

# Set a confidence threshold. Adjust this based on experimentation.
confidence_threshold = 80  # Example threshold; adjust based on your observations

while cv.waitKey(1) & 0xFF != ord('q'):
    ret, img = camera.read()
    if not ret:
        print("Failed to grab frame")
        continue  # Skip this iteration and try grabbing the next frame

    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

    for x, y, w, h in faces:
        faceRegion = grey[y:y + h, x:x + w]
        faceRegion = cv.resize(faceRegion, (220, 220))

        label, confidence = lbph.predict(faceRegion)
        
        if confidence > confidence_threshold:
            name = "Unknown"
        elif label not in id_names['id'].values:
            name = "Unknown"
        else:
            name = id_names[id_names['id'] == label]['name'].item()

        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(img, name + f' ({int(confidence)})', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    cv.imshow('Recognize', img)

camera.release()
cv.destroyAllWindows()