import cv2 as cv

classifier_path = 'C:\\Users\\oscar\\OneDrive\\Desktop\\Door AI new\\Face-Recognition-with-OpenCV\\Classifiers\\haarface.xml'
faceClassifier = cv.CascadeClassifier(classifier_path)

# Check if the classifier was loaded correctly
if faceClassifier.empty():
    print("Error: Couldn't load classifier.")
else:
    print("Classifier loaded successfully.")
