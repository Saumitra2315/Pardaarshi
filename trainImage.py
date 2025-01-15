import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread

# -------------- Get Images and Labels ------------------------

def getImagesAndLabels(path):
    # Get the paths of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []  # List to hold face images
    labels = [] # List to hold labels (names)

    for imagePath in imagePaths:
        # Load the image and convert it to grayscale
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')  # Convert to numpy array

        # Extract the name from the filename
        try:
            label = os.path.split(imagePath)[-1].split(".")[0]  # Extract name
            faces.append(imageNp)
            labels.append(label)
        except IndexError:
            print(f"Skipping file: {imagePath}, invalid filename format.")
    return faces, labels

# ----------- Train Images Function ---------------

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create LBPH recognizer
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    # Load images and labels
    faces, labels = getImagesAndLabels("TrainingImage")
    
    if len(faces) == 0:
        print("No images found in 'TrainingImage'. Please add images and try again.")
        return

    # Convert labels (names) to numeric IDs for training
    unique_labels = list(set(labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    ids = [label_to_id[label] for label in labels]

    # Train the recognizer with the images and labels
    recognizer.train(faces, np.array(ids))
    recognizer.save("TrainingImageLabel" + os.sep + "Trainner.yml")
    print(f"{len(faces)} Images Trained Successfully.")

    # Print label-to-ID mapping for reference
    print("Label to ID mapping:")
    for label, id in label_to_id.items():
        print(f"{label}: {id}")

    # Optional visual counter
    Thread(target=counter_img("TrainingImage")).start()

# Optional: Adds a visual counter for images trained
def counter_img(path):
    imgCounter = 1
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    for _ in imagePaths:
        print(f"{imgCounter} Images Trained", end="\r")
        time.sleep(0.008)
        imgCounter += 1
