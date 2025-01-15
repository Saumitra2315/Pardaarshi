import cv2
import os

# Check if input is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Capture images function
def takeImages():
    name = input("Enter Your Name: ")

    if name.isalpha():
        # Ensure the TrainingImage folder exists
        os.makedirs("TrainingImage", exist_ok=True)

        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (10, 159, 255), 2)
                sampleNum += 1
                # Save the captured face images
                cv2.imwrite(
                    f"TrainingImage/{name}.{sampleNum}.jpg",
                    gray[y:y+h, x:x+w]
                )
                cv2.imshow('frame', img)

            # Break conditions: 'q' key press or 100 images captured
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 100:
                break

        cam.release()
        cv2.destroyAllWindows()
        print(f"Images Saved for Name: {name}")
    else:
        print("Enter a valid alphabetical name")
