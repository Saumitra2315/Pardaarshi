import cv2

# -------------------------
def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # Create LBPH face recognizer
    recognizer.read("./TrainingImageLabel/Trainner.yml")  # Load the trained model
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start real-time video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set video width
    cam.set(4, 480)  # Set video height

    # Define minimum window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        _, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, 1.2, 5, minSize=(int(minW), int(minH)), flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (10, 159, 255), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

            if conf < 100:  # Confidence threshold
                label = f"Person {Id}"  # Replace this with actual labels if available
                confstr = f"  {round(100 - conf)}%"
                display_text = f"{label} is detected"
            else:
                display_text = "Unknown is detected"
                confstr = f"  {round(100 - conf)}%"

            cv2.putText(im, display_text, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(im, confstr, (x + 5, y + h - 5), font, 1, (0, 255, 0), 1)

        cv2.imshow("Face Recognition", im)

        if cv2.waitKey(1) == ord("q"):  # Press 'q' to quit
            break

    print("Face recognition ended.")
    cam.release()
    cv2.destroyAllWindows()
