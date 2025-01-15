import speech_recognition as sr

def check_microphone():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    try:
        print("Adjusting for ambient noise. Please wait...")
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Microphone is ready. Speak now!")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        
        print("Recognizing your speech...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        
        # Save the recognized text to a file
        with open("recognized_speech.txt", "w") as file:
            file.write(text)

        return text

    except sr.WaitTimeoutError:
        print("No speech detected. Please try again.")
    except sr.UnknownValueError:
        print("Speech was not clear. Could not recognize.")
    except sr.RequestError as e:
        print(f"Error with the speech recognition service: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

