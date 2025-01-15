import os  # accessing the os functions
import checkCamera
import captureImage
import trainImage
import recognize
import checkMicrophone
import captureVoice
import trainVoice
import recognizeVoice


def title_bar():
    os.system('cls') 

    # title of the program

    print("\t**********************************************")
    print("\t****** Face & Voice Recognition System *******")
    print("\t**********************************************")

def mainMenu():
    title_bar()
    print()
    print(10 * "*", "WELCOME MENU", 10 * "*")
    print("[1] Check Camera")
    print("[2] Capture Faces")
    print("[3] Train Images")
    print("[4] Recognize Face")
    print("[5] Check Microphone")
    print("[6] Capture Voice")
    print("[7] Train Voice")
    print("[8] Recognize Voice")
    print
    print("[9] Quit")


    while True:
        try:
            choice = int(input("Enter Choice: "))

            if choice == 1:
                checkCamera_()
                break
            elif choice == 2:
                CaptureFaces()
                break
            elif choice == 3:
                Trainimages()
                break
            elif choice == 4:
                RecognizeFaces()
                break
            elif choice == 5:
                Check_Microphone()
                break
            elif choice == 6:
                Capture_Voice()
                break
            elif choice == 7:
                Train_Voice()
                break

            elif choice == 8:
                Recognize_Voice()
                break
            elif choice == 9:
                print("Thank You")
                break
            else:
                print("Invalid Choice. Enter 1-4")
                mainMenu()
        except ValueError:
            print("Invalid Choice. Enter 1-4\n Try Again")
    exit


# ---------------------------------------------------------
# calling the camera test function from check camera.py file

def checkCamera_():
    checkCamera.camer()
    key = input("Enter any key to return main menu")
    mainMenu()

# --------------------------------------------------------------
# calling the take image function form capture image.py file

def CaptureFaces():
    captureImage.takeImages()
    key = input("Enter any key to return main menu")
    mainMenu()

# -----------------------------------------------------------------
# calling the train images from train_images.py file

def Trainimages():
    trainImage.TrainImages()
    key = input("Enter any key to return main menu")
    mainMenu()

# --------------------------------------------------------------------
# calling the recognize_attendance from recognize.py file

def RecognizeFaces():
    recognize.recognize_faces()
    key = input("Enter any key to return main menu")
    mainMenu()


def Check_Microphone():
    checkMicrophone.check_microphone()
    key= input("Enter any key to return main menu")
    mainMenu()

def Capture_Voice():
    captureVoice.capture_voice()
    key= input("Enter any key to return main menu")
    mainMenu()

def Train_Voice():
    trainVoice.train_voice_model()
    key= input("Enter any key to return main menu")
    mainMenu()

def Recognize_Voice():
    recognizeVoice.recognize_voice()
    key= input("Enter any key to return main menu")
    mainMenu()


mainMenu()