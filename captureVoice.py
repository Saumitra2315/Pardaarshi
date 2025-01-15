import os
import sounddevice as sd
import wave

def capture_voice():
    """
    Capture and save a user's voice sample for training.
    """
    # Configuration
    sample_rate = 16000  # Consistent sample rate with training
    duration = 5         # Duration in seconds
    output_dir = "TrainingVoice"  # Directory to save audio samples
    default_phrase = "Open Sesame"  # Default phrase to say for training

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prompt for user name
    user_name = input("Enter your name (for labeling): ").strip()
    if not user_name:
        print("Invalid input. Please provide a name.")
        return

    # Phrase to record
    print(f"The default phrase for training is: \"{default_phrase}\"")
    confirm = input("Would you like to use this phrase? (yes/no): ").strip().lower()
    if confirm != "yes":
        custom_phrase = input("Enter the phrase you would like to use: ").strip()
        if not custom_phrase:
            print("Invalid input. Using default phrase.")
        else:
            default_phrase = custom_phrase

    # File path
    file_count = len([f for f in os.listdir(output_dir) if f.startswith(user_name)]) + 1
    file_path = os.path.join(output_dir, f"{user_name}_{file_count}.wav")

    try:
        print("\nGet ready to record your voice for training.")
        print(f"Please say the following phrase clearly: \"{default_phrase}\"")
        print(f"Recording will start in 3 seconds. Speak clearly for {duration} seconds.\n")

        # Delay before recording
        sd.sleep(3000)

        print("Recording...")
        # Record audio
        audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished

        # Save to WAV file
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

        print(f"Audio captured successfully and saved as {file_path}")

    except Exception as e:
        print(f"An error occurred while recording audio: {e}")
