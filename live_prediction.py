import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import speech_recognition as sr
import threading
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Configure the Generative AI model
api=os.getenv("API_KEY")
genai.configure(api_key=api)
LLMmodel = genai.GenerativeModel("gemini-1.5-flash")

# Load the gesture recognition model
model = load_model('gesture_recog.h5')

r = None

# Mediapipe hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Variables for toggling
togg_recog = False
speech_recog = False
recognized_text = "None"

# Initialize video capture
cap = cv.VideoCapture(0)

# Gesture class mapping
class_mapping = {0: 'Backward', 1: 'Down', 2: 'Forward', 3: 'Left', 4: 'Right', 5: 'Up'}

def toggle_recog():
    """Toggle hand gesture recognition."""
    global togg_recog
    togg_recog = not togg_recog

def toggle_speech_recog():
    """Toggle speech recognition."""
    global speech_recog
    speech_recog = not speech_recog
    if speech_recog:
        threading.Thread(target=speech_recognition_thread, daemon=True).start()

def speech_recognition_thread():
    """Speech recognition processing."""
    global recognized_text, speech_recog, r
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Speech recognition started...")

    while speech_recog:
        try:
            with mic as source:
                print("Listening for speech...")
                audio = recognizer.listen(source, timeout=5)
                recognized_text = recognizer.recognize_google(audio)
                response1 = LLMmodel.generate_content(f"Just answer in Yes or No Q:{recognized_text}? A:")
                r = response1.text
                print(f"Recognized Speech: {recognized_text}")
                print(f"Answer: {r}")
        except sr.WaitTimeoutError:
            print("Speech recognition timed out.")
        except sr.UnknownValueError:
            recognized_text = "Unknown Speech"
        except Exception as e:
            print(f"Speech recognition error: {e}")
            recognized_text = "Error"

# Main loop
while True:
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)  # Flip the frame horizontally
    cap.set(cv.CAP_PROP_FPS, 15)
    
    # Hand gesture recognition
    if togg_recog:
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(rgb_image)

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                landmarks = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmarks)
                prediction = prediction.flatten()
                prediction1 = [x if x >= 0.5 else 0 for x in prediction]
                class_idx = np.argmax(prediction1)
                gesture_name = class_mapping[class_idx]

                # Display gesture name for each hand
                cv.putText(frame, f"Hand {idx+1}: {gesture_name}", (50, 50 + idx * 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            cv.putText(frame, "Detected: None", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    else:
        cv.putText(frame, "Hand Recognition Paused", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Speech recognition display
    if speech_recog:
        cv.putText(frame, f"Speech: {recognized_text} A: {r}", (50, 450), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    else:
        cv.putText(frame, "Speech Recognition Paused", (50, 450), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Display the frame
    cv.imshow("Hand Gesture and Speech Recognition", frame)
    key = cv.waitKey(1) & 0xFF
    
    # Keyboard controls
    if key == ord('q'):  # Quit the program
        break
    elif key == ord('t'):  # Toggle hand gesture recognition
        toggle_recog()
    elif key == ord('s'):  # Toggle speech recognition
        toggle_speech_recog()

# Cleanup
cap.release()
cv.destroyAllWindows()
