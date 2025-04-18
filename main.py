# This is a real-time exercise detection and feedback system

import cv2
import numpy as np
import time
import subprocess # runs external scripts based on action
import mediapipe as mp
import pyttsx3 # text-to-speech for spoken feedback
from collections import deque
from tensorflow.keras.models import load_model # loads and runs an ML model for action recognition

# Initializes the voice engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load the LSTM model and label encoder
model = load_model('pose_action_lstm_model.h5') # loads the pretrained model that classifies sequences of body poses (plank, pushup, squat)
label_encoder = np.load('label_encoder_classes.npy', allow_pickle=True) # holds the actual class names (plank, pushup, squat)

# initialize Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils # for landmark drawing

# Extract key points into a feature vector
def extract_keypoints(results):
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return keypoints
    return None

# Countdown function
def countdown(seconds=5):
    for i in range(seconds, 0, -1):
        print(f"Starting in {i}...")
        speak(str(i)) # voice-based countdown
        time.sleep(1)
    speak("Start now")

# initialize camera
cap = cv2.VideoCapture(0)
sequence_length = 70
movement_compare_gap = 40
movement_threshold = 0.02
frame_window = deque(maxlen=sequence_length) # stores pose keypoints later
keypoints_history = deque(maxlen=movement_compare_gap + 1) # stores keypoints for tracking motion
mode = "idle"
label_stable_count = 0 # helps ensure consistency between labels across several frames
label_threshold = 10
last_label = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # converts to RGB for MediaPipe
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB) # flips the image horizontally
    image.flags.writeable = False
    results = pose.process(image)

    keypoints = extract_keypoints(results) # extracts the keypoints
    if keypoints:
        frame_window.append(keypoints)
        keypoints_history.append(keypoints)
        moving = True
        if len(keypoints_history) >= movement_compare_gap:
            diff = np.abs(np.array(keypoints_history[-1]) - np.array(keypoints_history[0]))
            moving = np.mean(diff) > movement_threshold # checks if the person is moving by comparing the difference in keypoints over time
    else:
        moving = True

    # state control
    if mode == "idle":
        cv2.putText(frame, "Press 'r' to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif mode == "detect":
        if len(frame_window) == sequence_length:
            # preparing the sequence of the keypoints for the ML model
            input_data = np.array(frame_window).reshape(1, sequence_length, len(frame_window[0]))
            prediction = model.predict(input_data, verbose=0)
            label = label_encoder[np.argmax(prediction)] # converts prediction to a label

            # Interesting approach
            if label == "pushup" and not moving:
                label = "plank"

            # setting label_stable_count
            if label == last_label:
                label_stable_count += 1
            else:
                label_stable_count = 0
                last_label = label

            if label_stable_count >= label_threshold:
                print(f"Detected action: {label}")
                speak(f"{label} detected. Starting now.") # announces detected action
                if label == "squat":
                    subprocess.run(["python", "squat.py"])
                elif label == "pushup":
                    subprocess.run(["python", "pushup.py"])
                elif label == "plank":
                    subprocess.run(["python", "plank.py"])
                
                # reset for next detection cycle
                mode = "idle"
                frame_window.clear()
                keypoints_history.clear()
                label_stable_count = 0
                last_label = ""

    # draws the pose landmarks on the image (which comes from frame)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Main Program", image)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('r'): # starts the detection mode with countdown
        frame_window.clear()
        keypoints_history.clear()
        label_stable_count = 0
        last_label = ""
        countdown(5)
        mode = "detect"
    elif key == ord('q'): # quits the program
        break

cap.release()
cv2.destroyAllWindows()
