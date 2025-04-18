# This is a real-time pose prediction system

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import time

# init
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose() # the landmark detector
model = load_model('pose_action_lstm_model.h5') # the pose predictor
label_encoder = np.load('label_encoder_classes.npy', allow_pickle=True) # class labels

# model params
sequence_length = 70 # frames per sequence
movement_compare_gap = 40
movement_threshold = 0.02  

frame_window = deque(maxlen=sequence_length) # recent keypoints
keypoints_history = deque(maxlen=movement_compare_gap + 1) # historical keypoints

cap = cv2.VideoCapture(0)

# Extract 33 key points
def extract_keypoints(results):
    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return keypoints
    return None

# will help differentiate the pushup from the plank
def is_moving_from_past(history, current_kp, threshold):
    if len(history) < movement_compare_gap:
        return True 
    past_kp = history[0]
    diff = np.abs(np.array(current_kp) - np.array(past_kp))
    avg_movement = np.mean(diff)
    return avg_movement > threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    keypoints = extract_keypoints(results)
    if keypoints:
        frame_window.append(keypoints)
        keypoints_history.append(keypoints)
        moving = is_moving_from_past(keypoints_history, keypoints, movement_threshold)
    else:
        moving = True  

    # predict
    if len(frame_window) == sequence_length:
        input_data = np.array(frame_window).reshape(1, sequence_length, len(frame_window[0]))
        prediction = model.predict(input_data, verbose=0)
        label = label_encoder[np.argmax(prediction)]

        if label == "pushup" and not moving:
            label = "plank"
    else:
        label = "Collecting..."

    # visualize key points
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # text
    cv2.putText(image, f'Prediction: {label}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pose Action Recognition', image)
    if cv2.waitKey(10) & 0xFF == ord('q'): # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()