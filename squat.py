# This is a real-time squat form feedback system

import cv2
import numpy as np
import mediapipe as mp
import math
import pyttsx3

# Initialize Mediapipe and TTS engines
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
engine = pyttsx3.init()
engine.setProperty('rate', 150) # 150 words per minute

# used to compute two angles
#  - angle formed by: hip, knee, ankle
#  - angle formed by: shoulder, hip, knee
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

cap = cv2.VideoCapture(0)
state = "up"
counter = 0
min_knee_angle = 180
min_knee_dist = 1.0
min_chest_hip_angle = 180

while cap.isOpened():
    error_flags = {"depth": False, "knees_in": False, "leaning": False}
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    feedback = "Waiting..."

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark # gets the landmarks

        # gets spatial coordinates for the different body parts
        # left side
        left_hip = [landmarks[23].x, landmarks[23].y]
        left_knee = [landmarks[25].x, landmarks[25].y]
        left_ankle = [landmarks[27].x, landmarks[27].y]
        left_shoulder = [landmarks[11].x, landmarks[11].y]

        # right side
        right_hip = [landmarks[24].x, landmarks[24].y]
        right_knee = [landmarks[26].x, landmarks[26].y]
        right_ankle = [landmarks[28].x, landmarks[28].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]

        # computes the knee angle (formed by hip, knee, ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        # computes the angle of the torso when squatting
        left_chest_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_chest_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        avg_chest_hip_angle = (left_chest_hip_angle + right_chest_hip_angle) / 2

        is_front_facing = abs(landmarks[11].x - landmarks[12].x) > 0.1

        # checks to see if knees are caving in
        hip_width = abs(landmarks[23].x - landmarks[24].x)
        knee_width = abs(landmarks[25].x - landmarks[26].x)
        knee_hip_ratio = knee_width / hip_width if hip_width > 0 else 1.0

        # update min angles
        min_knee_angle = min(min_knee_angle, avg_knee_angle)
        min_chest_hip_angle = min(min_chest_hip_angle, avg_chest_hip_angle)
        min_knee_dist = min(min_knee_dist, knee_hip_ratio)

        if not is_front_facing and avg_knee_angle < 120 and state == "up":
            state = "down"

        if is_front_facing and avg_knee_angle < 170 and state == "up":
            state = "down"

        if avg_knee_angle > 170 and state == "down":
            state = "up"

            # Error Detection
            error_flags = {"depth": False, "knees_in": False, "leaning": False}
            if not is_front_facing and min_knee_angle > 95:
                error_flags["depth"] = True # not going deep enough
            if is_front_facing and min_knee_angle > 160:
                error_flags["depth"] = True # not going deep enough
            if is_front_facing and min_knee_dist < 0.5:
                error_flags["knees_in"] = True # knees are collapsing inward
            if not is_front_facing and min_chest_hip_angle < 40:
                error_flags["leaning"] = True # torso leaning forward too much

            # Count
            if not any(error_flags.values()):
                counter += 1
                engine.say(str(counter))
                engine.runAndWait()
            else:
                if error_flags["knees_in"]:
                    engine.say("Keep knees aligned!")
                elif error_flags["leaning"]:
                    engine.say("Keep back straight!")
                elif error_flags["depth"]:
                    engine.say("Go lower!")
                engine.runAndWait()

            # reset min values
            min_knee_angle = 180
            min_chest_hip_angle = 180
            min_knee_dist = 1.0
            error_flags = {"depth": False, "knees_in": False, "leaning": False}

        form_errors = [k for k, v in error_flags.items() if v] if state == "up" else []
        cv2.putText(image, f"Knee angle: {int(avg_knee_angle)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, f"Form errors: {', '.join(form_errors) or 'None'}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"Squats: {counter}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow('Squat Feedback', image)
    if cv2.waitKey(10) & 0xFF == ord('q'): # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
