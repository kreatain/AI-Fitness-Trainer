import cv2
import numpy as np
import mediapipe as mp
import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360 - angle

# camera
cap = cv2.VideoCapture(0)
stage = 'up'
counter = 0

# movement tracking
min_elbow_angle = 180
hip_higher_than_shoulder = False

while cap.isOpened():
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
        landmarks = results.pose_landmarks.landmark

        # extract key points
        left_shoulder = [landmarks[11].x, landmarks[11].y]
        right_shoulder = [landmarks[12].x, landmarks[12].y]
        left_elbow = [landmarks[13].x, landmarks[13].y]
        right_elbow = [landmarks[14].x, landmarks[14].y]
        left_wrist = [landmarks[15].x, landmarks[15].y]
        right_wrist = [landmarks[16].x, landmarks[16].y]
        left_hip = [landmarks[23].x, landmarks[23].y]
        right_hip = [landmarks[24].x, landmarks[24].y]

        # average
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        avg_elbow_angle = (left_angle + right_angle) / 2

        if avg_elbow_angle < 150 and stage == 'up':
            stage = 'down'
            min_elbow_angle = avg_elbow_angle
            hip_higher_than_shoulder = False

        if stage == 'down':
            min_elbow_angle = min(min_elbow_angle, avg_elbow_angle)
            if avg_hip_y < avg_shoulder_y - 0.05:
                hip_higher_than_shoulder = True

        if avg_elbow_angle > 160 and stage == 'down':
            stage = 'up'

            error_msgs = []
            if min_elbow_angle > 90:
                error_msgs.append("Go deeper")
            if hip_higher_than_shoulder:
                error_msgs.append("Lower your hips")

            if error_msgs:
                for msg in error_msgs:
                    speak(msg)
                feedback = " / ".join(error_msgs)
            else:
                counter += 1
                speak(f"Push-up {counter}")
                feedback = "Good form!"

        cv2.putText(image, f"Elbow angle: {int(avg_elbow_angle)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # text 
    cv2.putText(image, f"Feedback: {feedback}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if "Good" in feedback else (0, 0, 255), 2)

    cv2.putText(image, f"Push-up Count: {counter}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow('Push-up Feedback', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()