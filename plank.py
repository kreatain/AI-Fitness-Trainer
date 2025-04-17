import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import time

# initialize
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

# turn on camera
cap = cv2.VideoCapture(0)

# state variables
start_time = 0
elapsed_time = 0
plank_active = False
last_announce = 0

# error speaking reminder gap
ERROR_SPEAK_INTERVAL = 3.0  # 3 seconds for repeate error
last_error_speak_time = {
    "hips_sagging": 0,
    "head_bad": 0
}

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
        left_hip = [landmarks[23].x, landmarks[23].y]
        right_hip = [landmarks[24].x, landmarks[24].y]
        nose = [landmarks[0].x, landmarks[0].y]

        # average
        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        avg_hip_y = (left_hip[1] + right_hip[1]) / 2
        mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2,
                        (left_shoulder[1] + right_shoulder[1]) / 2]
        mid_hip = [(left_hip[0] + right_hip[0]) / 2,
                   (left_hip[1] + right_hip[1]) / 2]

        head_angle = calculate_angle(nose, mid_shoulder, mid_hip)

        # check 
        hips_sagging = avg_hip_y > avg_shoulder_y + 0.05
        head_bad = head_angle < 135 or head_angle > 195 
        current_time = time.time()

        if hips_sagging and current_time - last_error_speak_time["hips_sagging"] > ERROR_SPEAK_INTERVAL:
            feedback = "Raise your hips!"
            speak("Raise your hips!")
            plank_active = False
            last_error_speak_time["hips_sagging"] = current_time
            start_time = time.time()  # pause time count
        elif head_bad and current_time - last_error_speak_time["head_bad"] > ERROR_SPEAK_INTERVAL:
            feedback = "Keep head neutral!"
            speak("Keep head neutral!")
            last_error_speak_time["head_bad"] = current_time 
            plank_active = False
            start_time = time.time()  
        else:
            feedback = "Hold it!"
            if not plank_active:
                speak("Hold it!")
                plank_active = True
                start_time = time.time() - elapsed_time  # recovery timing
            else:
                elapsed_time = time.time() - start_time
                if int(elapsed_time) // 10 > last_announce:
                    last_announce = int(elapsed_time) // 10
                    speak(f"{last_announce * 10} seconds")

        # text on the top left corner
        cv2.putText(image, f"Head angle: {int(head_angle)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(image, f"Feedback: {feedback}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if "Raise" in feedback or "Keep" in feedback else (0, 255, 0), 2)
        cv2.putText(image, f"Time: {int(elapsed_time)}s", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow("Plank Feedback", image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()