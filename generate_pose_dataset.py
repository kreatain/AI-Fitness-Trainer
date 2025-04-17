import cv2
import mediapipe as mp
import os
import csv

# initialize mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# input and output setup
input_dir = 'data'  # videos
output_csv = 'pose_dataset.csv'

# write in CSV
csv_file = open(output_csv, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# CSV header
header = []
for i in range(33):
    header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
header.append('label')
csv_writer.writerow(header)

for action in os.listdir(input_dir):
    action_folder = os.path.join(input_dir, action)
    if not os.path.isdir(action_folder):
        continue

    # iterate each video
    for video_name in os.listdir(action_folder):
        if not video_name.endswith('.mp4'):
            continue
        video_path = os.path.join(action_folder, video_name)
        cap = cv2.VideoCapture(video_path)
        print(f"Processing {video_path}...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                row = []
                for lm in results.pose_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z, lm.visibility]
                row.append(action)  # add label
                csv_writer.writerow(row)

        cap.release()
        print(f"Done: {video_path}")

# clear
csv_file.close()
pose.close()
print("All videos have been processed in pose_dataset.csv")