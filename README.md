# AI Fitness Trainer

A real-time AI-based fitness trainer that uses pose estimation and action recognition to monitor and correct form during exercises like **squat**, **push-up**, and **plank**.

## Group Member Names

Yulong Feng, Alexander Leon, Ting Li, Duo Xu

## Project Description

Our group is developing an AI fitness trainer that uses real-time pose estimation to evaluate exercise form and provide voice feedback. Built using Python, OpenCV, and MediaPipe, the system will allow users to define three custom motions (squats, push-ups and planks), which are recorded and later used to guide form correction. 

## URLs

Presentation video:


Demo video:


Training dataset:
https://github.com/kreatain/AI-Fitness-Trainer

## Project Structure
## 📁 Project Structure

```
.
├── data/
│   ├── plank/            # Training/testing videos for plank action (MP4 format)
│   ├── pushup/           # Training/testing videos for push-up action
│   └── squat/            # Training/testing videos for squat action
├── squat.py              # Real-time squat feedback and counter
├── pushup.py             # Real-time push-up feedback and counter
├── plank.py              # Real-time plank timer with form correction
├── main.py               # Entry point: recognizes action and routes to correct script
├── generate_pose_dataset.py   # Extracts pose keypoints from videos to CSV
├── pose_dataset.csv      # Extracted keypoints used for LSTM training
├── train_lstm_model.py   # Trains LSTM model for action recognition
├── pose_action_lstm_model.h5  # Trained LSTM model
├── label_encoder_classes.npy  # Encoded class labels
├── requirements.txt      # Python dependencies
└── README.md             # Project overview and instructions
```

## Features

- **Automatic Action Detection**: Using a pre-trained LSTM model to classify user's pose as squat, push-up, or plank.
- **Real-time Form Feedback**:
  - **Squat**: detects depth, knees-in (front-facing), back-lean (side-facing)
  - **Push-up**: detects elbow angle and hips height
  - **Plank**: posture check with live timer
- **Voice Prompts**: Uses `pyttsx3` for real-time audio feedback and counting
- **Restart with 'r' key**: Main script supports restart with countdown
- **Webcam-based**: Uses Mediapipe pose landmarks + OpenCV for visualization

## Setup

### 1. Create virtual environment (optional but recommended)

```bash
python3.11 -m venv mp_env
source mp_env/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
Recommended TensorFlow version for macOS: tensorflow-macos==2.15.0

### 3. Run the main script
```bash
python main.py
```
- Press R to start countdown and begin detection
- The system will detect your action and launch the appropriate form-checking script

## 🧠 Model Training

If you want to re-train the model:

1. Place videos under the `data/` folder (one action per video)
2. Run the following commands:

   ```bash
   python generate_pose_dataset.py
   python train_lstm_model.py
   ```

This will produce the `.csv`, `.h5`, and `.npy` files for prediction.