# This code trains an LSTM model to recognize poses
# a whole sequence of frame data is passed at once to the LSTM model (this is one input)
# the LSTM model makes a prediction for each frame in the one input sequence
# the sequence length is 70, so it will make 70 predictions on what the pose was (plank, pushup, squat)

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# ====== params ======
csv_path = "pose_dataset.csv"  
sequence_length = 70        # frames of each video
model_output_path = "pose_action_lstm_model.h5"
label_output_path = "label_encoder_classes.npy"

# ====== load CSV data ======
df = pd.read_csv(csv_path)
print("Loaded data shape:", df.shape)

# ====== label encoder ======
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
np.save(label_output_path, le.classes_) 

# ====== Remove labels and extract features ======
features = df.drop(columns=['label']).values
labels = df['label'].values

# ====== Construct time series data (one sample per sequence_length row)=====
X, y = [], []
for i in range(len(features) - sequence_length):
    X.append(features[i:i+sequence_length])
    y.append(labels[i + sequence_length - 1]) 

X = np.array(X)
y = np.array(y)
print("Final dataset shape:", X.shape, y.shape)

# ====== One-hot label ======
y_cat = to_categorical(y)

# ====== Split the training and verification sets ======
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# ====== build LSTM model ======
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ====== training model ======
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# ====== save model ======
model.save(model_output_path)
print(f"Model saved to {model_output_path}")
print(f"Labels saved to {label_output_path}")