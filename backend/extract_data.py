import cv2
import mediapipe as mp
# Explicitly import the solutions module to prevent the AttributeError
from mediapipe.python.solutions import hands as mp_hands_module
import os
import pandas as pd

# 1. Initialize MediaPipe Hands
# We use the explicitly imported module here
hands = mp_hands_module.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

DATA_DIR = './dataset'
dataset_rows = []

# FIXED: Added the quotation marks back!
print("Starting extraction process...")

# 2. Loop through each medical sign folder (sakit, ubat, doktor, etc.)
for sign_label in os.listdir(DATA_DIR):
    sign_folder_path = os.path.join(DATA_DIR, sign_label)
    
    # Skip hidden files or system folders
    if not os.path.isdir(sign_folder_path):
        continue

    print(f"Processing folder: {sign_label}")

    # 3. Loop through every video or image in that folder
    for file_name in os.listdir(sign_folder_path):
        file_path = os.path.join(sign_folder_path, file_name)
        
        cap = cv2.VideoCapture(file_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break 
            
            # MediaPipe requires RGB color format, OpenCV loads in BGR
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # 4. If a hand is detected in the frame, grab the coordinates
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row_data = {'label': sign_label}
                    
                    # Extract the X, Y, Z for all 21 points on the hand
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        row_data[f'x{i}'] = landmark.x
                        row_data[f'y{i}'] = landmark.y
                        row_data[f'z{i}'] = landmark.z
                        
                    dataset_rows.append(row_data)

        cap.release()

# 5. Save all the numbers to a CSV file
df = pd.DataFrame(dataset_rows)
df.to_csv('medical_signs_data.csv', index=False)
print(f"Success! Extracted {len(dataset_rows)} frames of data into medical_signs_data.csv")