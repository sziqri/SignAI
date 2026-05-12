import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

DATA_DIR = './test'
dataset_rows = []

print("Starting Holistic extraction process...")

# Open the Holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for sign_label in os.listdir(DATA_DIR):
        sign_folder_path = os.path.join(DATA_DIR, sign_label)
        
        if not os.path.isdir(sign_folder_path):
            continue
            
        print(f"Processing folder: {sign_label}")
        
        for file_name in os.listdir(sign_folder_path):
            file_path = os.path.join(sign_folder_path, file_name)
            cap = cv2.VideoCapture(file_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break 
                
                # Convert color for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                # --- 1. POSE LANDMARKS (132 points) ---
                if results.pose_landmarks:
                    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
                else:
                    pose = np.zeros(33 * 4)
                    
                # --- 2. LEFT HAND LANDMARKS (63 points) ---
                if results.left_hand_landmarks:
                    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
                else:
                    lh = np.zeros(21 * 3)
                    
                # --- 3. RIGHT HAND LANDMARKS (63 points) ---
                if results.right_hand_landmarks:
                    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
                else:
                    rh = np.zeros(21 * 3)
                    
                # Concatenate all arrays into one giant flat array of 258 numbers
                row_data = np.concatenate([pose, lh, rh])
                
                # Convert into a dictionary for Pandas
                row_dict = {'label': sign_label}
                for i, val in enumerate(row_data):
                    row_dict[f'f{i}'] = val
                    
                dataset_rows.append(row_dict)
                
            cap.release()

# Save to a new CSV file
df = pd.DataFrame(dataset_rows)
df.to_csv('holistic_medical_data.csv', index=False)
print(f"Success! Extracted {len(dataset_rows)} frames of spatial data into holistic_medical_data.csv")