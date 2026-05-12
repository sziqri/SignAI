import cv2
import mediapipe as mp
import os

# 1. Initialize MediaPipe Holistic and Drawing Utils
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = './test'

print("Starting Dataset Visualizer...")
print("CONTROLS:")
print(" - Press 'n' to skip to the NEXT video/image.")
print(" - Press 'q' to QUIT the entire program.")
print("-" * 30)

# 2. Open the Holistic Model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through each sign language folder
    for sign_label in os.listdir(DATA_DIR):
        sign_folder_path = os.path.join(DATA_DIR, sign_label)
        
        if not os.path.isdir(sign_folder_path):
            continue
            
        print(f"\n>>> Opening Folder: {sign_label.upper()} <<<")
        
        # Loop through every file in the folder
        for file_name in os.listdir(sign_folder_path):
            file_path = os.path.join(sign_folder_path, file_name)
            
            print(f"Playing: {file_name}")
            cap = cv2.VideoCapture(file_path)
            
            # This variable helps us break completely if you press 'q'
            quit_program = False 
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                # If video ends or image is done, break out of the while loop to get the next file
                if not ret:
                    break
                
                # Convert color for MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # Extract the skeleton
                results = holistic.process(image_rgb)
                
                image_rgb.flags.writeable = True
                
                # --- DRAWING THE SKELETON ---
                # Draw Body Pose
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                    )
                # Draw Left Hand
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                    )
                # Draw Right Hand
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                    )

                # Add text to the screen so you know what you are watching
                cv2.putText(frame, f"Label: {sign_label} | File: {file_name}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Show the video player
                cv2.imshow('Dataset Visualizer', frame)

                # Handle Keyboard Controls
                key = cv2.waitKey(30) & 0xFF
                
                if key == ord('q'):
                    quit_program = True # Signal to quit everything
                    break
                elif key == ord('n'):
                    break # Break the current video loop, skips to the next file
            
            cap.release()
            
            # If user pressed 'q', break out of the file loop
            if quit_program:
                break
                
        # If user pressed 'q', break out of the folder loop
        if quit_program:
            print("Visualizer closed by user.")
            break

cv2.destroyAllWindows()
print("Visualization complete.")