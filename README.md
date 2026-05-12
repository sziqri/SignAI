# SmartSign AI - Medical Sign Language Translation System
## Project Overview
SmartSign AI is a real-time, bidirectional translation system designed to bridge the communication gap between healthcare professionals and deaf or hard-of-hearing patients.

By utilizing computer vision, machine learning, and advanced large language models, the system translates Malaysian Sign Language (BIM) into text, and simplifies complex medical instructions from doctors into easy-to-understand text and speech.

## Core Features
* ** Doctor Panel (Speech & Simplification)**
Speech-to-Text Transcription: Doctors can record instructions directly through the app using Google Cloud Speech-to-Text.

AI Medical Simplification: Integrates Gemini 2.5 Flash to automatically translate complex medical jargon into simplified, easy-to-understand language tailored for deaf patients.

Text-to-Speech (TTS): Plays back simplified instructions using Google Cloud TTS.

Quick Replies: Pre-configured quick response buttons (e.g., "Faham", "Tunggu", "Ubat") for rapid communication.

* ** Patient Panel (Sign Language Detection)**
Real-Time Sign Translation: Streams live camera feed via WebSockets to the backend, utilizing MediaPipe Holistic and a custom Random Forest model (258 spatial features) to detect and translate signs instantly.

Smart Sentence Builder: Patients can hold a sign for 2 seconds to capture it, building full sentences word-by-word.

Text-to-Sign Dictionary: Allows patients to type a phrase and retrieve the corresponding sign language visual guides.

## Tech Stack
Frontend: Ionic Framework, HTML/SCSS, TypeScript

Backend: FastAPI, WebSockets (for ultra-low latency inference), Uvicorn

Computer Vision: OpenCV, MediaPipe Holistic

Machine Learning: Scikit-Learn (Random Forest Classifier), Pandas, Joblib

Cloud AI Services: Google Cloud Speech-to-Text, Google Cloud Text-to-Speech, Google Gemini API

## Project Architecture
main.py: The core FastAPI server handling WebSocket streams for real-time inference, and HTTP endpoints for Google Cloud & Gemini integrations.

train_holistic.py: Training script that ingests the 258-feature spatial dataset (holistic_medical_data.csv) and outputs the optimized holistic_medical_model.pkl Random Forest model.

track.py: A dataset visualization and extraction tool that overlays MediaPipe pose and hand landmarks on training videos.

home.page.html: The interactive Ionic frontend UI, featuring dynamic dual-panel layouts for both doctors and patients.

## Setup & Installation
Step 1: Backend Setup (Python)

Clone the repository and install the backend dependencies.


# Clone the repository
```Bash
git clone https://github.com/yourusername/smartsign-ai.git
cd smartsign-ai/backend
```

# Create and activate a virtual environment
```Bash
python -m venv venv
```
# On Linux/macOS:
```Bash
source venv/bin/activate
```
# On Windows:
```Bash
venv\Scripts\activate
```

# Install dependencies
```Bash
pip install fastapi uvicorn pandas scikit-learn mediapipe opencv-python google-cloud-speech google-cloud-texttospeech google-generativeai joblib python-multipart
```

# Set Environment Variables
```Bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/gcp-service-account.json"
```

# Run the FastAPI Server
```Bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Step 2: Frontend Setup (Ionic)

Open a new terminal window to start the Ionic frontend.

```Bash
cd ../frontend
```

# Install dependencies
```Bash
npm install
```

# Run the Ionic development server
```Bash
ionic serve
```

## Model Training (Optional)
If you wish to retrain the holistic model with new signs:

Gather video data and use track.py to verify landmark extraction.

Ensure your extracted dataset is saved as holistic_medical_data.csv in the root directory.

Run the training script:

```Bash
python train_holistic.py
```
The new holistic_medical_model.pkl will be generated and automatically utilized by the FastAPI server on restart.
