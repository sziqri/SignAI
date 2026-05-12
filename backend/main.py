from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import json
import os

# Google Cloud & Gemini Imports for Audio
from google.cloud import texttospeech, speech
import google.generativeai as genai
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOAD YOUR UPGRADED HOLISTIC MODEL ---
print("Loading Holistic AI Model...")
try:
    # IMPORTANT: This must be the new model trained on 258 features!
    model = joblib.load('holistic_medical_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load holistic_medical_model.pkl. Error: {e}")

# Reconstruct the exact 258 column names used during training (f0 to f257)
feature_names = [f'f{i}' for i in range(258)]

# --- 2. WEBSOCKET ENDPOINT (With Crash Protection) ---
@app.websocket("/ws/predict-sign")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Ionic App Connected to Holistic Stream!")
    try:
        while True:
            data = await websocket.receive_text()
            coords = json.loads(data)
            
            # Verify we received exactly 258 data points
            if len(coords) == 258:
                try:
                    df_predict = pd.DataFrame([coords], columns=feature_names)
                    prediction = str(model.predict(df_predict)[0])
                    await websocket.send_text(prediction)
                except Exception as e:
                    print(f"Prediction Error: {e}")
                    await websocket.send_text("Ralat Model")
            else:
                print(f"Data length mismatch! Received {len(coords)} instead of 258.")
                
    except WebSocketDisconnect:
        print("Ionic App Disconnected.")

# --- 3. STANDARD HTTP ENDPOINTS (Audio Features) ---
@app.post("/api/text-to-speech")
async def generate_speech(text: str):
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="ms-MY", name="ms-MY-Standard-A")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        return {"status": "success", "audio_base64": audio_base64}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/speech-to-text")
async def process_speech(file: UploadFile = File(...)):
    try:
        audio_content = await file.read()
        client = speech.SpeechClient()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            sample_rate_hertz=48000,
            language_code="ms-MY"
        )
        response = client.recognize(config=config, audio=audio)
        raw_transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        
        if not raw_transcript:
            return {"simplified_text": "Maaf, suara tidak jelas. Sila ulang."}

        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        genai_model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"Sila ringkaskan dan mudahkan ayat formal ini untuk pesakit pekak menggunakan Bahasa Melayu yang sangat ringkas: '{raw_transcript}'"
        
        simplified_result = genai_model.generate_content(prompt)
        return {"simplified_text": simplified_result.text}
    except Exception as e:
        return {"simplified_text": "Ralat memproses audio."}