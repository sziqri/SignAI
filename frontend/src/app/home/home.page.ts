import { Component, ElementRef, ViewChild, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  IonHeader, IonToolbar, IonTitle, IonContent, IonButton, IonIcon,
  IonCard, IonCardHeader, IonCardSubtitle, IonCardContent,
  IonGrid, IonRow, IonCol, IonItem, IonInput
} from '@ionic/angular/standalone';

import { Holistic, POSE_CONNECTIONS, HAND_CONNECTIONS } from '@mediapipe/holistic';
import { Camera } from '@mediapipe/camera_utils';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import axios from 'axios';

interface SignWordResult {
  word: string;
  path: string;
  found: boolean;
}

// ── Sentence builder tuning ──────────────────────────────────
const HOLD_DURATION_MS       = 2500;  // must hold sign this long to capture
const MIN_CONFIDENCE_STREAK  = 12;    // consecutive same-prediction frames before timer starts
const POST_CAPTURE_BLOCK_MS  = 2500;  // ALL capturing is blocked for this long after any capture
const IDLE_THRESHOLD_MS      = 3000;  // idle this long → sentence marked "done"

// Words the model outputs when no meaningful sign is being made.
// Add your model's idle/neutral class name here.
const IDLE_LABELS = new Set(['...', '', 'TIADA', 'NEUTRAL', 'IDLE', 'NONE']);
// ─────────────────────────────────────────────────────────────

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
  standalone: true,
  imports: [
    CommonModule, FormsModule,
    IonHeader, IonToolbar, IonTitle, IonContent, IonButton, IonIcon,
    IonCard, IonCardHeader, IonCardSubtitle, IonCardContent,
    IonGrid, IonRow, IonCol, IonItem, IonInput
  ]
})
export class HomePage implements OnInit, OnDestroy {
  @ViewChild('videoElement', { static: true }) videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasElement', { static: true }) canvasElement!: ElementRef<HTMLCanvasElement>;

  // ── UI state ─────────────────────────────────────────────
  staffText: string = "Menunggu suara doktor...";
  liveSignTranslation: string = "...";
  isRecording: boolean = false;
  activeTab: string = 'doctor';

  // ── Connection ───────────────────────────────────────────
  ws!: WebSocket;
  camera!: Camera;
  backendUrl: string = "http://127.0.0.1:8000";
  wsUrl: string    = "ws://127.0.0.1:8000/ws/predict-sign";

  // ── Audio ────────────────────────────────────────────────
  mediaRecorder: any;
  audioChunks: any[] = [];
  lastSendTime: number = 0;

  // ── Text-to-Sign ─────────────────────────────────────────
  textToSearch: string = "";
  lastSearchedSentence: string = "";
  signImageResults: SignWordResult[] = [];

  // ── Sentence builder ─────────────────────────────────────
  isSignRecording: boolean = false; // controls whether sign→sentence capturing is active
  builtSentence: string[] = [];
  selectedWordIndex: number = -1;   // index of chip tapped for deletion (-1 = none selected)

  currentHoldWord: string = "";
  holdStartTime: number = 0;
  holdProgress: number = 0;
  confidenceStreak: number = 0;

  // Timestamp of the last successful capture — ALL capturing is
  // blocked until now > lastCaptureTime + POST_CAPTURE_BLOCK_MS.
  // This prevents the same idle pose from immediately re-triggering.
  lastCaptureTime: number = 0;

  lastSignSeenTime: number = 0;
  isSentenceComplete: boolean = false;

  constructor() {}

  ngOnInit() {
    this.connectWebSocket();
    this.setupHolistic();
  }

  ngOnDestroy() {
    if (this.ws) this.ws.close();
    if (this.camera) this.camera.stop();
  }

  // ── Tab switching: pause/resume camera ───────────────────
  switchTab(tab: string) {
    this.activeTab = tab;
    if (!this.camera) return;
    if (tab === 'patient') {
      this.camera.start();
    } else {
      this.camera.stop();
      // Clear the canvas so the frozen frame doesn't linger visibly
      const canvasEl = this.canvasElement.nativeElement;
      const ctx = canvasEl.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    }
  }

  // ── 1. WebSocket ──────────────────────────────────────────
  connectWebSocket() {
    this.ws = new WebSocket(this.wsUrl);
    this.ws.onopen  = () => console.log("WebSocket connected.");
    this.ws.onmessage = (event) => {
      const prediction: string = event.data;
      this.liveSignTranslation = prediction;          // always show live feed
      if (this.isSignRecording) {
        this.processPrediction(prediction);           // only capture when recording
      }
    };
    this.ws.onclose = () => {
      console.log("WebSocket closed, reconnecting...");
      setTimeout(() => this.connectWebSocket(), 3000);
    };
  }

  // ── 2. Sentence builder logic ─────────────────────────────
  processPrediction(prediction: string) {
    const now = Date.now();
    const isIdle = IDLE_LABELS.has(prediction.trim().toUpperCase()) ||
                   IDLE_LABELS.has(prediction.trim());

    // ── A. Idle / no meaningful sign ──
    if (isIdle) {
      this.resetHoldState();
      if (this.builtSentence.length > 0 && !this.isSentenceComplete) {
        if (now - this.lastSignSeenTime > IDLE_THRESHOLD_MS) {
          this.isSentenceComplete = true;
        }
      }
      return;
    }

    this.lastSignSeenTime = now;
    this.isSentenceComplete = false;

    // ── B. Post-capture global block ──
    // After any capture, freeze ALL capturing for POST_CAPTURE_BLOCK_MS.
    // This is the key fix: even if the user stays in the same pose,
    // the streak cannot complete until the block expires.
    if (now - this.lastCaptureTime < POST_CAPTURE_BLOCK_MS) {
      // Show a dimmed hold ring so user knows they're in cooldown
      this.holdProgress = 0;
      this.currentHoldWord = prediction;
      return;
    }

    // ── C. Confidence streak ──
    if (prediction === this.currentHoldWord) {
      this.confidenceStreak++;
    } else {
      // Different sign seen — start fresh streak, reset timer
      this.confidenceStreak = 1;
      this.currentHoldWord = prediction;
      this.holdStartTime = 0;
      this.holdProgress = 0;
    }

    // ── D. Hold timer (only after minimum streak) ──
    if (this.confidenceStreak >= MIN_CONFIDENCE_STREAK) {
      if (this.holdStartTime === 0) {
        this.holdStartTime = now;
      }
      const elapsed = now - this.holdStartTime;
      this.holdProgress = Math.min(100, (elapsed / HOLD_DURATION_MS) * 100);

      if (elapsed >= HOLD_DURATION_MS) {
        this.captureWord(prediction, now);
      }
    }
  }

  captureWord(word: string, now: number) {
    this.builtSentence.push(word);
    this.lastCaptureTime = now;   // starts the global cooldown block
    this.resetHoldState();
  }

  resetHoldState() {
    this.currentHoldWord = "";
    this.holdStartTime   = 0;
    this.holdProgress    = 0;
    this.confidenceStreak = 0;
  }

  // ── Word chip selection & deletion ───────────────────────

  /** Tap a chip once to select it (highlight), tap again to delete it. */
  tapWordChip(index: number) {
    if (this.selectedWordIndex === index) {
      // Second tap → delete
      this.builtSentence.splice(index, 1);
      this.selectedWordIndex = -1;
      this.isSentenceComplete = false;
    } else {
      // First tap → select
      this.selectedWordIndex = index;
    }
  }

  /** Dismiss selection without deleting (tap anywhere outside chips). */
  dismissSelection() {
    this.selectedWordIndex = -1;
  }

  // ── Sign recording control ────────────────────────────────
  startSignRecording() {
    this.isSignRecording = true;
    this.isSentenceComplete = false;
    this.lastCaptureTime = 0;
    this.resetHoldState();
  }

  stopSignRecording() {
    this.isSignRecording = false;
    this.resetHoldState();
    if (this.builtSentence.length > 0) {
      this.isSentenceComplete = true;
    }
  }

  clearBuiltSentence() {
    this.builtSentence = [];
    this.selectedWordIndex = -1;
    this.isSentenceComplete = false;
    this.lastCaptureTime = 0;
    this.resetHoldState();
  }

  removeLastWord() {
    this.builtSentence.pop();
    this.selectedWordIndex = -1;
    this.isSentenceComplete = false;
  }

  speakBuiltSentence() {
    const sentence = this.builtSentence.join(' ');
    if (sentence) this.speakText(sentence);
  }

  // ── 3. MediaPipe Holistic ─────────────────────────────────
  setupHolistic() {
    const videoEl  = this.videoElement.nativeElement;
    const canvasEl = this.canvasElement.nativeElement;
    const ctx      = canvasEl.getContext('2d')!;

    const holistic = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
    });

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    holistic.onResults((results) => {
      ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
      ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

      if (results.poseLandmarks) {
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 2 });
        drawLandmarks(ctx, results.poseLandmarks, { color: '#FF0000', lineWidth: 1 });
      }
      if (results.leftHandLandmarks) {
        drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#CC0000', lineWidth: 2 });
        drawLandmarks(ctx, results.leftHandLandmarks, { color: '#00FF00', lineWidth: 1 });
      }
      if (results.rightHandLandmarks) {
        drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#00CC00', lineWidth: 2 });
        drawLandmarks(ctx, results.rightHandLandmarks, { color: '#FF0000', lineWidth: 1 });
      }

      let coordinateData: number[] = [];
      if (results.poseLandmarks) {
        results.poseLandmarks.forEach(pt => coordinateData.push(pt.x, pt.y, pt.z, pt.visibility || 0));
      } else { for (let i = 0; i < 132; i++) coordinateData.push(0); }

      if (results.leftHandLandmarks) {
        results.leftHandLandmarks.forEach(pt => coordinateData.push(pt.x, pt.y, pt.z));
      } else { for (let i = 0; i < 63; i++) coordinateData.push(0); }

      if (results.rightHandLandmarks) {
        results.rightHandLandmarks.forEach(pt => coordinateData.push(pt.x, pt.y, pt.z));
      } else { for (let i = 0; i < 63; i++) coordinateData.push(0); }

      const now = Date.now();
      if (now - this.lastSendTime > 150) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN && coordinateData.length === 258) {
          if (results.leftHandLandmarks || results.rightHandLandmarks || results.poseLandmarks) {
            this.ws.send(JSON.stringify(coordinateData));
            this.lastSendTime = now;
          } else {
            this.liveSignTranslation = "...";
            this.processPrediction("...");
          }
        }
      }
    });

    this.camera = new Camera(videoEl, {
      onFrame: async () => {
        canvasEl.width  = videoEl.videoWidth;
        canvasEl.height = videoEl.videoHeight;
        await holistic.send({ image: videoEl });
      },
      width: 640,
      height: 480
    });
    // Only start camera if we open on the patient tab; otherwise wait for tab switch
    if (this.activeTab === 'patient') this.camera.start();
  }

  // ── 4. Audio recording ────────────────────────────────────
  async startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new (window as any).MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event: any) => {
        if (event.data.size > 0) this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        this.staffText = "Memproses terjemahan AI...";
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.webm");
        try {
          const response = await axios.post(`${this.backendUrl}/api/speech-to-text`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          this.staffText = response.data.simplified_text;
        } catch (error) {
          this.staffText = "Ralat pelayan / API Key tidak dijumpai.";
        }
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.staffText = "Sedang merakam... Cakap sekarang.";
    } catch {
      alert("Sila benarkan akses mikrofon di browser anda.");
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;
    }
  }

  async speakText(text: string) {
    try {
      const response = await axios.post(
        `${this.backendUrl}/api/text-to-speech?text=${encodeURIComponent(text)}`
      );
      if (response.data.status === "success") {
        const audio = new Audio(`data:audio/mp3;base64,${response.data.audio_base64}`);
        audio.play();
      }
    } catch {
      alert("Gagal menjana suara AI.");
    }
  }

  // ── 5. Text-to-Sign dictionary ────────────────────────────
  checkImageExists(url: string): Promise<boolean> {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload  = () => resolve(true);
      img.onerror = () => resolve(false);
      img.src = url;
    });
  }

  async searchSignSentence() {
    const sentence = this.textToSearch.trim();
    if (!sentence) return;
    this.lastSearchedSentence = sentence;
    this.signImageResults = [];
    const words = sentence.toLowerCase().replace(/[.,!?]/g, '').split(/\s+/);
    for (const word of words) {
      const imagePath = `/assets/signs/${word}.jpg`;
      const exists = await this.checkImageExists(imagePath);
      this.signImageResults.push({ word, path: imagePath, found: exists });
    }
    this.textToSearch = "";
  }
}