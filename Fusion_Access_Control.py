# -*- coding: utf-8 -*-
# ==========================================================
# 🎯 Multimodal Access Control: Emotion + Voice Fusion System
# ==========================================================

import cv2
import numpy as np
import os, datetime
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import librosa
import requests
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from difflib import SequenceMatcher
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

# ==========================================================
# 1️⃣ Model & Constants
# ==========================================================
print("\n[INFO] Loading emotion model...")
model = load_model('model.keras')

labels = ["neutral", "happy", "sad", "fear", "angry", "surprised", "disgust"]
le = LabelEncoder()
le.fit(labels)
EMOJI_MAP = {
    "neutral": "😐", "happy": "🙂", "sad": "😔",
    "fear": "😨", "angry": "😠", "surprised": "😲", "disgust": "🤢"
}

ACCESS_PHRASE = "emotion alpha secure"
SIMILARITY_THRESHOLD = 0.8
SPEECH_DURATION = 5  # seconds

# ==========================================================
# 🔔 Telegram Bot Configuration
# ==========================================================
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

def send_telegram_alert(status, emotion, similarity, img_path=None):
    """Send alert + optional image to Telegram."""
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phrase_text = f"🔑 Phrase Match: {float(similarity)*100:.2f}%" if isinstance(similarity, (int, float)) else "🔑 Phrase Match: N/A"

    message = (
        f"{status}\n"
        f"🕒 Time: {time_str}\n"
        f"🧠 Emotion: {emotion}\n"
        f"{phrase_text}"
    )

    try:
        # Send text message
        text_url = f"https://api.telegram.org/bot8550965886:AAFf0jyhz4j3j1aO_8nMlW8pqsfpB4OFNho/sendMessage"
        requests.post(text_url, data={"chat_id": 1636491839, "text": message})

        # Send image (if available)
        if img_path and os.path.exists(img_path):
            photo_url = f"https://api.telegram.org/bot8550965886:AAFf0jyhz4j3j1aO_8nMlW8pqsfpB4OFNho/sendPhoto"
            with open(img_path, "rb") as photo:
                requests.post(photo_url, data={"chat_id": 1636491839}, files={"photo": photo})
        print("[ALERT] Telegram notification sent successfully.")
    except Exception as e:
        print(f"[ERROR] Telegram alert failed: {e}")


# ==========================================================
# 2️⃣ Helper Functions
# ==========================================================

# ----------------------------------------------------------
# 🖼️ Select Image from Local System
# ----------------------------------------------------------
def select_local_image():
    print("\n🖼️ Please choose an image file for emotion analysis.")
    Tk().withdraw()  # hide tkinter root window
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        print("❌ No image selected.")
        return None
    print(f"✅ Image selected: {file_path}")
    return file_path


# ----------------------------------------------------------
# 🤖 Predict Emotion from Image
# ----------------------------------------------------------
def predict_emotion_from_image(img_path):
    img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    label = le.inverse_transform([pred.argmax()])[0]
    conf = np.max(pred) * 100
    emoji = EMOJI_MAP[label]

    # Display analyzed image using matplotlib
    disp = cv2.imread(img_path)
    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    cv2.putText(disp, f"{label.upper()} {emoji} ({conf:.1f}%)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    plt.imshow(disp)
    plt.axis('off')
    plt.title(f"{label.upper()} {emoji} ({conf:.1f}%)")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    print(f"🧠 Detected Emotion: {label.upper()} {emoji} ({conf:.2f}%)")
    return label, conf


# ----------------------------------------------------------
# 🎙️ Record Voice
# ----------------------------------------------------------
def record_voice(filename="voice_input.wav", duration=SPEECH_DURATION, sr_rate=16000):
    print("\n🔈 Press 'R' to start voice recording or 'Q' to cancel.")
    while True:
        key = input("➡️  Press R to record or Q to quit: ").strip().lower()
        if key == 'r':
            print(f"🎙️ Recording voice for {duration} seconds... Speak your code phrase now.")
            voice_data = sd.rec(int(duration * sr_rate), samplerate=sr_rate, channels=1)
            sd.wait()
            sf.write(filename, voice_data.flatten(), sr_rate)
            print(f"✅ Voice recorded and saved as {filename}")
            return filename
        elif key == 'q':
            print("🚪 Voice capture cancelled.")
            return None
        else:
            print("⚠️ Invalid input — press 'R' to record or 'Q' to quit.")


# ----------------------------------------------------------
# 🗣 Speech-to-Text + Hidden Voice Emotion Estimation
# ----------------------------------------------------------
def speech_to_text_and_emotion(filename):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(filename)
    with audio_file as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"🗣️ Detected phrase: {text}")
    except Exception as e:
        print("⚠️ Speech recognition failed:", e)
        text = ""

    # Hidden tone analysis
    y, sr_rate = librosa.load(filename, sr=None)
    try:
        pitch = librosa.yin(y, 50, 500, sr=sr_rate)
        avg_pitch = np.mean(pitch)
    except Exception:
        avg_pitch = 0
    energy = np.mean(np.abs(y))
    voice_emotion = "calm" if avg_pitch < 200 and energy < 0.05 else "excited"

    return text, voice_emotion


# ----------------------------------------------------------
# 🔑 Phrase Similarity
# ----------------------------------------------------------
def phrase_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


# ----------------------------------------------------------
# 🧩 Fusion Logic + Telegram Alert
# ----------------------------------------------------------
def fusion_decision(face_emotion, voice_emotion, phrase_sim, img_path=None):
    print("\n🧠 Fusion Analysis:")
    print(f"Face Emotion: {face_emotion}")
    print(f"Phrase Similarity: {phrase_sim*100:.2f}%")

    if phrase_sim >= SIMILARITY_THRESHOLD and (face_emotion in ["happy", "neutral"] and voice_emotion == "calm"):
        print("\n✅ ACCESS GRANTED — Emotion & Voice Confirmed")
        send_telegram_alert("✅ ACCESS GRANTED", face_emotion, phrase_sim, img_path)
        return True
    else:
        print("\n🚫 ACCESS DENIED — Condition mismatch")
        send_telegram_alert("🚫 ACCESS DENIED", face_emotion, phrase_sim, img_path)
        return False


# ==========================================================
# 3️⃣ Main Program
# ==========================================================
if __name__ == "__main__":
    print("[INFO] Starting Multimodal Access System...")

    # Step 1️⃣ Choose Image
    img_path = select_local_image()
    if not img_path:
        print("❌ No image selected — exiting.")
        exit()

    # Step 2️⃣ Predict Emotion
    face_emotion, conf = predict_emotion_from_image(img_path)

    # Step 3️⃣ Voice Recording
    voice_file = record_voice()
    if voice_file is None:
        print("❌ No voice captured — exiting.")
        exit()

    # Step 4️⃣ Analyze Voice & Phrase
    text, voice_emotion = speech_to_text_and_emotion(voice_file)
    similarity = phrase_similarity(text, ACCESS_PHRASE)

    # Step 5️⃣ Fusion Decision & Alert
    decision = fusion_decision(face_emotion, voice_emotion, similarity, img_path)

    if decision:
        print("🔓 System Unlocked!")
    else:
        print("🔒 Access Locked.")
