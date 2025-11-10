# -*- coding: utf-8 -*-
# ==========================================================
# 🎯 Streamlit: Multimodal Access Control (Image + Voice + Telegram)
# ==========================================================

import streamlit as st
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import requests
import datetime
import tempfile
import speech_recognition as sr
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from difflib import SequenceMatcher
from PIL import Image

# ==========================================================
# 1️⃣ Setup
# ==========================================================
st.set_page_config(page_title="Multimodal Access Control", page_icon="🔐", layout="centered")
st.title("🔐 Multimodal Emotion-Based Access Control System")

# Load model once
@st.cache_resource
def load_emotion_model():
    model = load_model('model.keras')
    labels = ["neutral", "happy", "sad", "fear", "angry", "surprised", "disgust"]
    le = LabelEncoder()
    le.fit(labels)
    return model, le

model, le = load_emotion_model()
EMOJI_MAP = {
    "neutral": "😐", "happy": "🙂", "sad": "😔",
    "fear": "😨", "angry": "😠", "surprised": "😲", "disgust": "🤢"
}

ACCESS_PHRASE = "emotion alpha secure"
SIMILARITY_THRESHOLD = 0.8
SPEECH_DURATION = 5

# Telegram config
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# ==========================================================
# 2️⃣ Helper Functions
# ==========================================================
def send_telegram_alert(status, emotion, similarity, img_path=None):
    """Send Telegram alert with photo."""
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phrase_text = f"🔑 Phrase Match: {float(similarity)*100:.2f}%" if isinstance(similarity, (int, float)) else "🔑 Phrase Match: N/A"
    message = f"{status}\n🕒 {time_str}\n🧠 Emotion: {emotion}\n{phrase_text}"

    try:
        text_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(text_url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

        if img_path:
            photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(img_path, "rb") as photo:
                requests.post(photo_url, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": photo})
        st.success("✅ Telegram Alert Sent!")
    except Exception as e:
        st.error(f"❌ Telegram Alert Failed: {e}")

def predict_emotion_from_image(image):
    img = image.convert("L").resize((48, 48))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    label = le.inverse_transform([pred.argmax()])[0]
    conf = np.max(pred) * 100
    return label, conf

def record_voice(filename="voice_input.wav", duration=SPEECH_DURATION, sr_rate=16000):
    st.info("🎙️ Recording for 5 seconds... Please speak your code phrase clearly.")
    voice_data = sd.rec(int(duration * sr_rate), samplerate=sr_rate, channels=1)
    sd.wait()
    sf.write(filename, voice_data.flatten(), sr_rate)
    return filename

def speech_to_text_and_emotion(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio).lower()
        st.write(f"🗣️ Detected phrase: {text}")
    except Exception:
        st.warning("⚠️ Speech recognition failed.")
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

def phrase_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def fusion_decision(face_emotion, voice_emotion, phrase_sim, img_path=None):
    st.subheader("🧠 Fusion Analysis")
    st.write(f"**Face Emotion:** {face_emotion}")
    st.write(f"**Phrase Similarity:** {phrase_sim*100:.2f}%")

    if phrase_sim >= SIMILARITY_THRESHOLD and (face_emotion in ["happy", "neutral"] and voice_emotion == "calm"):
        st.success("✅ ACCESS GRANTED — Emotion & Voice Confirmed")
        send_telegram_alert("✅ ACCESS GRANTED", face_emotion, phrase_sim, img_path)
        return True
    else:
        st.error("🚫 ACCESS DENIED — Condition mismatch")
        send_telegram_alert("🚫 ACCESS DENIED", face_emotion, phrase_sim, img_path)
        return False

# ==========================================================
# 3️⃣ Streamlit UI
# ==========================================================
st.markdown("---")
st.header("🖼️ Upload Image for Emotion Analysis")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Emotion"):
        label, conf = predict_emotion_from_image(image)
        emoji = EMOJI_MAP[label]
        st.success(f"🧠 Detected Emotion: {label.upper()} {emoji} ({conf:.2f}%)")

        # Save temporarily for Telegram
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_img.name)

        # Voice Recording Section
        st.markdown("---")
        st.header("🎙️ Record Your Voice")
        if st.button("🎤 Start Voice Recording"):
            voice_file = record_voice()
            text, voice_emotion = speech_to_text_and_emotion(voice_file)
            similarity = phrase_similarity(text, ACCESS_PHRASE)
            decision = fusion_decision(label, voice_emotion, similarity, temp_img.name)

            if decision:
                st.balloons()
                st.subheader("🔓 System Unlocked!")
            else:
                st.subheader("🔒 Access Locked.")
