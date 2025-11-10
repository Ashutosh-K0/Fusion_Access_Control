# -*- coding: utf-8 -*-
# ==========================================================
# 🔐 Fusion Access Control – Streamlit Cloud Stable Version
# ==========================================================

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import requests
import tempfile
import datetime
from difflib import SequenceMatcher
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# ==========================================================
# 1️⃣ Setup & Constants
# ==========================================================
st.set_page_config(page_title="Fusion Access Control", page_icon="🔐", layout="centered")
st.title("🔐 Multimodal Emotion + Voice Access Control")

@st.cache_resource
def load_emotion_model():
    model = load_model("model.keras")
    labels = ["neutral", "happy", "sad", "fear", "angry", "surprised", "disgust"]
    le = LabelEncoder()
    le.fit(labels)
    return model, le

model, le = load_emotion_model()

EMOJI_MAP = {
    "neutral": "😐",
    "happy": "🙂",
    "sad": "😔",
    "fear": "😨",
    "angry": "😠",
    "surprised": "😲",
    "disgust": "🤢",
}

ACCESS_PHRASE = "emotion alpha secure"
SIMILARITY_THRESHOLD = 0.8
TELEGRAM_BOT_TOKEN = "8550965886:AAFf0jyhz4j3j1aO_8nMlW8pqsfpB4OFNho"
TELEGRAM_CHAT_ID = "1636491839"

# ==========================================================
# 2️⃣ Helper Functions
# ==========================================================
def send_telegram_alert(status, emotion, similarity, img_path=None):
    """Send alert + optional image to Telegram."""
    time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"{status}\n🕒 {time_str}\n🧠 Emotion: {emotion}\n🔑 Phrase Match: {similarity*100:.2f}%"
    try:
        text_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(text_url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})
        if img_path:
            photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(img_path, "rb") as photo:
                requests.post(photo_url, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": photo})
    except Exception as e:
        st.error(f"⚠️ Telegram alert failed: {e}")


def predict_emotion_from_image(image):
    img = image.convert("L").resize((48, 48))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    label = le.inverse_transform([pred.argmax()])[0]
    conf = np.max(pred) * 100
    return label, conf


def phrase_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()


def speech_to_text_and_emotion(audio_path):
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio).lower()
        st.write(f"🗣️ Detected phrase: {text}")
    except Exception:
        st.warning("⚠️ Speech recognition failed.")
        text = ""

    # Basic hidden tone analysis
    y, sr_rate = librosa.load(audio_path, sr=None)
    try:
        pitch = librosa.yin(y, 50, 500, sr=sr_rate)
        avg_pitch = np.mean(pitch)
    except Exception:
        avg_pitch = 0
    energy = np.mean(np.abs(y))
    voice_emotion = "calm" if avg_pitch < 200 and energy < 0.05 else "excited"
    return text, voice_emotion


def fusion_decision(face_emotion, voice_emotion, phrase_sim, img_path=None):
    st.subheader("🧠 Fusion Analysis")
    st.write(f"**Face Emotion:** {face_emotion}")
    st.write(f"**Phrase Similarity:** {phrase_sim*100:.2f}%")

    if phrase_sim >= SIMILARITY_THRESHOLD and (
        face_emotion in ["happy", "neutral"] and voice_emotion == "calm"
    ):
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
st.header("🖼️ Upload Image for Emotion Analysis")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Emotion"):
        label, conf = predict_emotion_from_image(image)
        emoji = EMOJI_MAP[label]
        st.success(f"🧠 Detected Emotion: {label.upper()} {emoji} ({conf:.2f}%)")

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        image.save(temp_img.name)

        st.markdown("---")
        st.header("🎙️ Upload Your Voice Sample (.wav)")

        uploaded_audio = st.file_uploader("Upload audio", type=["wav"])
        if uploaded_audio:
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio.write(uploaded_audio.read())

            st.audio(temp_audio.name, format="audio/wav")

            text, voice_emotion = speech_to_text_and_emotion(temp_audio.name)
            similarity = phrase_similarity(text, ACCESS_PHRASE)
            decision = fusion_decision(label, voice_emotion, similarity, temp_img.name)

            if decision:
                st.balloons()
                st.subheader("🔓 System Unlocked!")
            else:
                st.subheader("🔒 Access Locked.")
