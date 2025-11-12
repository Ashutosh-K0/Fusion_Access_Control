# ==============================================================
# 🔐 Multimodal Access Control System — Stable Local Version
# ==============================================================

import streamlit as st
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from difflib import SequenceMatcher
import speech_recognition as sr
import librosa
import requests
from PIL import Image
from datetime import datetime

# ==============================================================
# 1️⃣ Setup & Constants
# ==============================================================

st.set_page_config(page_title="Fusion Access Control", page_icon="🔐", layout="wide")
st.title("🔐 Fusion Access Control System")
st.markdown("Combining **Facial Emotion** 🧠 and **Voice Phrase** 🎙️ for secure access verification.")

# --------------------------------------------------------------
# Telegram Configuration (set via environment or inline)
# --------------------------------------------------------------
BOT_TOKEN = os.getenv("8550965886:AAFf0jyhz4j3j1aO_8nMlW8pqsfpB4OFNho") or "8550965886:AAFf0jyhz4j3j1aO_8nMlW8pqsfpB4OFNho"
CHAT_ID = os.getenv("1636491839") or "1636491839"

# --------------------------------------------------------------
# Model & Constants
# --------------------------------------------------------------
@st.cache_resource
def load_emotion_model():
    return load_model("model.keras")

model = load_emotion_model()

labels = ["neutral", "happy", "sad", "fear", "angry", "surprised", "disgust"]
le = LabelEncoder()
le.fit(labels)
EMOJI_MAP = {
    "neutral": "😐", "happy": "🙂", "sad": "😔",
    "fear": "😨", "angry": "😠", "surprised": "😲", "disgust": "🤢"
}

ACCESS_PHRASE = "emotion alpha secure"
SIMILARITY_THRESHOLD = 0.8
LOG_FILE = "access_log.csv"

# ==============================================================
# 2️⃣ Helper Functions
# ==============================================================

def send_telegram_alert(message, image_path=None):
    """Send alert to Telegram with optional image"""
    try:
        if BOT_TOKEN and CHAT_ID:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                data={"chat_id": CHAT_ID, "text": message},
            )
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as photo:
                    requests.post(
                        f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                        data={"chat_id": CHAT_ID},
                        files={"photo": photo},
                    )
    except Exception as e:
        st.warning(f"⚠️ Telegram alert failed: {e}")

def record_voice(duration=5, sr_rate=16000):
    """Record voice for a set duration"""
    st.info(f"🎙️ Recording for {duration} seconds... Speak your code phrase now!")
    voice_data = sd.rec(int(duration * sr_rate), samplerate=sr_rate, channels=1)
    sd.wait()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, voice_data.flatten(), sr_rate)
    st.success("✅ Voice recorded successfully!")
    return temp_file.name

def phrase_similarity(text1, text2):
    """Compute similarity between two phrases"""
    return SequenceMatcher(None, text1, text2).ratio()

def analyze_voice(file_path):
    """Analyze voice emotion + speech-to-text"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as src:
            audio = recognizer.record(src)
            text = recognizer.recognize_google(audio).lower()
            st.info(f"🗣️ Detected Phrase: **{text}**")
    except Exception as e:
        st.warning(f"⚠️ Speech recognition failed: {e}")
        text = ""

    try:
        y, sr_rate = librosa.load(file_path, sr=None)
        energy = np.mean(np.abs(y))
        avg_pitch = np.mean(librosa.yin(y, 50, 500, sr=sr_rate))
        # 🔧 Adjusted thresholds for normal human speech
        voice_emotion = "calm" if avg_pitch < 250 and energy < 0.08 else "excited"
    except Exception:
        voice_emotion = "unknown"

    st.info(f"🎧 Voice Emotion Estimated: **{voice_emotion.upper()}**")
    return text, voice_emotion


def log_access(emotion, phrase, similarity, decision):
    """Append access attempt to CSV log"""
    data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Emotion": [emotion],
        "Phrase": [phrase],
        "Match%": [round(similarity * 100, 2)],
        "AccessDecision": [decision],
    }
    df = pd.DataFrame(data)
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

# ==============================================================
# 3️⃣ Facial Emotion Detection (via Image Upload)
# ==============================================================

st.subheader("📸 Step 1: Upload Your Face Image")

# Initialize session variables
if "face_emotion" not in st.session_state:
    st.session_state["face_emotion"] = None
if "face_image_path" not in st.session_state:
    st.session_state["face_image_path"] = None

uploaded_image = st.file_uploader("📁 Upload an image file", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="🖼 Uploaded Face Image", width=300)

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48))
    img = img_to_array(resized) / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Analyze Emotion"):
        preds = model.predict(img)
        label = le.inverse_transform([preds.argmax()])[0]
        conf = np.max(preds) * 100
        emoji = EMOJI_MAP[label]
        st.success(f"✅ Emotion: **{label.upper()} {emoji} ({conf:.2f}%)**")

        st.session_state["face_emotion"] = label
        path = "uploaded_face.jpg"
        image.save(path)
        st.session_state["face_image_path"] = path

# ==============================================================
# 4️⃣ Voice Recording & Analysis
# ==============================================================

st.markdown("---")
st.subheader("🎙️ Step 2: Record Your Voice")

if st.button("🎤 Start 5-second Recording"):
    voice_file = record_voice()
    text, voice_emotion = analyze_voice(voice_file)
    similarity = phrase_similarity(text, ACCESS_PHRASE)
    st.info(f"🗝️ Access Phrase Match: **{similarity*100:.2f}%**")

    # ==========================================================
    # 🧠 Fusion Logic
    # ==========================================================
    # ==========================================================
    # 🧠 Fusion Logic
    # ==========================================================
    face_emotion = st.session_state.get("face_emotion")
    img_path = st.session_state.get("face_image_path")
    access_granted = (
        face_emotion
        and similarity >= SIMILARITY_THRESHOLD
        and voice_emotion in ["calm", "unknown"]  # ✅ Now accepts 'unknown'
        and face_emotion in ["happy", "neutral"]
    )

    # ==========================================================
    # 🎯 Final Summary Card
    # ==========================================================
    st.markdown("---")
    st.markdown("### 🧾 Final Access Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Emotion", f"{face_emotion.upper() if face_emotion else 'N/A'} {EMOJI_MAP.get(face_emotion, '')}")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Phrase", text if text else "Unrecognized")
    with col4:
        st.metric("Match %", f"{similarity*100:.2f}")

    if access_granted:
        decision = "Access Granted"
        st.success("✅ ACCESS GRANTED — Emotion & Voice Verified!")
    else:
        decision = "Access Denied"
        st.error("🚫 ACCESS DENIED — Emotion or Voice mismatch.")

    # ==========================================================
    # 🔔 Telegram + CSV Logging
    # ==========================================================
    send_telegram_alert(
        f"{'✅' if access_granted else '🚫'} {decision}\n"
        f"Emotion: {face_emotion}\nVoice: {voice_emotion}\n"
        f"Phrase: {text}\nMatch: {similarity*100:.2f}%",
        img_path,
    )
    log_access(face_emotion, text, similarity, decision)

# ==============================================================
# 5️⃣ Conditional Reset / Clear Session Button
# ==============================================================

if st.session_state.get("face_emotion") or any(k.startswith("voice") for k in st.session_state.keys()):
    st.markdown("---")
    if st.button("🔄 Reset / Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("🔁 Session cleared! You can start fresh now.")
        st.rerun()