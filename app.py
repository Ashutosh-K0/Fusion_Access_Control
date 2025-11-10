# ==========================================================
# 🔐 Fusion Access Control System – Streamlit Version (Final)
# ==========================================================
import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
import os
import requests
import speech_recognition as sr
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from difflib import SequenceMatcher
from PIL import Image
from pydub import AudioSegment
import soundfile as sf

# ==========================================================
# 1️⃣ Setup & Constants
# ==========================================================
st.set_page_config(page_title="Fusion Access Control", page_icon="🔐", layout="centered")

ACCESS_PHRASE = "emotion alpha secure"
SIMILARITY_THRESHOLD = 0.8

# ---- Telegram Setup ----
TELEGRAM_BOT_TOKEN = "8550965886:AAFf0jyhz4j3j1aO_8nMlW8pqsfpB4OFNho"
TELEGRAM_CHAT_ID = "1636491839"

# ---- Load Emotion Model ----
@st.cache_resource
def load_emotion_model():
    model = load_model("model.keras")
    return model

model = load_emotion_model()

labels = ["neutral", "happy", "sad", "fear", "angry", "surprised", "disgust"]
le = LabelEncoder()
le.fit(labels)
EMOJI_MAP = {
    "neutral": "😐", "happy": "🙂", "sad": "😔",
    "fear": "😨", "angry": "😠", "surprised": "😲", "disgust": "🤢"
}

# ==========================================================
# 2️⃣ Helper Functions
# ==========================================================

# ---- Telegram Alerts ----
def send_telegram_alert(summary_text, image_path=None):
    try:
        # send text message
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": summary_text, "parse_mode": "Markdown"}
        requests.post(url, data=data)

        # send image if available
        if image_path and os.path.exists(image_path):
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": photo})
    except Exception:
        pass  # Silently ignore Telegram errors in UI

# ---- Predict Emotion from Uploaded Image ----
def predict_face_emotion(image):
    image = image.convert("L")
    image = image.resize((48, 48))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    label = le.inverse_transform([pred.argmax()])[0]
    conf = np.max(pred) * 100
    emoji = EMOJI_MAP[label]
    return label, conf, emoji

# ---- Safe Audio Analysis ----
def analyze_voice(file_path):
    recognizer = sr.Recognizer()
    text = ""

    # Step 1: Try converting audio safely (ignore ffprobe warning)
    try:
        audio = AudioSegment.from_file(file_path)
        converted_path = file_path.replace(".wav", "_converted.wav")
        audio.export(converted_path, format="wav")
        file_path = converted_path
    except Exception:
        # silently ignore any conversion issue
        pass

    # Step 2: Speech-to-Text
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio).lower()
    except Exception:
        text = ""

    # Step 3: Voice Emotion
    try:
        y, sr_rate = librosa.load(file_path, sr=None)
    except Exception:
        try:
            y, sr_rate = sf.read(file_path)
            if y.ndim > 1:
                y = y.mean(axis=1)
        except Exception:
            return text, "unknown"

    try:
        pitch = librosa.yin(y, 50, 500, sr=sr_rate)
        avg_pitch = np.mean(pitch)
    except Exception:
        avg_pitch = 0

    energy = np.mean(np.abs(y))
    voice_emotion = "calm" if avg_pitch < 200 and energy < 0.05 else "excited"
    return text, voice_emotion

# ---- Phrase Similarity ----
def phrase_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

# ---- Fusion Decision ----
def fusion_decision(face_emotion, voice_emotion, similarity):
    if similarity >= SIMILARITY_THRESHOLD and (face_emotion in ["happy", "neutral"] and voice_emotion == "calm"):
        return True
    return False

# ==========================================================
# 3️⃣ Streamlit Interface
# ==========================================================
st.title("🔐 Fusion Access Control System")
st.write("Facial Emotion + Voice Phrase Verification with Telegram Alerts")

# ---- Image Upload ----
st.subheader("📷 Upload Your Face Image")
uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

face_emotion, conf, emoji = None, None, None
image_path = None

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Emotion"):
        face_emotion, conf, emoji = predict_face_emotion(image)
        st.success(f"🧠 Detected Emotion: **{face_emotion.upper()} {emoji} ({conf:.2f}%)**")

        # Save for Telegram
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            image.save(temp_img.name)
            image_path = temp_img.name

        st.session_state["face_emotion"] = face_emotion
        st.session_state["face_conf"] = conf
        st.session_state["image_path"] = image_path

# ---- Voice Upload ----
st.markdown("---")
st.subheader("🎙️ Upload Your Voice Sample (.wav / .mp3 / .m4a / .ogg)")
uploaded_audio = st.file_uploader("Upload voice file", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(uploaded_audio.read())
        tmp_path = tmp_audio.name

    st.audio(tmp_path)

    if st.button("🔎 Analyze Voice & Grant Access"):
        text, voice_emotion = analyze_voice(tmp_path)
        similarity = phrase_similarity(text, ACCESS_PHRASE)
        face_emotion = st.session_state.get("face_emotion", "neutral")
        image_path = st.session_state.get("image_path", None)

        # ---- Show Recognized Phrase ----
        st.info(f"🗣️ Detected Phrase: `{text or 'Unrecognized'}`")
        st.info(f"🔑 Access Phrase Match: **{similarity*100:.2f}%**")

        decision = fusion_decision(face_emotion, voice_emotion, similarity)

        # ---- Telegram Summary ----
        summary = f"""
**🔐 Fusion Access Log**

🧠 *Face Emotion:* {face_emotion.upper() if face_emotion else 'UNKNOWN'}
🎧 *Voice Emotion:* {voice_emotion.upper()}
🗣️ *Detected Phrase:* {text or 'Unrecognized'}
🔑 *Phrase Match:* {similarity*100:.2f}%
📋 *Final Decision:* {"✅ ACCESS GRANTED" if decision else "🚫 ACCESS DENIED"}
"""
        send_telegram_alert(summary, image_path)

        # ---- Display Result ----
        if decision:
            st.success("✅ ACCESS GRANTED — System Unlocked!")
        else:
            st.error("🚫 ACCESS DENIED — Emotion or Voice mismatch.")

    os.remove(tmp_path)
