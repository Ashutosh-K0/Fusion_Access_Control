# ==========================================================
# 🔐 Fusion Access Control – Whisper Enhanced Version (with Phrase Display)
# ==========================================================
import streamlit as st
import cv2
import numpy as np
import librosa
import tempfile
import os
import requests
import whisper
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

TELEGRAM_BOT_TOKEN = "8550965886:AAFf0jyhz4j3j1aO_8nMlW8pqsfpB4OFNho"
TELEGRAM_CHAT_ID = "1636491839"

# ==========================================================
# 2️⃣ Load Models
# ==========================================================
@st.cache_resource
def load_emotion_model():
    return load_model("model.keras")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

emotion_model = load_emotion_model()
whisper_model = load_whisper_model()

labels = ["neutral", "happy", "sad", "fear", "angry", "surprised", "disgust"]
le = LabelEncoder()
le.fit(labels)
EMOJI_MAP = {
    "neutral": "😐", "happy": "🙂", "sad": "😔",
    "fear": "😨", "angry": "😠", "surprised": "😲", "disgust": "🤢"
}

# ==========================================================
# 3️⃣ Helper Functions
# ==========================================================
def send_telegram_alert(summary_text, image_path=None):
    """Send access log and image to Telegram"""
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": summary_text, "parse_mode": "Markdown"}
        )
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as photo:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                    data={"chat_id": TELEGRAM_CHAT_ID},
                    files={"photo": photo}
                )
    except Exception as e:
        print(f"[ERROR] Telegram alert failed: {e}")

def predict_face_emotion(image):
    """Predict facial emotion using CNN model"""
    image = image.convert("L")
    image = image.resize((48, 48))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = emotion_model.predict(img_array)
    label = le.inverse_transform([pred.argmax()])[0]
    conf = np.max(pred) * 100
    emoji = EMOJI_MAP[label]
    return label, conf, emoji

def analyze_voice_whisper(file_path):
    """Transcribe audio using Whisper and analyze tone"""
    # Convert any uploaded format to WAV
    try:
        audio = AudioSegment.from_file(file_path)
        file_path = file_path.replace(".mp3", ".wav")
        audio.export(file_path, format="wav")
    except Exception:
        pass

    # Transcription
    try:
        result = whisper_model.transcribe(file_path)
        text = result["text"].strip().lower()
    except Exception:
        text = ""

    # Tone analysis
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

def phrase_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def fusion_decision(face_emotion, voice_emotion, similarity):
    if similarity >= SIMILARITY_THRESHOLD and (face_emotion in ["happy", "neutral"] and voice_emotion == "calm"):
        return True
    return False

# ==========================================================
# 4️⃣ Streamlit Interface
# ==========================================================
st.title("🔐 Fusion Access Control System")
st.write("Facial Emotion + Whisper-based Voice Recognition + Telegram Alerts")

# ------------------- Face Emotion -------------------------
st.subheader("📷 Upload Your Face Image")
uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

face_emotion, conf, emoji, image_path = None, None, None, None

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("🔍 Analyze Emotion"):
        face_emotion, conf, emoji = predict_face_emotion(image)
        st.success(f"🧠 Detected Emotion: **{face_emotion.upper()} {emoji} ({conf:.2f}%)**")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            image.save(temp_img.name)
            image_path = temp_img.name
        st.session_state["face_emotion"] = face_emotion
        st.session_state["image_path"] = image_path

# ------------------- Voice Recognition ---------------------
st.markdown("---")
st.subheader("🎙️ Upload Your Voice Sample (.wav / .mp3 / .m4a / .ogg)")
uploaded_audio = st.file_uploader("Upload voice file", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(uploaded_audio.read())
        tmp_path = tmp_audio.name

    st.audio(tmp_path)

    if st.button("🔎 Analyze Voice & Grant Access"):
        text, voice_emotion = analyze_voice_whisper(tmp_path)
        similarity = phrase_similarity(text, ACCESS_PHRASE)
        face_emotion = st.session_state.get("face_emotion", "neutral")
        image_path = st.session_state.get("image_path", None)

        # 🧠 Display Detected Phrase Clearly
        if text:
            st.info(f"🗣️ **Detected Phrase:** “{text}”")
        else:
            st.warning("⚠️ No phrase detected. Please ensure clear speech input.")

        st.info(f"🔑 **Access Phrase Match:** {similarity*100:.2f}%")

        decision = fusion_decision(face_emotion, voice_emotion, similarity)

        summary = f"""
**🔐 Fusion Access Log**

🧠 *Face Emotion:* {face_emotion.upper() if face_emotion else 'UNKNOWN'}
🎧 *Voice Emotion:* {voice_emotion.upper()}
🗣️ *Detected Phrase:* {text or 'Unrecognized'}
🔑 *Phrase Match:* {similarity*100:.2f}%
📋 *Final Decision:* {"✅ ACCESS GRANTED" if decision else "🚫 ACCESS DENIED"}
"""
        send_telegram_alert(summary, image_path)

        if decision:
            st.success("✅ ACCESS GRANTED — System Unlocked!")
        else:
            st.error("🚫 ACCESS DENIED — Emotion or Voice mismatch.")

    os.remove(tmp_path)
