# 🔐 Fusion Access Control System

**AI-driven multimodal security system** combining facial emotion recognition 🧠 and voice authentication 🎙️ to grant or deny access intelligently.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Model-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Overview
The **Fusion Access Control System** analyzes:
- Facial Emotion (via Deep Learning CNN)
- Spoken Phrase (Speech Recognition)
- Voice Tone (Calm/Excited)
  
Then fuses these parameters to decide whether to grant or deny access.

---

## ⚡ Features
- 🎥 Real-time camera capture (Streamlit camera input)
- 🧠 Facial emotion recognition (TensorFlow model)
- 🎤 5-sec live voice recording & speech analysis
- 🔐 Smart fusion logic for access verification
- 📲 Telegram alerts with image & access decision
- 💾 CSV logging of all access attempts

---

## 🧩 Tech Stack
| Component | Technology |
|------------|-------------|
| Frontend | Streamlit |
| AI/ML | TensorFlow, Keras |
| Speech | SpeechRecognition, Librosa |
| Image | OpenCV, Pillow |
| Alerts | Telegram Bot API |
| Logging | Pandas CSV |

---

## ⚙️ Installation
```bash
git clone https://github.com/Ashutosh-K0/Fusion_Access_Control.git
cd Fusion_Access_Control
pip install -r requirements.txt
```

### Run the App:
```bash
streamlit run app.py
```

---

## 🧠 Usage Guide
1. **Capture Face** – Take photo via webcam.  
2. **Analyze Emotion** – Model predicts emotion.  
3. **Record Voice** – Speak the phrase: `emotion alpha secure`.  
4. **Fusion Decision** – System grants or denies access.  
5. **Alert & Log** – Telegram message + CSV entry generated.

---

## 📲 Telegram Setup
1. Create a bot with [@BotFather](https://t.me/BotFather).  
2. Get your **chat ID** from [@userinfobot](https://t.me/userinfobot).  
3. Add these as environment variables:
   ```
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   ```

---

## 🧾 Access Log Example
| Timestamp | Emotion | Phrase | Match% | Decision |
|------------|----------|---------|---------|-----------|
| 2025-11-10 20:30:44 | happy | emotion alpha secure | 97.8 | Access Granted |

---

## 📚 Example Outcomes
**✅ Access Granted:**  
Emotion = Happy, Voice = Calm, Match ≥ 80%

**🚫 Access Denied:**  
Any mismatch in emotion, tone, or phrase.

---

## 🧱 Requirements
```
streamlit
tensorflow
opencv-python-headless
scikit-learn
numpy
pillow
librosa
soundfile
sounddevice
SpeechRecognition
requests
pydub
ffmpeg-python
pandas
```
---

## 🧩 Future Enhancements
- Integrate face recognition (identity + emotion)
- Cloud-based access dashboard
- Whisper or on-device speech model integration

---

## 🏁 License
Released under the **MIT License**. 
