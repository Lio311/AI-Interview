# 🎥 AI Video Interview Coach

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-interview-practice.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Gemini](https://img.shields.io/badge/Google%20AI-Gemini%201.5-orange)](https://deepmind.google/technologies/gemini/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green)](https://openai.com/)

An advanced **Hybrid AI** interview simulator that combines **Google Gemini** (for deep multimodal analysis) and **OpenAI GPT-4o** (for professional conversational roleplay).

This application doesn't just "listen" to your answers—it **watches** you, analyzing your body language, eye contact, and vocal energy to provide "tough but fair" executive-level feedback.

---

## 🚀 Quick Start

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ai-interview-coach.git
   cd ai-interview-coach
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires `ffmpeg` installed on your system.*

3. **Set up API Keys**
   Create a `.env` file or export variables:
   ```bash
   export OPENAI_API_KEY="sk-..."
   export GEMINI_API_KEY="AIza..."
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

---

## 🌟 Key Features

### 🧠 Hybrid AI Architecture
The system leverages the best of both worlds:
- **Google Gemini 1.5 Flash**: Acts as the **Analytical Engine**. It parses your CV, generates technical questions, and performs deep multimodal analysis of your video and audio.
- **OpenAI GPT-4o**: Acts as the **Interviewer Persona**. It takes the raw data from Gemini and delivers it in a human-like, professional, conversational tone.

### 👁️ Multimodal Body Language Analysis
Unlike standard voice assistants, this coach **sees** you via your webcam:
- **Eye Contact**: Detects if you are looking at the camera or avoiding gaze.
- **Posture & Gestures**: Identifies slouching, fidgeting, or distracting hand movements.
- **Facial Expressions**: Analyzes smiles, stress cues, and engagement levels.

### 📊 Full Session Artifacts
- **Concatenated Video**: Automatically merges your entire session into a single MP4 file.
- **Detailed Report**: Generates a comprehensive text report with scores, strengths, and actionable drills.

---

## 📁 Project Structure

```
ai_interview_coach/
├── app.py                  # Main Application Entry Point
├── ai_handlers.py          # AI Logic (Gemini/OpenAI)
├── media_handlers.py       # Audio/Video Processing
├── utils.py                # Helper Functions
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (ffmpeg for cloud)
├── .gitignore              # Security rules
└── README.md               # Project documentation
```

---

## 🎯 How It Works

1.  **Setup Phase**:
    *   Upload your **CV (PDF)**.
    *   Paste the **Job Description**.
    *   The AI generates 5-7 tailored technical questions.

2.  **Interview Phase**:
    *   **Auto-Connect**: Camera connects automatically.
    *   **Auto-Flow**: The AI reads the question, and you record your answer.
    *   **Real-time Analysis**: 
        *   Gemini checks your content (STAR method), delivery (Monotony/Energy), and Body Language.
        *   GPT-4o provides immediate verbal feedback.

3.  **Review Phase**:
    *   Receive a final **Performance Report**.
    *   Download the **Full Session Video**.

---

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Analysis**: Google Gemini 1.5 Flash (Multimodal)
- **Conversation**: OpenAI GPT-4o
- **Video Processing**: Streamlit WebRTC, MoviePy
- **Audio Processing**: OpenAI Whisper (STT)

---

## 🚢 Deployment (Streamlit Cloud)

This app is optimized for **Streamlit Community Cloud**.

1.  **Push to GitHub**:
    ```bash
    git add .
    git commit -m "Initial commit"
    git push origin main
    ```

2.  **Deploy**:
    *   Go to [share.streamlit.io](https://ai-interview-practice.streamlit.app/).
    *   Select your repository.

3.  **Configure Secrets (CRITICAL)**:
    *   In the deployed app settings, go to **Advanced Settings** -> **Secrets**.
    *   Add your keys:
        ```toml
        OPENAI_API_KEY = "sk-..."
        GEMINI_API_KEY = "AIza..."
        ```
    *   *Streamlit Cloud will automatically handle the system dependencies via `packages.txt`.*

---

## 🔄 Updates

### v2.0 - Hybrid Architecture
- **Refactor**: Split responsibilities between Gemini (Analysis) and OpenAI (Talk).
- **Vision**: Added video file upload for body language analysis.
- **Security**: Added support for Streamlit Secrets.
- **UI**: Added "Laptop Friendly" mode (Auto-Cam, No Emojis, Auto-Start).

<div align="center">

**Current Model: Gemini 1.5 Flash + GPT-4o**

</div>
