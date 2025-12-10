# AI Video Interview Coach 🎥

A "Tough but Fair" AI Interview Coach that uses **Google Gemini** for deep analysis (CV, Audio, Video, Body Language) and **OpenAI GPT-4o** for the conversational persona.

## Features
- **Hybrid AI Architecture**: 
  - **Gemini 2.5 Flash**: Analyzing CVs, Video Body Language (Eye Contact, Posture), and Vocal Tone.
  - **OpenAI GPT-4o**: Acts as the conversational interviewer.
- **Real-time Video Recording**: Record answers directly in the browser.
- **Multimodal Feedback**: Gets feedback on *what* you said and *how* you looked saying it.
- **Final Report**: Generates a detailed PDF-ready summary and a concatenated video of the session.

## Installation (Local)

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You also need `ffmpeg` installed on your system.*

3.  Set up environment variables in a `.env` file:
    ```ini
    OPENAI_API_KEY=sk-...
    GEMINI_API_KEY=AIza...
    ```

4.  Run the app:
    ```bash
    streamlit run app.py
    ```

## Deployment to Streamlit Community Cloud (Recommended)

This app is optimized for Streamlit Cloud.

1.  Push this code to a **GitHub Repository**.
2.  Go to [share.streamlit.io](https://share.streamlit.io/).
3.  Click **New App** and select your repository.
4.  **CRITICAL**: Before deploying, click **Advanced Settings** -> **Secrets**.
5.  Add your API keys in TOML format:
    ```toml
    OPENAI_API_KEY = "sk-..."
    GEMINI_API_KEY = "AIza..."
    ```
6.  Click **Deploy**.

## Requirements
- Access to camera/microphone.
- Modern browser (Chrome/Edge recommended).
