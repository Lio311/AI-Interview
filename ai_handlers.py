import os
import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from utils import get_api_key

# Initialize clients lazily or via a setup function
client = None

def setup_ai_clients():
    global client
    openai_key = get_api_key("OPENAI_API_KEY")
    gemini_key = get_api_key("GEMINI_API_KEY")
    
    if not openai_key:
        st.error("OPENAI_API_KEY missing.")
        st.stop()
    
    client = OpenAI(api_key=openai_key)
    
    if gemini_key:
        genai.configure(api_key=gemini_key)
    else:
        st.warning("GEMINI_API_KEY missing.")

def text_to_speech(text, filename):
    """Generates TTS audio using OpenAI."""
    if not client: setup_ai_clients()
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        path = f"temp_recordings/{filename}"
        response.stream_to_file(path)
        return path
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

def transcribe_audio(audio_path):
    if not client: setup_ai_clients()
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        return f"Error transcribing: {e}"

def generate_interview_questions(cv_text, job_description):
    model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
    prompt = f"""
    You are an expert technical interviewer.
    Based on the following Candidate CV and Job Description, generate 5 distinct, challenging, role-specific interview questions.
    
    CV CONTENT:
    {cv_text[:4000]}
    
    JOB DESCRIPTION:
    {job_description[:4000]}
    
    OUTPUT FORMAT:
    Return ONLY a raw JSON list of strings. Example: ["Question 1", "Question 2"]
    Do not include markdown formatting like ```json.
    """
    try:
        response = model.generate_content(prompt)
        import json
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
        return json.loads(text)
    except Exception as e:
        st.error(f"Gemini Error (Questions): {e}")
        return []

def analyze_answer_with_gemini(transcript, cv_text, job_desc, question, video_path=None):
    model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
    
    # Upload video if provided
    video_file = None
    if video_path and os.path.exists(video_path):
        try:
             video_file = genai.upload_file(video_path)
             import time
             while video_file.state.name == "PROCESSING":
                 time.sleep(1)
                 video_file = genai.get_file(video_file.name)
             if video_file.state.name == "FAILED":
                 video_file = None
        except Exception as e:
            print(f"Video upload failed: {e}")

    prompt = f"""
    Analyze this interview answer.
    
    Question: {question}
    Candidate Answer (Transcript): "{transcript}"
    
    Job Description: {job_desc[:1000]}
    
    Provide a JSON analysis with the following fields:
    - strong_points (list of strings)
    - weak_points (list of strings)
    - score (1-100)
    - is_strong_enough (boolean)
    - delivery_analysis (object):
        - monotonic (bool)
        - low_energy (bool)
        - rushing (bool)
        - feedback (string)
    - body_language (object):
        - eye_contact (string: "Good", "Poor", "Avoidant")
        - posture (string)
        - facial_expressions (string)
        - overall_impression (string)
    
    Return ONLY raw JSON.
    """
    
    inputs = [prompt]
    if video_file:
        inputs.append(video_file)
        inputs.append("Please analyze the visual body language from the video as well.")
        
    try:
        response = model.generate_content(inputs)
        import json
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "")
        return json.loads(text)
    except Exception as e:
        st.error(f"Gemini Analysis Error: {e}")
        return {"error": str(e), "is_strong_enough": False}

def generate_coach_response_with_gpt(analysis_json, transcript, question):
    if not client: setup_ai_clients()
    
    system_prompt = """
    You are a tough but fair senior interviewer.
    I will provide you with the analysis of a candidate's answer (Content + Body Language).
    Your job is to speak DIRECTLY to the candidate.
    
    Rules:
    - Be professional but strict.
    - Mention their body language if it was noted as poor.
    - Keep it under 100 words.
    - Do NOT use emojis.
    """
    
    user_content = f"""
    Question: {question}
    My Answer: {transcript}
    
    AI Analysis Data: {analysis_json}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating feedback: {e}"

def generate_session_report_with_gemini(transcripts, analyses):
    model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
    
    data_str = ""
    for k, v in transcripts.items():
        data_str += f"Q: {k}\nTranscript: {v}\nAnalysis: {analyses.get(k)}\n---\n"
        
    prompt = f"""
    Generate a comprehensive Final Interview Report based on this session data:
    {data_str}
    
    Include:
    1. Executive Summary
    2. Strengths & Weaknesses
    3. Body Language & Tone Analysis (Dedicated Section)
    4. Actionable Improvement Plan
    
    No emojis. Professional formatting.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Report generation error: {e}"
