
import streamlit as st
import os
import time
import threading
import queue
import wave
import numpy as np
import subprocess
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI
import google.generativeai as genai
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips

# Load environment variables
load_dotenv()

# Initialize API Clients
def get_api_key(key_name):
    """Try to get API key from Streamlit secrets, then environment variables."""
    if key_name in st.secrets:
        return st.secrets[key_name]
    return os.getenv(key_name)

openai_api_key = get_api_key("OPENAI_API_KEY")
gemini_api_key = get_api_key("GEMINI_API_KEY")

if not openai_api_key:
    st.error("OPENAI_API_KEY not found in Streamlit secrets or environment variables.")
    st.stop()
    
if not gemini_api_key:
    st.warning("GEMINI_API_KEY not found. Please set it in secrets or .env.")
else:
    genai.configure(api_key=gemini_api_key)

client = OpenAI(api_key=openai_api_key)

# Ensure temp directory exists
os.makedirs("temp_recordings", exist_ok=True)

# Core Questions List
CORE_QUESTIONS = [
    "Tell me about yourself.",
    "Why do you want to leave your current job?",
    "Why do you want to work here?",
    "What is your greatest weakness?",
    "What is your greatest professional achievement?",
    "Tell me about a conflict you had at work.",
    "What kind of work environment do you prefer?",
    "Where do you see yourself in 5 years?",
    "How do you handle pressure?",
    "Are you a team player?",
    "What are your salary expectations?",
    "Do you have any questions for us?"
]

# Page configuration
st.set_page_config(
    page_title="AI Video Interview Coach",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'cv_text' not in st.session_state:
    st.session_state.cv_text = ""
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'questions_list' not in st.session_state:
    st.session_state.questions_list = []
if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0
if 'interview_phase' not in st.session_state:
    st.session_state.interview_phase = "setup" # setup, countdown, recording, interviewing, feedback, finished
if 'transcripts' not in st.session_state:
    st.session_state.transcripts = {} # Key: question_index, Value: transcript text
if 'feedback' not in st.session_state:
    st.session_state.feedback = {} # Key: question_index, Value: feedback text
if 'recordings' not in st.session_state:
    st.session_state.recordings = {} # Key: question_index, Value: path to video file
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = None
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {} # Store raw gemini analysis

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def generate_interview_questions(cv_text, job_description):
    """Generates role-specific questions using Gemini."""
    
    # Use Gemini for Analysis/Reasoning
    model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
    
    prompt = f"""
    You are an expert technical interviewer. 
    Analyze the following Candidate CV and Job Description.
    Generate a list of 10 highly relevant, role-specific interview questions.
    
    The questions should:
    - Assess technical skills mentioned in the JD.
    - Test behavioral traits relevant to the role.
    - Be challenging but fair.
    - NOT repeat standard generic questions.
    
    **Job Description:**
    {job_description[:4000]}
    
    **Candidate CV:**
    {cv_text[:4000]}
    
    Return ONLY a valid JSON list of strings. 
    Example: ["Question 1?", "Question 2?", ...]
    """
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        
        # Clean up markdown code blocks if present
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        elif content.startswith("```"):
            content = content.replace("```", "")
            
        custom_questions = json.loads(content)
        return custom_questions  
    
    except Exception as e:
        st.error(f"Error generating questions with Gemini: {e}")
        return []

def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return "(Transcription failed)"

def analyze_answer_with_gemini(transcript, cv_text, job_desc, question, video_path, history=None):
    """
    Step 1: Gemini analyzes the VIDEO and Transcript.
    """
    model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
    
    # 1. Upload Video to Gemini
    # We need to wait for processing? usually small clips are fast.
    video_file = None
    if video_path and os.path.exists(video_path):
        try:
            print(f"Uploading {video_path} to Gemini...")
            video_file = genai.upload_file(video_path, mime_type="video/mp4")
            
            # Wait for processing (polling)
            while video_file.state.name == "PROCESSING":
                print("Processing video...")
                time.sleep(1)
                video_file = genai.get_file(video_file.name)
                
            if video_file.state.name == "FAILED":
                raise ValueError("Video processing failed.")
        except Exception as e:
            st.error(f"Video Upload Error: {e}")
            video_file = None
            
    hist_ctx = ""
    if history:
        hist_ctx = f"Previous feedback history: {history}"
        
    prompt = f"""
    You are a strict but fair senior hiring manager and expert body language coach.
    
    Task: Evaluate the candidate's interview answer based on the VIDEO and Transcript.
    
    Context:
    - Job Description: {job_desc[:2000]}...
    - Question: "{question}"
    - Previous Context: {hist_ctx}
    - Transcript: "{transcript}"
    
    **MANDATORY ANALYSIS SECTIONS**:
    
    1. **CONTENT & SPEECH ANALYSIS** (Based on Audio/Transcript):
       - Answer Quality (STAR method, relevance).
       - Monotonic Speech (Flat vs Dynamic).
       - Low Energy / Low Engagement (Boring vs Enthusiastic).
       - Confidence in Speech (Weak words like "maybe", "I think").
    
    2. **VIDEO & BODY LANGUAGE ANALYSIS** (Based on Video):
       - **Eye Contact**: Look for consistent eye contact with camera vs looking away/down.
       - **Facial Expressiveness**: Neutral/Flat vs Expressive? Smiles vs Tension?
       - **Posture**: Slouched vs Upright? Swaying/Rocking?
       - **Hand Gestures**: Hidden? Distracting? Natural?
       - **Visual Energy**: Does the candidate look bored or excited?
       - **Stress Cues**: Fidgeting, touching face, hair, tight jaw.
    
    Provide a structured JSON output:
    {{
        "summary": "Brief summary",
        "is_strong_enough": boolean,
        "content_score": 1-10,
        "content_feedback": {{
             "strengths": ["list"],
             "improvements": ["list"],
             "speech_delivery": {{
                 "monotonic": boolean,
                 "low_energy": boolean,
                 "confidence_issues": ["list"]
             }}
        }},
        "body_language": {{
             "eye_contact": {{ "score": 1-10, "observation": "...", "tips": "..." }},
             "facial_expression": {{ "score": 1-10, "observation": "...", "tips": "..." }},
             "posture": {{ "score": 1-10, "observation": "...", "tips": "..." }},
             "gestures": {{ "score": 1-10, "observation": "...", "tips": "..." }},
             "energy_visual": {{ "score": 1-10, "observation": "...", "tips": "..." }},
             "confidence_signals": "Summary of visual confidence",
             "stress_indicators": "Summary of stress cues",
             "overall_impression": "Paragraph summary of body language"
        }},
        "pronunciation_notes": "...",
        "coaching_tips": "..."
    }}
    """
    
    try:
        inputs = [prompt]
        if video_file:
            inputs.append(video_file)
            
        response = model.generate_content(inputs)
        content = response.text.strip()
        
        # Cleanup
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        elif content.startswith("```"):
            content = content.replace("```", "")
            
        return json.loads(content)
    except Exception as e:
        st.error(f"Gemini Analysis Error: {e}")
        return {
            "summary": "Analysis failed.",
            "is_strong_enough": False,
            "content_feedback": {"improvements": [f"Error: {e}"]},
            "body_language": {}
        }

def generate_coach_response_with_gpt(analysis, transcript, question):
    """
    Step 2: ChatGPT speaks to the user.
    """
    
    system_prompt = f"""
    You are a professional, strict but fair Executive Interview Coach.
    Deliver feedback based on the analysis.
    
    Analysis Data:
    {json.dumps(analysis, indent=2)}
    
    Instructions:
    1. Be conversational and direct ("You said...", "I noticed...").
    2. **Content**: Briefly mention if the answer was good or needs work.
    3. **Body Language (CRITICAL)**: You MUST comment on their video presence using the 'body_language' data.
       - If Eye Contact score < 7: "You need to look at the camera more."
       - If Posture score < 7: "Sit up straight, you looked a bit slouched."
       - If Stress detected: "I saw you touching your face/fidgeting. Try to relax."
    4. **Delivery**: Mention if they sounded robotic or low energy.
    5. Final verdict: Move on or Retry.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error responding: {e}"

def generate_session_report_with_gemini(transcripts, analysis_history):
    """Generates the final report."""
    model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
    
    # Construct a summary of the session
    session_data = ""
    for key in transcripts:
        transcript = transcripts[key]
        analysis = analysis_history.get(key, {})
        session_data += f"Q/A Key: {key}\nTranscript: {transcript}\nAnalysis: {analysis}\n\n"
        
    prompt = f"""
    Generate a detailed Interview Session Report.
    
    Session Data:
    {session_data[:20000]}
    
    Report Structure:
    1. **Executive Summary**: Overall readiness.
    2. **Content & Storytelling**: STAR method usage, relevance.
    3. **Vocal Delivery Analysis**:
       - Monotony, Energy, Confidence phrases.
       - Grammar/Pronunciation.
    4. **Body Language Summary & Improvement Plan** (Detailed):
       - Eye Contact performance.
       - Facial Expressions & Smile.
       - Posture & Gestures.
       - Stress indicators observed.
       - **Actionable Drills** for improvement.
    5. **Question-by-Question Breakdown** (Brief).
    6. **Final Verdict**.
    
    Format: Professional Markdown Report.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating report: {e}"

# --- Helper Functions (Video/Audio) ---
# ... (Keep existing RecorderManager and merge_audio_video) ...

class RecorderManager:
    def __init__(self):
        self.video_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.recording = False
        self.output_video_path = None
        self.output_audio_path = None
        self.worker_thread = None
        self.final_path = None
        
    def start_recording(self, video_path, audio_path, final_path):
        self.video_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.output_video_path = video_path
        self.output_audio_path = audio_path
        self.final_path = final_path
        self.recording = True
        
        self.worker_thread = threading.Thread(target=self._process_queues)
        self.worker_thread.start()
        
    def stop_recording(self):
        self.recording = False
        if self.worker_thread:
            self.worker_thread.join()
        
    def _process_queues(self):
        import av # Import av here as it's used locally
        video_container = None
        video_stream = None
        audio_file = None
        CHANNELS = 2
        RATE = 48000
        
        while True:
            if not self.recording and self.video_queue.empty() and self.audio_queue.empty():
                break
            try:
                frame = self.video_queue.get(timeout=0.01)
                if video_container is None:
                    video_container = av.open(self.output_video_path, mode='w')
                    video_stream = video_container.add_stream('h264', rate=30)
                    video_stream.width = frame.width
                    video_stream.height = frame.height
                    video_stream.pix_fmt = 'yuv420p'
                packet = video_stream.encode(frame)
                video_container.mux(packet)
            except queue.Empty: pass
            except Exception as e: print(f"Video Error: {e}")

            try:
                audio_frame = self.audio_queue.get(timeout=0.01)
                if audio_file is None:
                    audio_file = wave.open(self.output_audio_path, 'wb')
                    audio_file.setnchannels(CHANNELS)
                    audio_file.setsampwidth(2)
                    audio_file.setframerate(RATE)
                audio_data = audio_frame.to_ndarray()
                audio_data = (audio_data * 32767).astype(np.int16)
                audio_file.writeframes(audio_data.tobytes())
            except queue.Empty: pass
            except Exception as e: print(f"Audio Error: {e}")

        if video_container:
            for packet in video_stream.encode():
                video_container.mux(packet)
            video_container.close()
        if audio_file:
            audio_file.close()

if 'recorder_manager' not in st.session_state:
    st.session_state.recorder_manager = RecorderManager()
manager = st.session_state.recorder_manager

def video_frame_callback(frame):
    if manager.recording:
        manager.video_queue.put(frame)
    return frame

def audio_frame_callback(frame):
    if manager.recording:
        manager.audio_queue.put(frame)
    return frame 

def merge_audio_video(video_path, audio_path, output_path):
    try:
        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            clip = VideoFileClip(video_path)
            audio = AudioFileClip(audio_path)
            final_clip = clip.with_audio(audio)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)
            return True
        return False
    except Exception as e:
        print(f"Merge Error: {e}")
        return False

def create_final_session_video(recordings_dict):
    clips = []
    all_attempts = []
    sorted_q_idxs = sorted(recordings_dict.keys())
    for q_idx in sorted_q_idxs:
        attempts = recordings_dict[q_idx]
        for att in attempts:
            all_attempts.append(att)
    try:
        for attempt in all_attempts:
            p = attempt.get('final')
            if p and os.path.exists(p):
                clips.append(VideoFileClip(p))
        if not clips: return None
        final_concat = concatenate_videoclips(clips)
        output_path = f"temp_recordings/interview_session_{int(time.time())}.mp4"
        final_concat.write_videofile(output_path, codec='libx264', audio_codec='aac')
        return output_path
    except Exception as e:
        st.error(f"Concatenation Error: {e}")
        return None

# --- UI Functions ---

# --- Helper Functions (TTS) ---
def text_to_speech(text, filename):
    """Generates TTS audio using OpenAI."""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy", # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        # Use temp dir
        path = f"temp_recordings/{filename}"
        response.stream_to_file(path)
        return path
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# --- UI Functions ---

def main():
    st.set_page_config(page_title="AI Video Interview Coach", page_icon="🎥", layout="wide")
    
    # Custom CSS for Full Screen Focus
    st.markdown("""
    <style>
        .stApp { margin-top: -50px; }
        section[data-testid="stSidebar"] { display: none; } 
    </style>
    """, unsafe_allow_html=True)
    
    phase = st.session_state.interview_phase

    if phase == "setup":
        render_setup_panel()
    elif phase == "camera_check":
        render_camera_check()
    elif phase == "recording" or phase == "feedback":
        render_interview_interface()
    elif phase == "finished":
        render_finished_screen()

def render_setup_panel():
    st.title("AI Interview Coach: Setup 🛠️")
    st.info("Let's tailor the session to your needs. This screen will disappear once we start.")
    
    col_up, col_desc = st.columns(2)
    
    with col_up:
        st.markdown("### 1. Upload Your CV (PDF)")
        uploaded_file = st.file_uploader("Upload a single PDF", type=['pdf'], key="cv_uploader")
        if uploaded_file is not None:
             if st.session_state.cv_text == "":
                with st.spinner("Parsing CV..."):
                    text = extract_text_from_pdf(uploaded_file)
                    if text:
                        st.session_state.cv_text = text
                        st.success("CV Processed Successfully")
        
    with col_desc:
        st.markdown("### 2. Job Description")
        jd_input = st.text_area("Paste the Job Description", height=150)
        if jd_input:
            st.session_state.job_description = jd_input

    st.markdown("### 3. Start")
    start_disabled = not (uploaded_file and st.session_state.job_description)
    
    if st.button("Start Interview Session 🚀", disabled=start_disabled, type="primary", use_container_width=True):
        GEMINI_KEY = get_api_key("GEMINI_API_KEY")
        if not GEMINI_KEY:
            st.error("Cannot start: GEMINI_API_KEY is missing.")
            return
            
        with st.spinner("Analyzing profile and generating questions..."):
            custom_questions = generate_interview_questions(st.session_state.cv_text, st.session_state.job_description)
            if custom_questions:
                full_list = CORE_QUESTIONS[:1] + custom_questions # Mix core and custom
                st.session_state.questions_list = full_list
                st.session_state.interview_phase = "camera_check" # NEXT PHASE
                st.rerun()
            else:
                st.error("Failed to generate questions. Please try again.")

def render_camera_check():
    st.title("Camera & Audio Check 📸")
    st.write("Please ensure you are visible and your microphone is working.")
    
    col_cam, col_inst = st.columns([2, 1])
    
    with col_cam:
        webrtc_streamer(
            key="camera-check",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": True},
            async_processing=True,
        )
        
    with col_inst:
        st.markdown("""
        **Checklist:**
        1.  Are you centered in the frame?
        2.  Is the lighting good?
        3.  Is your background professional?
        """)
        st.markdown("---")
        if st.button("I'm Ready! Start Question 1 ➡️", type="primary", use_container_width=True):
            st.session_state.interview_phase = "recording"
            st.session_state.current_question_index = 0
            st.rerun()

def render_interview_interface():
    q_idx = st.session_state.current_question_index
    total_q = len(st.session_state.questions_list)
    question_text = st.session_state.questions_list[q_idx]

    # Generate TTS for Question if not exists
    q_audio_key = f"q_audio_{q_idx}"
    if q_audio_key not in st.session_state:
        # Generate audio
        path = text_to_speech(f"Question {q_idx + 1}: {question_text}", f"q_{q_idx}.mp3")
        st.session_state[q_audio_key] = path
    
    # Header
    st.progress((q_idx) / total_q, text=f"Question {q_idx + 1} of {total_q}")
    st.title(f"🗣️ {question_text}")
    
    # Auto-play Question Audio (Only once ideally, using key prevents re-render loop if controlled)
    if st.session_state.interview_phase == "recording" and st.session_state.get('q_active_state', 'ready') == 'ready':
         if st.session_state[q_audio_key]:
             st.audio(st.session_state[q_audio_key], autoplay=True)

    phase = st.session_state.interview_phase
    
    # ---------------- RECORDING PHASE ----------------
    if phase == "recording":
        col_vid_main, col_controls = st.columns([2, 1])
        with col_vid_main:
            ctx = webrtc_streamer(
                key=f"interview-q-{q_idx}",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                media_stream_constraints={"video": True, "audio": True},
                video_frame_callback=video_frame_callback,
                audio_frame_callback=audio_frame_callback,
                async_processing=True,
            )
        
        with col_controls:
            st.markdown("### Controls")
            if 'q_active_state' not in st.session_state: st.session_state.q_active_state = 'ready'
            
            if ctx.state.playing:
                if st.session_state.q_active_state == 'ready':
                    if st.button("🔴 Start Recording Answer", type="primary", use_container_width=True):
                         base_filename = f"temp_recordings/q_{q_idx}_{int(time.time())}"
                         vid_path = f"{base_filename}.mp4"
                         aud_path = f"{base_filename}.wav"
                         final_path = f"{base_filename}_merged.mp4"
                         manager.start_recording(vid_path, aud_path, final_path)
                         st.session_state.current_recording = {"video": vid_path, "audio": aud_path, "final": final_path}
                         st.session_state.q_active_state = 'recording'
                         st.rerun()
                elif st.session_state.q_active_state == 'recording':
                    st.error("Recording... (Speak clearly!)")
                    if st.button("⏹️ Finish Answer", type="primary", use_container_width=True):
                         manager.stop_recording()
                         st.session_state.q_active_state = 'ready'
                         if q_idx not in st.session_state.recordings: st.session_state.recordings[q_idx] = []
                         st.session_state.recordings[q_idx].append(st.session_state.current_recording)
                         st.session_state.interview_phase = "feedback"
                         st.rerun()
            else:
                st.warning("Waiting for camera...")

    # ---------------- FEEDBACK PHASE ----------------
    elif phase == "feedback":
        # Process logic (Generate Feedback)
        current_rec_list = st.session_state.recordings.get(q_idx, [])
        current_rec_data = current_rec_list[-1]
        key = f"{q_idx}_{len(current_rec_list)-1}"
        
        if 'processing_done' not in current_rec_data:
             with st.spinner("Coach is analyzing your answer (Content, Voice, Video)..."):
                 merge_audio_video(current_rec_data['video'], current_rec_data['audio'], current_rec_data['final'])
                 transcript = transcribe_audio(current_rec_data['audio'])
                 st.session_state.transcripts[key] = transcript
                 
                 analysis = analyze_answer_with_gemini(
                     transcript, st.session_state.cv_text, st.session_state.job_description, question_text, video_path=current_rec_data['final']
                 )
                 st.session_state.analysis_data[key] = analysis
                 
                 coach_text = generate_coach_response_with_gpt(analysis, transcript, question_text)
                 st.session_state.feedback[key] = coach_text
                 
                 # NEW: Generate Audio for Coach Feedback
                 tts_path = text_to_speech(coach_text, f"feedback_{key}.mp3")
                 st.session_state.feedback_audio = tts_path
                 
                 current_rec_data['processing_done'] = True
                 st.rerun()

        # Display Feedback
        col_feedback_left, col_feedback_right = st.columns([1, 1])
        
        with col_feedback_left:
             st.markdown("### 📹 Your Answer")
             if os.path.exists(current_rec_data['final']):
                st.video(current_rec_data['final'])
             
             st.markdown("### 📝 Transcript")
             st.info(f'"{st.session_state.transcripts.get(key, "")}"')

        with col_feedback_right:
             st.markdown("### Coach Feedback")
             
             # PLAY AUDIO
             if hasattr(st.session_state, 'feedback_audio') and st.session_state.feedback_audio:
                 st.audio(st.session_state.feedback_audio, autoplay=True)
             
             feedback_text = st.session_state.feedback.get(key, "")
             st.write(feedback_text)
             
             # Expandable deeper analysis
             analysis = st.session_state.analysis_data.get(key, {})
             with st.expander("View Body Language & Delivery Analysis"):
                 st.json(analysis)

             st.write("---")
             is_passed = analysis.get("is_strong_enough", False)
             if is_passed:
                 st.success("✅ Good Answer! Ready for next.")
                 if st.button("Next Question ➡️", type="primary"):
                      st.session_state.current_question_index += 1
                      if st.session_state.current_question_index >= len(st.session_state.questions_list):
                          st.session_state.interview_phase = "finished"
                      else:
                          st.session_state.interview_phase = "recording"
                      st.rerun()
             else:
                 st.warning("⚠️ Improvement Needed.")
                 if st.button("Try Again 🔄"):
                      st.session_state.interview_phase = "recording"
                      st.rerun()
                 if st.button("Skip Anyway (Force)"):
                      st.session_state.current_question_index += 1
                      if st.session_state.current_question_index >= len(st.session_state.questions_list):
                          st.session_state.interview_phase = "finished"
                      else:
                          st.session_state.interview_phase = "recording"
                      st.rerun()

def render_finished_screen():
    st.balloons()
    st.title("🎉 Interview Session Complete")
    
    if 'summary_generated' not in st.session_state:
        with st.spinner("Compiling final report and video..."):
            vid_path = create_final_session_video(st.session_state.recordings)
            st.session_state.final_video_path = vid_path
            
            report = generate_session_report_with_gemini(st.session_state.transcripts, st.session_state.analysis_data)
            st.session_state.final_report = report
            st.session_state.summary_generated = True
    
    st.success("You made it! Here is your comprehensive performance package.")
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.markdown("### 📄 Review Report")
        st.download_button("Download Full PDF Report", st.session_state.final_report, "report.txt")
        st.text_area("Preview", st.session_state.final_report, height=300)
    
    with col_dl2:
        st.markdown("### 🎬 Full Session Video")
        if st.session_state.final_video_path:
            with open(st.session_state.final_video_path, "rb") as f:
                st.download_button("Download Video (MP4)", f, "full_interview.mp4")
            st.video(st.session_state.final_video_path)

if __name__ == "__main__":
    main()

