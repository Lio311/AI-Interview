import streamlit as st
import time
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Import modules
from utils import extract_text_from_pdf, get_api_key
from ai_handlers import (
    setup_ai_clients,
    generate_interview_questions,
    transcribe_audio,
    analyze_answer_with_gemini,
    generate_coach_response_with_gpt,
    generate_session_report_with_gemini,
    text_to_speech
)
from media_handlers import (
    manager,
    video_frame_callback,
    audio_frame_callback,
    merge_audio_video,
    create_final_session_video
)

# Constants
CORE_QUESTIONS = [
    "Tell me about yourself.",
    "What are your greatest strengths?",
    "Why do you want to work here?"
]

def init_session_state():
    if 'cv_text' not in st.session_state: st.session_state.cv_text = ""
    if 'job_description' not in st.session_state: st.session_state.job_description = ""
    if 'questions_list' not in st.session_state: st.session_state.questions_list = []
    if 'current_question_index' not in st.session_state: st.session_state.current_question_index = 0
    if 'recordings' not in st.session_state: st.session_state.recordings = {} 
    if 'interview_phase' not in st.session_state: st.session_state.interview_phase = "setup"
    if 'transcripts' not in st.session_state: st.session_state.transcripts = {}
    if 'feedback' not in st.session_state: st.session_state.feedback = {}
    if 'analysis_data' not in st.session_state: st.session_state.analysis_data = {}

def main():
    st.set_page_config(page_title="AI Video Interview Coach", layout="wide")
    
    # Custom CSS for Full Screen Focus, No Sidebar, and Centering
    st.markdown("""
    <style>
        .stApp { margin-top: -50px; }
        section[data-testid="stSidebar"] { display: none; } 
        /* Center video and limit size */
        div[data-testid="stVerticalBlock"] > div:has(div.stVideo) { display: flex; justify-content: center; }
        video { 
            width: 400px !important; 
            height: auto !important; 
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Persistent Streamer
    # We use a key that doesn't change to keep connection alive
    ctx = webrtc_streamer(
        key="persistent-cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": True},
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        async_processing=True,
        video_html_attrs={
            "style": {"width": "400px", "margin": "0 auto", "border-radius": "10px"},
            "muted": True  # Fixes the echo issue!
        }
    )
    
    # Center Layout container for controls below camera
    main_container = st.container()

    with main_container:
        if phase == "camera_check":
            render_camera_check(ctx)
        elif phase == "recording":
            # Pass ctx to recording phase
            render_recording_phase(ctx)
        elif phase == "feedback":
            render_feedback_phase(ctx)
        elif phase == "finished":
            render_finished_screen()

def render_setup_panel():
    st.title("AI Interview Coach: Setup")
    st.info("Let's tailor the session to your needs.")
    
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
    
    if st.button("Start Interview Session", disabled=start_disabled, type="primary", use_container_width=True):
        if not get_api_key("GEMINI_API_KEY"):
            st.error("Cannot start: GEMINI_API_KEY is missing.")
            return
            
        with st.spinner("Analyzing profile and generating questions..."):
            custom_questions = generate_interview_questions(st.session_state.cv_text, st.session_state.job_description)
            if custom_questions:
                full_list = CORE_QUESTIONS[:1] + custom_questions 
                st.session_state.questions_list = full_list
                st.session_state.interview_phase = "camera_check"
                st.rerun()
            else:
                st.error("Failed to generate questions. Please try again.")

def render_camera_check(ctx):
    st.markdown("---")
    st.info("Check your camera above. When ready, click Start.")
    
    if ctx.state.playing:
        if st.button("I'm Ready! Start Interview", type="primary", use_container_width=True):
            st.session_state.interview_phase = "recording"
            st.session_state.current_question_index = 0
            st.session_state.q_active_state = 'ready_to_start' # Trigger auto-start
            st.rerun()
    else:
        st.warning("Waiting for camera connection...")

def render_recording_phase(ctx):
    q_idx = st.session_state.current_question_index
    total_q = len(st.session_state.questions_list)
    question_text = st.session_state.questions_list[q_idx]
    
    # TTS Logic
    q_audio_key = f"q_audio_{q_idx}"
    if q_audio_key not in st.session_state:
        path = text_to_speech(f"Question {q_idx + 1}: {question_text}", f"q_{q_idx}.mp3")
        st.session_state[q_audio_key] = path
    
    # UI
    st.markdown(f"<h3 style='text-align: center;'>Question {q_idx + 1} / {total_q}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center;'>{question_text}</h1>", unsafe_allow_html=True)
    
    # Auto-Play Audio
    if st.session_state[q_audio_key]:
        st.audio(st.session_state[q_audio_key], autoplay=True)

    # Auto-Recording Logic
    if ctx.state.playing:
        if st.session_state.get('q_active_state') == 'ready_to_start':
             # Start Manager
             base_filename = f"temp_recordings/q_{q_idx}_{int(time.time())}"
             vid_path = f"{base_filename}.mp4"
             aud_path = f"{base_filename}.wav"
             final_path = f"{base_filename}_merged.mp4"
             
             manager.start_recording(vid_path, aud_path, final_path)
             st.session_state.current_recording = {"video": vid_path, "audio": aud_path, "final": final_path}
             
             # ADVANCE STATE
             st.session_state.q_active_state = 'recording'
             st.rerun() 
             
        elif st.session_state.get('q_active_state') == 'recording':
            st.error("🔴 Recording in progress... (Speak clearly!)")
            
            # Use columns to center the button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("⏹️ Stop Recording & JSON Report", type="primary", use_container_width=True, key=f"stop_btn_{q_idx}"):
                    manager.stop_recording()
                    st.session_state.q_active_state = 'finished'
                    if q_idx not in st.session_state.recordings: 
                        st.session_state.recordings[q_idx] = []
                    st.session_state.recordings[q_idx].append(st.session_state.current_recording)
                    st.session_state.interview_phase = "feedback"
                    st.rerun()
    else:
        st.warning("Camera disconnected. Please reconnect.")

def render_feedback_phase(ctx):
    q_idx = st.session_state.current_question_index
    question_text = st.session_state.questions_list[q_idx]

    current_rec_list = st.session_state.recordings.get(q_idx, [])
    current_rec_data = current_rec_list[-1]
    key = f"{q_idx}_{len(current_rec_list)-1}"
    
    # Analysis Logic (Auto run)
    if 'processing_done' not in current_rec_data:
         with st.spinner("Analyzing your answer..."):
             merge_audio_video(current_rec_data['video'], current_rec_data['audio'], current_rec_data['final'])
             transcript = transcribe_audio(current_rec_data['audio'])
             st.session_state.transcripts[key] = transcript
             analysis = analyze_answer_with_gemini(
                 transcript, st.session_state.cv_text, st.session_state.job_description, question_text, video_path=current_rec_data['final']
             )
             st.session_state.analysis_data[key] = analysis
             coach_text = generate_coach_response_with_gpt(analysis, transcript, question_text)
             st.session_state.feedback[key] = coach_text
             tts_path = text_to_speech(coach_text, f"feedback_{key}.mp3")
             st.session_state.feedback_audio = tts_path
             current_rec_data['processing_done'] = True
             st.rerun()
             
    # Display Feedback
    st.markdown("### 🤖 Coach Feedback")
    
    # Auto-Play Feedback Audio
    if hasattr(st.session_state, 'feedback_audio') and st.session_state.feedback_audio:
         st.audio(st.session_state.feedback_audio, autoplay=True)
         
    feedback_text = st.session_state.feedback.get(key, "")
    st.success(feedback_text)
    
    # Transcript & Details
    with st.expander("View Transcript"):
        st.write(st.session_state.transcripts.get(key, ""))
        
    analysis = st.session_state.analysis_data.get(key, {})
    with st.expander("Detailed Analysis (Body Language & Content)"):
        st.json(analysis)
        
    # Controls
    st.write("---")
    col_next, col_retry = st.columns(2)
    
    if col_next.button("Next Question ➡️", type="primary", use_container_width=True):
        st.session_state.current_question_index += 1
        if st.session_state.current_question_index >= len(st.session_state.questions_list):
            st.session_state.interview_phase = "finished"
        else:
            st.session_state.interview_phase = "recording"
            st.session_state.q_active_state = 'ready_to_start' # Reset auto-start for next Q
        st.rerun()
        
    if col_retry.button("Try Again 🔄", use_container_width=True):
        st.session_state.interview_phase = "recording"
        st.session_state.q_active_state = 'ready_to_start'
        st.rerun()

def render_finished_screen():
    st.balloons()
    st.title("Interview Session Complete")
    
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
        st.markdown("### Review Report")
        st.download_button("Download Full PDF Report", st.session_state.final_report, "report.txt")
        st.text_area("Preview", st.session_state.final_report, height=300)
    
    with col_dl2:
        st.markdown("### Full Session Video")
        if st.session_state.final_video_path:
            with open(st.session_state.final_video_path, "rb") as f:
                st.download_button("Download Video (MP4)", f, "full_interview.mp4")
            st.video(st.session_state.final_video_path)

if __name__ == "__main__":
    main()
