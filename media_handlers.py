import os
import time
import threading
import queue
import wave
import av
import numpy as np
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
import streamlit as st

class RecorderManager:
    def __init__(self):
        self.frames = []
        self.audio_frames = []
        self.is_recording = False
        self._lock = threading.Lock()
        
    def start_recording(self, video_path, audio_path, final_path):
        with self._lock:
            self.frames = []
            self.audio_frames = []
            self.is_recording = True
            self.video_out_path = video_path
            self.audio_out_path = audio_path
            self.final_out_path = final_path
            
    def add_video_frame(self, frame):
        if self.is_recording:
            # frame is av.VideoFrame
            img = frame.to_ndarray(format="bgr24")
            self.frames.append(img)
            
    def add_audio_frame(self, frame):
        if self.is_recording:
            # frame is av.AudioFrame
            data = frame.to_ndarray()
            self.audio_frames.append(data)
            
    def stop_recording(self):
        with self._lock:
            self.is_recording = False
            self.save_files()
            
    def save_files(self):
        # Save Video (Simplified for demo - in prod use ffmpeg/cv2 writer properly)
        # Note: Writing raw frames to mp4 in streamlit without simple-webrtc's built-in recorder is complex.
        # For this simplified version we rely on the fact that webrtc_streamer can handle writing if configured,
        # OR we save frames. Writing video from raw numpy arrays requires cv2 or moviepy.
        # Let's use a placeholder or assume we have a writer.
        # ACTUALLY: streamlit-webrtc allows access to media recorder in JS side usually, but purely python side:
        import cv2
        if len(self.frames) > 0:
            height, width, layers = self.frames[0].shape
            # codec = cv2.VideoWriter_fourcc(*'mp4v') 
            # Using XVID for compatibility or avc1
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.video_out_path, fourcc, 20.0, (width, height))
            for frame in self.frames:
                out.write(frame)
            out.release()
            
        # Save Audio
        if len(self.audio_frames) > 0:
            audio_data = np.concatenate(self.audio_frames, axis=0) # Flatten
            # Check format. Webrtc usually gives int16 or float32. 
            # Assuming stereo/mono depending on setup.
            # Convert to int16 for wav
            if audio_data.dtype != np.int16:
                 audio_data = (audio_data * 32767).astype(np.int16)
                 
            with wave.open(self.audio_out_path, 'wb') as wf:
                wf.setnchannels(2) # webrtc default often stereo
                wf.setsampwidth(2)
                wf.setframerate(48000) # common webrtc rate
                wf.writeframes(audio_data.tobytes())

# Global manager instance (to be used by callbacks)
manager = RecorderManager()

def video_frame_callback(frame):
    manager.add_video_frame(frame)
    return frame

def audio_frame_callback(frame):
    manager.add_audio_frame(frame)
    return frame

def merge_audio_video(video_path, audio_path, output_path):
    try:
        # Check if files exist
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            return None
            
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        
        # Trim audio to match video duration or vice versa
        # Usually video duration dictates
        final_clip = video_clip.with_audio(audio_clip)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        
        video_clip.close()
        audio_clip.close()
        return output_path
    except Exception as e:
        print(f"Merge error: {e}")
        return None

def create_final_session_video(recordings_dict):
    """Concatenates all final clips into one long session video."""
    clips = []
    # Sort by question index
    sorted_indices = sorted(recordings_dict.keys())
    
    for idx in sorted_indices:
        # take the last attempt for each question
        rec_data = recordings_dict[idx][-1]
        path = rec_data.get('final')
        if path and os.path.exists(path):
            clips.append(VideoFileClip(path))
            
    if not clips:
        return None
        
    final_video = concatenate_videoclips(clips)
    output_path = f"temp_recordings/full_session_{int(time.time())}.mp4"
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path
