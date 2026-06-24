import cv2
import yt_dlp
import numpy as np
import re
import os
import tempfile
import uuid

def clean_youtube_url(url: str) -> str:
    """Normalizes YouTube URLs to standard watch format and strips timestamps or tracking parameters."""
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/)|youtu\.be/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    if match:
        video_id = match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

def get_middle_frame(youtube_url: str) -> np.ndarray:
    """
    Downloads a low-resolution version of the YouTube video to a temporary file
    using yt-dlp, seeks to the exact middle, and captures that single frame.
    
    This avoids HTTP 403 blocking issues that occur when streaming directly
    from cloud environments like Streamlit Cloud.
    """
    cleaned_url = clean_youtube_url(youtube_url)
    
    # Create a unique temporary file path
    temp_dir = tempfile.gettempdir()
    temp_filename = f"canopy_temp_{uuid.uuid4().hex}.mp4"
    temp_filepath = os.path.join(temp_dir, temp_filename)
    
    # We prioritize low-resolution video-only format to minimize download size and time
    ydl_opts = {
        'format': 'worstvideo[ext=mp4]/18/worst[ext=mp4]/worst',
        'outtmpl': temp_filepath,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([cleaned_url])
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Failed to download video. The URL might be private, geo-blocked, or invalid. Details: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred during video download: {e}")
        
    # Open the downloaded file using OpenCV
    cap = cv2.VideoCapture(temp_filepath)
    
    if not cap.isOpened():
        # Cleanup file before raising error
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
        raise ValueError("Could not open the downloaded video file. The file format might be incompatible.")
        
    try:
        # Get total frames to find the middle one
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            middle_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            
            ret, frame = cap.read()
            if not ret:
                # Fallback to first frame if seeking failed
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        else:
            ret, frame = cap.read()
            
        if not ret or frame is None:
            raise ValueError("Could not read frame from the video file.")
            
        # Convert BGR (OpenCV) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
        
    finally:
        cap.release()
        # Clean up the temporary file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass

