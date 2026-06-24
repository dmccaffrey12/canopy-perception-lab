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

def get_youtube_thumbnail(video_id: str) -> np.ndarray:
    """Downloads the highest resolution thumbnail available for the video."""
    import urllib.request
    
    urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/0.jpg"
    ]
    
    for url in urls:
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                image_data = response.read()
                arr = np.asarray(bytearray(image_data), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            continue
            
    raise ValueError("Could not retrieve any video thumbnail.")

def get_middle_frame(youtube_url: str) -> tuple[np.ndarray, bool]:
    """
    Downloads a low-resolution version of the YouTube video to a temporary file
    using yt-dlp, seeks to the exact middle, and captures that single frame.
    
    If YouTube blocks the request (HTTP 403, common on cloud platforms like Streamlit Cloud),
    it falls back to downloading the video's static high-resolution thumbnail.
    
    Returns:
        tuple[np.ndarray, bool]: (image_matrix_rgb, is_fallback_thumbnail)
    """
    cleaned_url = clean_youtube_url(youtube_url)
    
    # Extract video ID for thumbnail fallback
    pattern = r'v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, cleaned_url)
    video_id = match.group(1) if match else None
    
    # Create a unique temporary file path
    temp_dir = tempfile.gettempdir()
    temp_filename = f"canopy_temp_{uuid.uuid4().hex}.mp4"
    temp_filepath = os.path.join(temp_dir, temp_filename)
    
    ydl_opts = {
        'format': 'worstvideo[ext=mp4]/18/worst[ext=mp4]/worst',
        'outtmpl': temp_filepath,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([cleaned_url])
            
        # Open the downloaded file using OpenCV
        cap = cv2.VideoCapture(temp_filepath)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")
            
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                middle_frame_idx = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
            else:
                ret, frame = cap.read()
                
            if not ret or frame is None:
                raise ValueError("Could not read frame.")
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb, False
            
        finally:
            cap.release()
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass
                    
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
                
        # If video download fails (e.g. 403), try downloading the static thumbnail
        if video_id:
            try:
                thumbnail = get_youtube_thumbnail(video_id)
                return thumbnail, True
            except Exception as thumb_err:
                raise ValueError(f"Failed to extract frame and thumbnail fallback failed. Details: {e} | Thumbnail Error: {thumb_err}")
        else:
            raise ValueError(f"Failed to download video stream and could not determine video ID for thumbnail. Details: {e}")


