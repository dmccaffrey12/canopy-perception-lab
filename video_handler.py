import cv2
import yt_dlp
import numpy as np
import re
import os
import tempfile
import uuid
import urllib.request

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

def get_frame(source, frame_pct: float = 0.5) -> tuple[np.ndarray, bool]:
    """
    Ingests video from YouTube, direct URLs, or uploaded file buffers,
    and extracts a frame at the specified percentage through the video (0.0 to 1.0).
    
    Returns:
        tuple[np.ndarray, bool]: (frame_rgb, is_fallback_thumbnail)
    """
    # Create unique temp file path
    temp_dir = tempfile.gettempdir()
    temp_filename = f"canopy_temp_{uuid.uuid4().hex}.mp4"
    temp_filepath = os.path.join(temp_dir, temp_filename)
    
    is_youtube = False
    video_id = None
    
    try:
        # Case 1: Uploaded file buffer (BytesIO / Streamlit UploadedFile)
        if hasattr(source, 'read'):
            source.seek(0)
            with open(temp_filepath, 'wb') as f:
                f.write(source.read())
                
        # Case 2: URL (string)
        elif isinstance(source, str) and source.startswith(('http://', 'https://')):
            cleaned_url = clean_youtube_url(source)
            if "youtube.com" in cleaned_url or "youtu.be" in cleaned_url:
                is_youtube = True
                pattern = r'v=([a-zA-Z0-9_-]{11})'
                match = re.search(pattern, cleaned_url)
                video_id = match.group(1) if match else None
                
                # Try downloading via yt-dlp
                ydl_opts = {
                    'format': 'worstvideo[ext=mp4]/18/worst[ext=mp4]/worst',
                    'outtmpl': temp_filepath,
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([cleaned_url])
            else:
                # Direct MP4 / Video URL
                req = urllib.request.Request(
                    source, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response:
                    with open(temp_filepath, 'wb') as f:
                        f.write(response.read())
        else:
            raise ValueError("Unsupported video source type.")
            
        # Open and extract the frame
        cap = cv2.VideoCapture(temp_filepath)
        if not cap.isOpened():
            raise ValueError("Could not open the video file.")
            
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                target_frame = int(total_frames * frame_pct)
                target_frame = min(max(0, target_frame), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                
                ret, frame = cap.read()
                if not ret:
                    # Fallback to first frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
            else:
                ret, frame = cap.read()
                
            if not ret or frame is None:
                raise ValueError("Could not read frame from video.")
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb, False
            
        finally:
            cap.release()
            
    except Exception as e:
        # If it's a YouTube link and failed, try fallback to thumbnail
        if is_youtube and video_id:
            try:
                thumbnail = get_youtube_thumbnail(video_id)
                return thumbnail, True
            except Exception as thumb_err:
                raise ValueError(f"Failed to load YouTube stream and thumbnail fallback failed. Details: {e} | Thumbnail Error: {thumb_err}")
        else:
            raise ValueError(f"Failed to process video source. Details: {e}")
            
    finally:
        # Clean up temp file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
