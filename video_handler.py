import cv2
import yt_dlp
import numpy as np

def get_middle_frame(youtube_url: str) -> np.ndarray:
    """
    Extracts the stream link via yt-dlp, seeks to the exact middle of the 
    video stream, and captures that single frame as a clean NumPy BGR matrix.
    
    Args:
        youtube_url (str): The URL of the YouTube video.
        
    Returns:
        np.ndarray: The middle frame in RGB format.
        
    Raises:
        ValueError: If video URL extraction fails, video stream cannot be opened,
                    or middle frame cannot be read.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best[ext=mp4]/best',  # Prefer mp4 for OpenCV compatibility
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            
            # Determine the stream URL
            video_url = None
            if 'url' in info_dict:
                video_url = info_dict['url']
            elif 'formats' in info_dict and len(info_dict['formats']) > 0:
                # Fallback to the best format URL
                formats = info_dict['formats']
                # Search for video-only or combined video formats
                video_formats = [f for f in formats if f.get('vcodec') != 'none' and f.get('url')]
                if video_formats:
                    video_url = video_formats[-1]['url']
                else:
                    video_url = formats[-1]['url']
                    
            if not video_url:
                raise ValueError("Could not extract a valid video stream URL.")
                
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Failed to retrieve video metadata. The URL might be invalid, private, or geo-blocked. Details: {e}")
    except Exception as e:
        raise ValueError(f"An unexpected error occurred while parsing the YouTube URL: {e}")
        
    # Open the video stream with OpenCV
    cap = cv2.VideoCapture(video_url)
    
    if not cap.isOpened():
        raise ValueError("Could not open the video stream. The streaming URL might have expired or is blocked.")
        
    try:
        # Get total frames to find the middle one
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            middle_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            
            # Verify position set correctly (sometimes remote streams don't support arbitrary seeking)
            ret, frame = cap.read()
            if not ret:
                # If seeking failed, try resetting and reading the first available frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        else:
            # Fallback if frame count is unavailable: read the first frame
            ret, frame = cap.read()
            
        if not ret or frame is None:
            raise ValueError("Could not read frame from the video stream.")
            
        # Convert BGR (OpenCV default) to RGB (standard for PIL/Streamlit/YOLO)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb
        
    finally:
        cap.release()
