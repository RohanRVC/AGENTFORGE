
import os
import subprocess
from typing import List


def ensure_dir(path: str):
    """
    Make sure a directory exists.
    """
    os.makedirs(path, exist_ok=True)


def extract_keyframes_ffmpeg(video_path: str, output_dir: str, seconds_interval: int = 3) -> List[str]:
    """
    Use FFmpeg to extract key frames from the video.

    - video_path: path to the input video file
    - output_dir: directory where frame images will be stored
    - seconds_interval: interval in seconds between frames

    Returns a list of file paths to the extracted frames.
    """
    ensure_dir(output_dir)

    output_pattern = os.path.join(output_dir, "frame_%03d.jpg")

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps=1/{seconds_interval}",
        "-q:v",
        "2",
        output_pattern,
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    subprocess.run(cmd, check=True)

    frames = [
        os.path.join(output_dir, f)
        for f in sorted(os.listdir(output_dir))
        if f.lower().endswith(".jpg")
    ]

    return frames


def extract_audio_ffmpeg(video_path: str, output_audio_path: str) -> str:
    """
    Extract the audio track from a video file and save as WAV.

    Returns the path to the extracted audio file.
    """
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_audio_path,
        "-hide_banner",
        "-loglevel",
        "error",
    ]

    subprocess.run(cmd, check=True)
    return output_audio_path
