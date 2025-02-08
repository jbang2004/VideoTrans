import asyncio
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FFmpegTool:
    async def run_command(self, cmd: List[str]) -> Tuple[bytes, bytes]:
        logger.debug(f"Running command: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode() or "Unknown error"
            logger.error(f"Command failed: {error_msg}")
            raise RuntimeError(f"FFmpeg command failed: {error_msg}")
        return stdout, stderr

    async def extract_audio(self, input_path: str, output_path: str, start: float = 0.0, duration: Optional[float] = None) -> None:
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if start > 0:
            cmd += ["-ss", str(start)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-vn", "-acodec", "pcm_f32le", "-ac", "1", output_path]
        await self.run_command(cmd)

    async def extract_video(self, input_path: str, output_path: str, start: float = 0.0, duration: Optional[float] = None) -> None:
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if start > 0:
            cmd += ["-ss", str(start)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-tune", "fastdecode", output_path]
        await self.run_command(cmd)

    async def hls_segment(self, input_path: str, segment_pattern: str, playlist_path: str, hls_time: int = 10) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c", "copy",
            "-f", "hls",
            "-hls_time", str(hls_time),
            "-hls_list_size", "0",
            "-hls_segment_type", "mpegts",
            "-hls_segment_filename", segment_pattern,
            playlist_path
        ]
        await self.run_command(cmd)

    async def cut_video_track(self, input_path: str, output_path: str, start: float, end: float) -> None:
        duration = end - start
        if duration <= 0:
            raise ValueError(f"Invalid duration: {duration}")
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", str(start),
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "superfast",
            "-an",
            "-vsync", "vfr",
            output_path
        ]
        await self.run_command(cmd)

    async def cut_video_with_audio(self, input_video_path: str, input_audio_path: str, output_path: str) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video_path,
            "-i", input_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ]
        await self.run_command(cmd)

    async def cut_video_with_subtitles_and_audio(self, input_video_path: str, input_audio_path: str, subtitles_path: str, output_path: str) -> None:
        for file_path in [input_video_path, input_audio_path, subtitles_path]:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
        escaped_path = subtitles_path.replace(':', r'\\:')
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-i", input_audio_path,
                "-filter_complex",
                f"[0:v]scale=1920:-2:flags=lanczos,subtitles='{escaped_path}'[v]",
                "-map", "[v]",
                "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "superfast",
                "-crf", "23",
                "-c:a", "aac",
                output_path
            ]
            await self.run_command(cmd)
        except RuntimeError as e:
            logger.warning(f"Subtitles filter failed: {e}")
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-i", input_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                output_path
            ]
            await self.run_command(cmd)

    async def get_duration(self, input_path: str) -> float:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ]
        stdout, _ = await self.run_command(cmd)
        return float(stdout.decode().strip())
