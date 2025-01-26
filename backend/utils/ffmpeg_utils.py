# utils/ffmpeg_utils.py
import asyncio
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FFmpegTool:
    """统一封装 FFmpeg 常见用法的工具类。"""

    async def run_command(self, cmd: List[str]) -> Tuple[bytes, bytes]:
        """
        运行 ffmpeg 子进程命令，返回 (stdout, stderr)。
        若返回码非 0，则抛出 RuntimeError。
        """
        logger.debug(f"[FFmpegTool] Running command: {' '.join(cmd)}")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode() or "Unknown error"
            logger.error(f"[FFmpegTool] Command failed with error: {error_msg}")
            raise RuntimeError(f"FFmpeg command failed: {error_msg}")
        return stdout, stderr

    async def extract_audio(
        self,
        input_path: str,
        output_path: str,
        start: float = 0.0,
        duration: Optional[float] = None
    ) -> None:
        """
        使用 ffmpeg 提取音频，可选指定起始时间与持续时长。
        """
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if start > 0:
            cmd += ["-ss", str(start)]
        if duration is not None:
            cmd += ["-t", str(duration)]

        cmd += [
            "-vn",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            output_path
        ]
        await self.run_command(cmd)

    async def extract_video(
        self,
        input_path: str,
        output_path: str,
        start: float = 0.0,
        duration: Optional[float] = None
    ) -> None:
        """
        使用 ffmpeg 提取纯视频（无音轨），可选指定起始时间与持续时长。
        """
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if start > 0:
            cmd += ["-ss", str(start)]
        if duration is not None:
            cmd += ["-t", str(duration)]

        # 这里选择了重新编码以确保兼容性
        cmd += [
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "18",
            "-tune", "fastdecode",
            output_path
        ]
        await self.run_command(cmd)

    async def hls_segment(
        self,
        input_path: str,
        segment_pattern: str,
        playlist_path: str,
        hls_time: int = 10
    ) -> None:
        """
        使用 ffmpeg 将输入文件切割为 HLS 片段。
        """
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

    async def cut_video_with_audio(
        self, input_video_path: str, input_audio_path: str, output_path: str
    ) -> None:
        """
        将已切割好的无声视频与音频合并，输出含音轨的视频。
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", input_video_path,
            "-i", input_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            output_path
        ]
        await self.run_command(cmd)

    async def cut_video_track(
        self, input_path: str, output_path: str, start: float, end: float
    ) -> None:
        """
        从输入文件截取一段视频片段（无音轨），
        其中 end 为绝对时间（秒），因此持续时长 = end - start。
        """
        duration = end - start
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

    async def get_duration(self, input_path: str) -> float:
        """
        调用 ffprobe 获取输入文件的时长（秒）。
        """
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ]
        stdout, stderr = await self.run_command(cmd)
        return float(stdout.decode().strip())
