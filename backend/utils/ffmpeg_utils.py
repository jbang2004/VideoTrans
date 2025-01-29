# --------------------------------------
# utils/ffmpeg_utils.py
# 彻底移除 force_style, 仅使用 .ass 内部样式
# --------------------------------------
import asyncio
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FFmpegTool:
    """
    统一封装 FFmpeg 常见用法的工具类。
    通过异步方式执行 ffmpeg 命令，并在出错时抛出异常。
    """

    async def run_command(self, cmd: List[str]) -> Tuple[bytes, bytes]:
        """
        运行 ffmpeg 命令，返回 (stdout, stderr)。
        若命令返回码非 0，则抛出 RuntimeError。
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
        提取音频，可选指定起始时间与持续时长。
        输出为单声道 PCM float32 (48k/16k 视需求).
        """
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if start > 0:
            cmd += ["-ss", str(start)]
        if duration is not None:
            cmd += ["-t", str(duration)]

        cmd += [
            "-vn",                # 去掉视频
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
        提取纯视频（去掉音轨），可选指定起始时间与持续时长。
        """
        cmd = ["ffmpeg", "-y", "-i", input_path]
        if start > 0:
            cmd += ["-ss", str(start)]
        if duration is not None:
            cmd += ["-t", str(duration)]

        cmd += [
            "-an",                # 去掉音频
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
        将输入视频切割为 HLS 片段。
        segment_pattern 形如 "out%03d.ts"
        playlist_path   形如 "playlist.m3u8"
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

    async def cut_video_track(
        self,
        input_path: str,
        output_path: str,
        start: float,
        end: float
    ) -> None:
        """
        截取 [start, end] 的无声视频段，end为绝对秒数。
        """
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
            "-an",  # 去除音轨
            "-vsync", "vfr",
            output_path
        ]
        await self.run_command(cmd)

    async def cut_video_with_audio(
        self,
        input_video_path: str,
        input_audio_path: str,
        output_path: str
    ) -> None:
        """
        将无声视频与音频合并 (视频copy，音频AAC)。
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

    async def cut_video_with_subtitles_and_audio(
        self,
        input_video_path: str,
        input_audio_path: str,
        subtitles_path: str,
        output_path: str
    ) -> None:
        """
        将无声视频 + 音频 + .ass字幕 合并输出到 output_path。
        (无 force_style, 由 .ass 内样式全权决定)
        
        若字幕渲染失败，则回退到仅合并音视频。
        """
        # 检查输入文件是否存在
        for file_path in [input_video_path, input_audio_path, subtitles_path]:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

        # 构建“subtitles”过滤器, 不带 force_style
        escaped_path = subtitles_path.replace(':', r'\\:')

        try:
            # 方式1: subtitles 滤镜
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-i", input_audio_path,
                "-filter_complex",
                f"[0:v]scale=1920:1080:flags=lanczos,subtitles='{escaped_path}'[v]",  # 添加 scale 滤镜，使用 lanczos 缩放算法
                "-map", "[v]",
                "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "superfast",
                "-crf", "23",  # 建议添加 CRF 参数控制视频质量
                "-c:a", "aac",
                output_path
            ]
            await self.run_command(cmd)

        except RuntimeError as e:
            logger.warning(f"[FFmpegTool] subtitles滤镜方案失败: {str(e)}")
            # 方式2: 最终回退 - 仅合并音视频
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-i", input_audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                output_path
            ]
            await self.run_command(cmd)
            logger.warning("[FFmpegTool] 已跳过字幕，仅合并音视频")

    async def get_duration(self, input_path: str) -> float:
        """
        调用 ffprobe 获取输入文件的时长(秒)。
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
