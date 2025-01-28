# --------------------------------------
# utils/ffmpeg_utils.py (改进版, 完整可复制)
# --------------------------------------
import asyncio
import logging
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FFmpegTool:
    """
    统一封装 FFmpeg 常见用法的工具类。
    异步方式调用 FFmpeg，并在出错时抛出异常。
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
        使用 ffmpeg 提取音频，可选指定起始时间与持续时长。
        输出为 PCM float32 单声道格式。
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
        使用 ffmpeg 提取纯视频（去掉音轨），可选指定起始时间与持续时长。
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
        使用 ffmpeg 将输入文件切割为 HLS 片段。
        segment_pattern 类似 "out%03d.ts"
        playlist_path 类似 "playlist.m3u8"
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
        从输入文件截取一段 [start, end] 的视频片段（无音轨）。
        end 是绝对时间(秒)，持续时长 = end - start。
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
            "-an",              # 去除音轨
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
        将"无声视频"与"音频"合并为一个新视频文件 (视频 copy，音频编码成 AAC)。
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
        output_path: str,
        font_size: int = 24,
        font_color: str = "white",
        font_outline: int = 2,
        position: str = "center"  # 可以是 "bottom" / "center" / "top"
    ) -> None:
        """
        将"无声视频" + "音频" + "字幕"合成为含音轨并烧制好字幕的视频。

        1. 主方式: 使用 ffmpeg 的 subtitles filter (例如: subtitles=xxx.ass)
           并结合 force_style 指定字体大小、颜色、描边等。
        2. 若失败，则尝试备用方案(-vf subtitles=...)。
        3. 若仍失败，则仅合并视频和音频，跳过字幕。
        """
        # 确保这些文件都存在
        for file_path in [input_video_path, input_audio_path, subtitles_path]:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")

        # 准备 force_style，给 SRT/ASS 字幕设置基础样式
        style_str = f"FontSize={font_size},PrimaryColour={font_color},Outline={font_outline}"
        # 根据 position 控制margin
        if position == "bottom":
            margin_v = 20
        elif position == "top":
            margin_v = 10
        else:
            margin_v = 0

        style_str = f"MarginV={margin_v}," + style_str

        # 构建 filter
        escaped_sub_path = subtitles_path.replace(':', r'\\:')
        subtitles_filter = f"subtitles='{escaped_sub_path}':force_style='{style_str}'"

        try:
            # 方式 1: 字幕过滤器
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-i", input_audio_path,
                "-filter_complex", f"[0:v]{subtitles_filter}[v]",
                "-map", "[v]",
                "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "superfast",
                "-c:a", "aac",
                output_path
            ]
            await self.run_command(cmd)

        except RuntimeError as e:
            logger.warning(f"[FFmpegTool] 主字幕渲染方案失败: {str(e)}")
            # 方式 2: 备用 -vf subtitles=
            try:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", input_video_path,
                    "-i", input_audio_path,
                    "-vf", f"subtitles='{subtitles_path}'",
                    "-c:v", "libx264",
                    "-preset", "superfast",
                    "-c:a", "aac",
                    output_path
                ]
                await self.run_command(cmd)

            except RuntimeError as e2:
                logger.error(f"[FFmpegTool] 字幕渲染完全失败: {str(e2)}")
                # 方式 3: 最终回退 - 仅合并音视频
                cmd = [
                    "ffmpeg", "-y",
                    "-i", input_video_path,
                    "-i", input_audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    output_path
                ]
                await self.run_command(cmd)
                logger.warning("[FFmpegTool] 已跳过字幕, 仅合并音视频")

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
