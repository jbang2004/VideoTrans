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
        如果视频没有音频流，则生成一个空的音频文件。
        """
        # 首先检查视频是否有音频流
        has_audio = await self._check_audio_stream(input_path)
        
        if has_audio:
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
        else:
            # 如果没有音频流，生成一个空的音频文件
            logger.warning(f"[FFmpegTool] 视频 {input_path} 没有音频流，生成空音频文件")
            # 获取视频时长
            video_duration = await self._get_video_duration(input_path)
            if duration is not None:
                video_duration = min(video_duration, duration)
            
            # 生成静音音频
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=r=48000:cl=mono",
                "-t", str(video_duration),
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
        hls_time: int = 10,
        hls_flags: str = "independent_segments",
    ) -> None:
        """
        将输入视频切分成HLS分片
        Args:
            input_path: 输入视频路径
            segment_pattern: 分片文件名模式
            playlist_path: 临时m3u8文件路径
            hls_time: 每个分片的目标时长(秒)
            hls_flags: HLS特殊标志
            extra_options: 额外的 FFmpeg 选项
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

        # 构建"subtitles"过滤器, 不带 force_style
        escaped_path = subtitles_path.replace(':', r'\\:')

        try:
            # 方式1: subtitles 滤镜
            # 当前设置是合理的，但需要注意：
            cmd = [
                "ffmpeg", "-y",
                "-i", input_video_path,
                "-i", input_audio_path,
                "-filter_complex",
                f"[0:v]scale=1920:-2:flags=lanczos,subtitles='{escaped_path}'[v]",  # 修改点说明：
                # 1. scale=1920:-2 保持宽高比，-2 保证高度为偶数（兼容编码要求）
                # 2. flags=lanczos 使用高质量的缩放算法
                # 3. 滤镜顺序：先缩放视频，再加字幕（确保字幕在缩放后的画面上）
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

    async def _check_audio_stream(self, input_path: str) -> bool:
        """检查视频是否有音频流"""
        cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", 
               "stream=codec_type", "-of", "json", input_path]
        try:
            stdout, _ = await self.run_command(cmd)
            import json
            result = json.loads(stdout)
            return 'streams' in result and len(result['streams']) > 0
        except Exception as e:
            logger.error(f"[FFmpegTool] 检查音频流失败: {e}")
            return False
            
    async def _get_video_duration(self, input_path: str) -> float:
        """获取视频时长"""
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", input_path]
        stdout, _ = await self.run_command(cmd)
        return float(stdout.decode().strip())

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
