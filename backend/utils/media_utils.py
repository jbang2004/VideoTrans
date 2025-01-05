import logging
import asyncio
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Union
import subprocess
from utils.decorators import handle_errors
from utils.temp_file_manager import TempFileManager

logger = logging.getLogger(__name__)

class MediaUtils:
    def __init__(self, config, audio_separator):
        self.config = config
        self.target_sr = self.config.TARGET_SR
        self.vad_sr = self.config.VAD_SR
        self.logger = logging.getLogger(__name__)
        self.audio_separator = audio_separator

    def normalize_and_resample(self, 
                             audio_input: Union[Tuple[int, np.ndarray], np.ndarray],
                             target_sr: int = None) -> np.ndarray:
        """音频标准化和重采样处理"""
        target_sr = target_sr or self.target_sr
        
        # 解析输入
        if isinstance(audio_input, tuple):
            fs, audio_input = audio_input
        else:
            fs = target_sr
        
        # 确保数据类型为 float32
        audio_input = audio_input.astype(np.float32)
        
        # 归一化到 [-1, 1]
        max_val = np.abs(audio_input).max()
        if max_val > 0:  # 避免除以0
            audio_input = audio_input / max_val
        
        # 转换为单声道
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=-1)
        
        # 重采样到目标采样率
        if fs != target_sr:
            audio_input = np.ascontiguousarray(audio_input)
            resampler = torchaudio.transforms.Resample(
                orig_freq=fs,
                new_freq=target_sr,
                dtype=torch.float32
            )
            audio_input = resampler(torch.from_numpy(audio_input)[None, :])[0].numpy()
        
        return audio_input

    @handle_errors(logger)
    async def _run_ffmpeg_command(self, cmd: List[str]) -> Tuple[bytes, bytes]:
        """执行 FFmpeg 命令"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"FFmpeg 命令执行失败: {error_msg}")
            
        return stdout, stderr

    @handle_errors(logger)
    async def extract_audio(self, video_path: str, output_path: str, start: float = 0, duration: float = None) -> str:
        """从视频中提取音频
        
        Args:
            video_path: 视频文件路径
            output_path: 输出音频文件路径
            start: 开始时间（秒）
            duration: 持续时间（秒），不指定则处理到文件结束
        """
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-ss', str(start)
        ]
        if duration is not None:
            cmd.extend(['-t', str(duration)])
        
        cmd.extend([
            '-vn',
            '-acodec', 'pcm_f32le',
            '-ac', '1',
            output_path
        ])
        
        await self._run_ffmpeg_command(cmd)
        return output_path

    @handle_errors(logger)
    async def extract_video(self, video_path: str, output_path: str, start: float = 0, duration: float = None) -> str:
        """提取无声视频
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频文件路径
            start: 开始时间（秒）
            duration: 持续时间（秒），不指定则处理到文件结束
        """
        # 如果是从头开始提取，直接使用 -an 去除音频
        if start == 0:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-an',  # 移除音频
                '-c:v', 'copy',  # 直接复制视频流，避免重新编码
                output_path
            ]
        else:
            # 如果需要从中间开始，先解码再编码以确保精确的时间点
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(start)
            ]
            if duration is not None:
                cmd.extend(['-t', str(duration)])
            
            cmd.extend([
                '-an',  # 移除音频
                '-c:v', 'libx264',  # 使用 x264 编码器
                '-preset', 'ultrafast',  # 使用最快的编码预设
                '-crf', '18',  # 使用较高质量的设置，因为这是中间片段
                '-tune', 'fastdecode',  # 优化解码速度
                output_path
            ])
        
        self.logger.debug(f"执行 FFmpeg 命令: {' '.join(cmd)}")
        await self._run_ffmpeg_command(cmd)
        return output_path

    @handle_errors(logger)
    async def get_audio_segments(self, duration: float) -> List[Tuple[float, float]]:
        """获取音频分段信息
        
        Args:
            duration: 音频时长（秒）
            
        Returns:
            List[Tuple[float, float]]: 分段列表，每个元素为 (开始时间, 持续时间)
        """
        segment_length = self.config.SEGMENT_MINUTES * 60
        min_length = self.config.MIN_SEGMENT_MINUTES * 60
        
        if duration <= min_length:
            return [(0, duration)]
        
        segments = []
        current_pos = 0
        
        while current_pos < duration:
            remaining_duration = duration - current_pos
            
            if remaining_duration <= segment_length:
                if remaining_duration < min_length and segments:
                    # 如果剩余时长小于最小分段长度且不是第一段，则并入前一段
                    start = segments[-1][0]
                    new_duration = duration - start
                    segments[-1] = (start, new_duration)
                else:
                    segments.append((current_pos, remaining_duration))
                break
            
            segments.append((current_pos, segment_length))
            current_pos += segment_length
        
        return segments

    @handle_errors(logger)
    async def extract_segment(self, video_path: str, start: float, duration: float, output_dir: Path, segment_index: int) -> Dict[str, Union[str, float]]:
        """从视频中提取并分离片段
        
        Args:
            video_path: 视频文件路径
            start: 开始时间（秒）
            duration: 持续时间（秒）
            output_dir: 输出目录
            segment_index: 分段索引
            
        Returns:
            Dict[str, Union[str, float]]: 包含分离后的媒体文件路径
        """
        temp_files = {}
        try:
            # 为当前分段创建输出路径
            silent_video = str(output_dir / f"video_silent_{segment_index}.mp4")
            full_audio = str(output_dir / f"audio_full_{segment_index}.wav")
            vocals_audio = str(output_dir / f"vocals_{segment_index}.wav")
            background_audio = str(output_dir / f"background_{segment_index}.wav")
            
            # 并行提取音频和无声视频
            await asyncio.gather(
                self.extract_audio(video_path, full_audio, start, duration),
                self.extract_video(video_path, silent_video, start, duration)
            )
            
            # 使用音频分离器处理音频
            vocals, background, sr = self.audio_separator.separate_audio(full_audio)
            
            # 重采样背景音频
            background = self.normalize_and_resample((sr, background), self.target_sr)
            
            # 保存人声和背景音频
            sf.write(vocals_audio, vocals, sr, subtype='FLOAT')
            sf.write(background_audio, background, self.target_sr, subtype='FLOAT')
            
            # 计算人声音频时长
            segment_duration = len(vocals) / sr
            
            # 删除原始音频文件
            Path(full_audio).unlink()
            
            # 将临时文件路径添加到返回字典
            temp_files = {
                'video': silent_video,
                'vocals': vocals_audio,
                'background': background_audio,
                'duration': segment_duration
            }
            
            return temp_files
            
        except Exception as e:
            # 发生错误时清理已创建的临时文件
            for file_path in temp_files.values():
                if isinstance(file_path, str) and Path(file_path).exists():
                    Path(file_path).unlink()
            raise

    @handle_errors(logger)
    async def get_video_duration(self, video_path: str) -> float:
        """获取视频时长
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            float: 视频时长（秒）
        """
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"获取视频时长失败: {stderr.decode()}")
        
        duration = float(stdout.decode().strip())
        return duration
