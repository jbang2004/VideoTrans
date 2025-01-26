import logging
import asyncio
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Union
from utils.decorators import handle_errors
from utils import concurrency

logger = logging.getLogger(__name__)

class MediaUtils:
    def __init__(self, config, audio_separator, target_sr: int = 24000):
        self.config = config
        self.target_sr = target_sr
        self.logger = logging.getLogger(__name__)
        # audio_separator应为实现了 async 方法的实例 (ClearVoiceSeparator等)
        self.audio_separator = audio_separator

    def normalize_and_resample(
        self,
        audio_input: Union[Tuple[int, np.ndarray], np.ndarray],
        target_sr: int = None
    ) -> np.ndarray:
        """
        同步方式的重采样和归一化。
        若音频比较大，建议用 asyncio.to_thread(...) 包装本函数，以防阻塞事件循环。
        """
        if isinstance(audio_input, tuple):
            fs, audio_input = audio_input
        else:
            fs = target_sr
        
        audio_input = audio_input.astype(np.float32)
        
        max_val = np.abs(audio_input).max()
        if max_val > 0:
            audio_input = audio_input / max_val
        
        # 如果多通道, 转单通道
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=-1)
        
        # 如果源采样率与目标采样率不一致, 用torchaudio进行重采样
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
        """
        启动ffmpeg子进程并异步等待执行完毕，不会阻塞事件循环
        """
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
    async def extract_audio(
        self,
        video_path: str,
        output_path: str,
        start: float = 0,
        duration: float = None
    ) -> str:
        """
        调用 FFmpeg 提取音频 (异步)
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
    async def extract_video(
        self,
        video_path: str,
        output_path: str,
        start: float = 0,
        duration: float = None
    ) -> str:
        """
        调用 FFmpeg 提取无音视频 (异步)
        """
        if start == 0:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-an',
                '-c:v', 'copy',
                output_path
            ]
        else:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(start)
            ]
            if duration is not None:
                cmd.extend(['-t', str(duration)])
            
            cmd.extend([
                '-an',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '18',
                '-tune', 'fastdecode',
                output_path
            ])
        
        self.logger.debug(f"执行 FFmpeg 命令: {' '.join(cmd)}")
        await self._run_ffmpeg_command(cmd)
        return output_path

    @handle_errors(logger)
    async def get_video_duration(self, video_path: str) -> float:
        """
        用 ffprobe 查询视频时长 (异步)
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

    @handle_errors(logger)
    async def get_audio_segments(self, duration: float) -> List[Tuple[float, float]]:
        """
        按照配置中的SEGMENT_MINUTES分割时间片。仅仅是一些轻量级计算，不会阻塞。
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
    async def extract_segment(
        self,
        video_path: str,
        start: float,
        duration: float,
        output_dir: Path,
        segment_index: int
    ) -> Dict[str, Union[str, float]]:
        """
        1) 提取纯视频 + 音频
        2) 调用 audio_separator 分离人声/背景
        3) 重采样+写音频文件
        4) 返回分段文件信息 (video, vocals, background, duration)
        """
        temp_files = {}
        try:
            silent_video = str(output_dir / f"video_silent_{segment_index}.mp4")
            full_audio = str(output_dir / f"audio_full_{segment_index}.wav")
            vocals_audio = str(output_dir / f"vocals_{segment_index}.wav")
            background_audio = str(output_dir / f"background_{segment_index}.wav")
            
            # (1) 并发提取音频 & 视频
            await asyncio.gather(
                self.extract_audio(video_path, full_audio, start, duration),
                self.extract_video(video_path, silent_video, start, duration)
            )
            
            # [MODIFIED] 用 concurrency 代替 to_thread
            # step (2) 分离人声
            def do_separate():
                return self.audio_separator.separate_audio(full_audio)
            vocals, background, sr = await concurrency.run_sync(do_separate)

            # step (3) 重采样
            def do_resample():
                return self.normalize_and_resample((sr, background), self.target_sr)
            background = await concurrency.run_sync(do_resample)

            # 写文件
            def write_vocals():
                sf.write(vocals_audio, vocals, sr, subtype='FLOAT')

            def write_bg():
                sf.write(background_audio, background, self.target_sr, subtype='FLOAT')

            await concurrency.run_sync(write_vocals)
            await concurrency.run_sync(write_bg)
                       
            segment_duration = len(vocals) / sr

            # optional: 删除完整音频
            Path(full_audio).unlink(missing_ok=True)

            temp_files = {
                'video': silent_video,
                'vocals': vocals_audio,
                'background': background_audio,
                'duration': segment_duration
            }
            return temp_files
            
        except Exception as e:
            # 清理已生成的临时文件
            for file_path in temp_files.values():
                if isinstance(file_path, str) and Path(file_path).exists():
                    Path(file_path).unlink()
            raise
