# utils/media_utils.py
import logging
import asyncio
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
import ray
from typing import List, Tuple, Dict, Union, Optional
import time

# 引入统一的 FFmpegTool
from utils.ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class MediaUtils:
    def __init__(self, config, audio_separator_actor, target_sr: int = 24000):
        """
        初始化MediaUtils
        
        Args:
            config: 配置对象
            audio_separator_actor: ClearVoiceActor引用
            target_sr: 目标采样率
        """
        self.config = config
        self.target_sr = target_sr
        self.logger = logging.getLogger(__name__)
        self.audio_separator_actor = audio_separator_actor

        # 新增 ffmpeg 工具类
        self.ffmpeg_tool = FFmpegTool()

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

        # 如果源采样率与目标采样率不一致, 用 torchaudio 进行重采样
        if fs != target_sr:
            audio_input = np.ascontiguousarray(audio_input)
            resampler = torchaudio.transforms.Resample(
                orig_freq=fs,
                new_freq=target_sr,
                dtype=torch.float32
            )
            audio_input = resampler(torch.from_numpy(audio_input)[None, :])[0].numpy()

        return audio_input

    async def get_video_duration(self, video_path: str) -> float:
        """
        用 ffprobe 查询视频时长 (异步)，统一改用 FFmpegTool。
        """
        start_time = time.time()
        try:
            duration = await self.ffmpeg_tool.get_duration(video_path)
            elapsed = time.time() - start_time
            logger.debug(f"get_video_duration 完成，耗时 {elapsed:.2f}s")
            return duration
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"get_video_duration 执行出错，耗时 {elapsed:.2f}s, 错误: {e}")
            raise

    async def get_audio_segments(self, duration: float) -> List[Tuple[float, float]]:
        """
        按照配置中的 SEGMENT_MINUTES 分割时间片。仅做一些计算，不会阻塞。
        """
        start_time = time.time()
        try:
            segment_length = self.config.SEGMENT_MINUTES * 60
            min_length = self.config.MIN_SEGMENT_MINUTES * 60

            if duration <= min_length:
                return [(0, duration)]

            segments = []
            current_pos = 0.0

            while current_pos < duration:
                remaining_duration = duration - current_pos

                if remaining_duration <= segment_length:
                    # 如果剩余片段过短且已有片段，和前一个合并
                    if remaining_duration < min_length and segments:
                        start = segments[-1][0]
                        new_duration = duration - start
                        segments[-1] = (start, new_duration)
                    else:
                        segments.append((current_pos, remaining_duration))
                    break

                segments.append((current_pos, segment_length))
                current_pos += segment_length

            elapsed = time.time() - start_time
            logger.debug(f"get_audio_segments 完成，耗时 {elapsed:.2f}s")
            return segments
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"get_audio_segments 执行出错，耗时 {elapsed:.2f}s, 错误: {e}")
            raise

    async def extract_segment(
        self,
        video_path: str,
        start: float,
        duration: float,
        output_dir: Path,
        segment_index: int
    ) -> Dict[str, Union[str, float]]:
        """
        提取视频片段，分离人声和背景音乐。
        返回临时文件路径字典：
        {
            'video': 无声视频路径,
            'vocals': 人声音频路径,
            'background': 背景音乐路径,
            'duration': 实际片段时长
        }
        """
        start_time = time.time()
        temp_files = {}
        try:
            # 创建临时目录
            output_dir.mkdir(parents=True, exist_ok=True)
            
            silent_video = str(output_dir / f"video_silent_{segment_index}.mp4")
            full_audio = str(output_dir / f"audio_full_{segment_index}.wav")
            vocals_audio = str(output_dir / f"vocals_{segment_index}.wav")
            background_audio = str(output_dir / f"background_{segment_index}.wav")

            # (1) 并发提取音频 & 视频
            await asyncio.gather(
                self.ffmpeg_tool.extract_audio(video_path, full_audio, start, duration),
                self.ffmpeg_tool.extract_video(video_path, silent_video, start, duration)
            )

            # (2) 分离人声 - 使用Actor
            vocals, background, sr = await self.audio_separator_actor.separate_audio.remote(full_audio)

            # (3) 重采样
            def do_resample_bg():
                return self.normalize_and_resample((sr, background), self.target_sr)

            background = await asyncio.to_thread(do_resample_bg)

            # 写入人声/背景音频
            def write_vocals():
                sf.write(vocals_audio, vocals, sr, subtype='FLOAT')

            def write_bg():
                sf.write(background_audio, background, self.target_sr, subtype='FLOAT')

            await asyncio.to_thread(write_vocals)
            await asyncio.to_thread(write_bg)

            segment_duration = len(vocals) / sr

            # optional: 删除原始整段音频
            Path(full_audio).unlink(missing_ok=True)

            temp_files = {
                'video': silent_video,
                'vocals': vocals_audio,
                'background': background_audio,
                'duration': segment_duration
            }

            # 返回临时文件路径
            elapsed = time.time() - start_time
            logger.debug(f"extract_segment 完成，耗时 {elapsed:.2f}s")
            return temp_files
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"extract_segment 执行出错，耗时 {elapsed:.2f}s, 错误: {e}")
            
            # 清理已生成的临时文件
            for file_path in temp_files.values():
                if isinstance(file_path, str) and Path(file_path).exists():
                    Path(file_path).unlink()
            
            raise
