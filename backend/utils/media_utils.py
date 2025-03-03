# utils/media_utils.py
import logging
import asyncio
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from utils.decorators import handle_errors
from utils import concurrency
import ray

# 引入统一的 FFmpegTool
from utils.ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class MediaUtils:
    """媒体工具类，处理视频、音频的提取和处理"""
    
    def __init__(self, config, audio_separator_actor=None, target_sr=None):
        """初始化
        
        Args:
            config: 配置对象
            audio_separator_actor: 音频分离Actor引用
            target_sr: 目标采样率
        """
        self.config = config
        self.audio_separator_actor = audio_separator_actor  
        self.target_sr = target_sr
        self.ffmpeg_tool = FFmpegTool()
        self.logger = logging.getLogger(__name__)

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

    @handle_errors(logger)
    async def get_video_segments(self, video_path: str) -> List[Tuple[float, float]]:
        """
        获取视频时长并计算分段信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[Tuple[float, float]]: 分段列表，每个元素为(开始时间, 持续时间)
        """
        # 1. 获取视频时长
        duration = await self.ffmpeg_tool.get_duration(video_path)
        self.logger.info(f"视频时长: {duration:.2f}秒")
        
        # 2. 计算分段
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
            
        self.logger.info(f"视频分为 {len(segments)} 个分段")
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
        提取视频片段并分离人声/背景
        
        1) 提取纯视频 + 音频
        2) 调用 audio_separator 分离人声/背景
        3) 重采样 + 写音频文件
        4) 返回分段文件信息 (video, vocals, background, duration)
        """
        temp_files = {}
        try:
            # 准备文件路径
            silent_video = str(output_dir / f"video_silent_{segment_index}.mp4")
            full_audio = str(output_dir / f"audio_full_{segment_index}.wav")
            vocals_audio = str(output_dir / f"vocals_{segment_index}.wav")
            background_audio_path = str(output_dir / f"background_{segment_index}.wav")

            self.logger.info(f"开始处理视频片段: {segment_index}, 起始={start:.2f}s, 持续={duration:.2f}s")
            
            # (1) 并发提取音频 & 视频 - 高效利用异步
            await asyncio.gather(
                self.ffmpeg_tool.extract_audio(video_path, full_audio, start, duration),
                self.ffmpeg_tool.extract_video(video_path, silent_video, start, duration)
            )

            # (2) 使用Actor进行音频分离
            if self.audio_separator_actor:
                self.logger.info(f"使用Actor分离音频: {full_audio}")
                # 使用更清晰的直接await语法
                enhanced_audio, background_audio, sep_sr = await self.audio_separator_actor.separate_audio.remote(str(full_audio))
                
                # 如果需要重采样到target_sr
                if self.target_sr and sep_sr != self.target_sr:
                    background_audio = self.normalize_and_resample((sep_sr, background_audio), self.target_sr)

                # 写入人声/背景音频 - 并发写入
                async def write_audio_files():
                    await asyncio.gather(
                        concurrency.run_sync(lambda: sf.write(vocals_audio, enhanced_audio, sep_sr, subtype='FLOAT')),
                        concurrency.run_sync(lambda: sf.write(background_audio_path, background_audio, 
                                                           self.target_sr or sep_sr, subtype='FLOAT'))
                    )
                
                await write_audio_files()
                segment_duration = len(enhanced_audio) / sep_sr

                # 删除原始整段音频，节省空间
                Path(full_audio).unlink(missing_ok=True)

                temp_files = {
                    'video': silent_video,
                    'vocals': vocals_audio,
                    'background': background_audio_path,
                    'duration': segment_duration
                }
                
                self.logger.info(f"视频片段处理完成: 索引={segment_index}, 实际持续时间={segment_duration:.2f}s")
                return temp_files
            else:
                self.logger.warning("未提供音频分离Actor，无法分离音频")
                raise ValueError("缺少音频分离Actor")

        except Exception as e:
            self.logger.error(f"视频片段处理失败: {str(e)}")
            
            # 清理已生成的临时文件
            for file_path in temp_files.values():
                if isinstance(file_path, str) and Path(file_path).exists():
                    Path(file_path).unlink()
            raise
