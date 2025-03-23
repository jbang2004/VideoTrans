import ray
from ray import serve
import logging
import torch
import numpy as np
import soundfile as sf
from typing import Tuple, Dict, Union, Optional
from pathlib import Path
import time
from config import Config

from models.ClearerVoice.clearvoice import ClearVoice
from utils.ffmpeg_utils import extract_audio, extract_video

@serve.deployment(
    name="video_separator"
)
class VideoSeparator:
    """
    视频分离器，负责分割视频片段、提取音频并分离人声和背景音乐
    """
    def __init__(self, model_name='MossFormer2_SE_48K'):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化视频分离器: {model_name}")
        
        self.model_name = model_name
        self.clearvoice = ClearVoice(
            task='speech_enhancement',
            model_names=[model_name]
        )
        self.config = Config()
    
    def separate_audio(self, input_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        分离音频中的人声和背景音
        
        Args:
            input_path: 输入音频文件路径
            
        Returns:
            Tuple[np.ndarray, np.ndarray, int]: (人声音频, 背景音频, 采样率)
        """
        enhanced_audio, background_audio = self.clearvoice(
            input_path=input_path,
            online_write=False,
            extract_noise=True
        )
        
        if self.model_name.endswith('16K'):
            sr = 16000
        elif self.model_name.endswith('48K'):
            sr = 48000
        else:
            sr = 48000
        
        return enhanced_audio, background_audio, sr
    
    async def separate_video(
        self,
        video_path: str,
        start: float,
        output_dir: str,
        segment_index: int,
        target_sr: int,
        duration: float = None
    ) -> Dict[str, Union[str, float]]:
        """
        提取视频片段，分离人声和背景音乐
        
        Args:
            video_path: 视频文件路径
            start: 开始时间（秒）
            output_dir: 输出目录
            segment_index: 分段索引
            target_sr: 目标采样率
            duration: 分段持续时间（可选）
            
        Returns:
            Dict[str, Union[str, float]]: 包含分离后文件路径的字典
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
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            
            silent_video = str(output_dir_path / f"video_silent_{segment_index}.mp4")
            full_audio = str(output_dir_path / f"audio_full_{segment_index}.wav")
            vocals_audio = str(output_dir_path / f"vocals_{segment_index}.wav")
            background_audio = str(output_dir_path / f"background_{segment_index}.wav")

            # (1) 提取音频 & 视频
            await extract_audio.remote(video_path, full_audio, start, duration)
            await extract_video.remote(video_path, silent_video, start, duration)

            # (2) 分离人声
            vocals, background, sr = self.separate_audio(full_audio)

            # (3) 重采样和归一化
            background = self._normalize_and_resample((sr, background), target_sr)

            # 写入人声/背景音频
            sf.write(vocals_audio, vocals, sr, subtype='FLOAT')
            sf.write(background_audio, background, target_sr, subtype='FLOAT')

            segment_duration = len(vocals) / sr

            # 删除原始整段音频
            Path(full_audio).unlink(missing_ok=True)

            temp_files = {
                'video': silent_video,
                'vocals': vocals_audio,
                'background': background_audio,
                'duration': segment_duration
            }

            # 返回临时文件路径
            elapsed = time.time() - start_time
            self.logger.debug(f"separate_video 完成，耗时 {elapsed:.2f}s，segment_index={segment_index}")
            return temp_files
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"separate_video 执行出错，耗时 {elapsed:.2f}s, 错误: {e}, segment_index={segment_index}")
            
            # 清理已生成的临时文件
            for file_path in temp_files.values():
                if isinstance(file_path, str) and Path(file_path).exists():
                    Path(file_path).unlink()
            
            raise
    
    def _normalize_and_resample(
        self,
        audio_input: Union[Tuple[int, np.ndarray], np.ndarray],
        target_sr: int = None
    ) -> np.ndarray:
        """
        重采样和归一化音频
        
        Args:
            audio_input: 音频数据，可以是元组(采样率, 音频数据)或者直接是音频数据
            target_sr: 目标采样率
        
        Returns:
            np.ndarray: 处理后的音频数据
        """
        import torch
        import torchaudio
        
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