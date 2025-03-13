import ray
import logging
import torch
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from config import Config

from models.ClearerVoice.clearvoice import ClearVoice

class AudioSeparator(ABC):
    """音频分离器接口"""
    @abstractmethod
    def separate_audio(self, input_path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

@ray.remote(num_gpus=Config().CLEARVOICE_ACTOR_NUM_GPUS)
class ClearVoiceActor:
    """
    音频分离Actor，负责分离人声和背景音乐
    """
    def __init__(self, model_name='MossFormer2_SE_48K'):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化音频分离Actor: {model_name}")
        
        self.model_name = model_name
        self.clearvoice = ClearVoice(
            task='speech_enhancement',
            model_names=[model_name]
        )
    
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