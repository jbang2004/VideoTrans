import ray
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from models.ClearerVoice.clearvoice import ClearVoice

class AudioSeparator(ABC):
    """音频分离器接口"""
    @abstractmethod
    def separate_audio(self, input_path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        pass

@ray.remote
class ClearVoiceActor:
    """
    ClearVoice音频分离器的Ray Actor实现
    使用Ray Actor封装ClearVoice模型，避免重复加载模型
    """
    def __init__(self, model_name: str = 'MossFormer2_SE_48K'):
        """
        初始化ClearVoice Actor
        
        Args:
            model_name: 使用的ClearVoice模型名称
        """
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