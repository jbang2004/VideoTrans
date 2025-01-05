from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from models.ClearerVoice.clearvoice import ClearVoice

class AudioSeparator(ABC):
    """音频分离器接口"""
    @abstractmethod
    def separate_audio(self, input_path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        分离音频为语音和背景音
        Args:
            input_path: 输入音频路径
            **kwargs: 额外参数
        Returns:
            Tuple[语音数组, 背景音数组]
        """
        pass

class ClearVoiceSeparator(AudioSeparator):
    """使用 ClearVoice 实现的音频分离器"""
    def __init__(self, model_name: str = 'MossFormer2_SE_48K'):
        self.model_name = model_name
        self.clearvoice = ClearVoice(
            task='speech_enhancement',
            model_names=[model_name]
        )
    
    def separate_audio(self, input_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        分离音频为语音和背景音
        Returns:
            Tuple[增强语音数据, 背景音数据, 采样率]
        """
        # 使用 ClearVoice 处理音频
        enhanced_audio, background_audio = self.clearvoice(
            input_path=input_path,
            online_write=False,
            extract_noise=True
        )
        
        # 根据模型名称确定采样率
        if self.model_name.endswith('16K'):
            sr = 16000
        elif self.model_name.endswith('48K'):
            sr = 48000
        else:
            # 默认采样率
            sr = 48000
        
        return enhanced_audio, background_audio, sr