from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from models.ClearerVoice.clearvoice import ClearVoice

class AudioSeparator(ABC):
    """音频分离器接口"""
    @abstractmethod
    def separate_audio(self, input_path: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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
