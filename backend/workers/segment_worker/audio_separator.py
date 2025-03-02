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
    
    async def separate(self, input_path: str, vocals_output: str, background_output: str) -> None:
        """
        分离音频，并将人声和背景音保存到指定路径
        
        Args:
            input_path: 输入音频路径
            vocals_output: 人声输出路径
            background_output: 背景音输出路径
        """
        import soundfile as sf
        
        # 分离音频
        enhanced_audio, background_audio, sr = self.separate_audio(input_path)
        
        # 保存音频
        sf.write(vocals_output, enhanced_audio, sr)
        sf.write(background_output, background_audio, sr)
