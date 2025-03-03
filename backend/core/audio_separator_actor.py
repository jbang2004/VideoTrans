import ray
import sys
import logging
import numpy as np
from typing import Tuple

@ray.remote(num_gpus=0.1)  # 分配较少的GPU资源
class AudioSeparatorActor:
    """音频分离器Actor封装"""
    
    def __init__(self, model_name: str = 'MossFormer2_SE_48K'):
        """初始化音频分离器Actor"""
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化音频分离器Actor: {model_name}")
        
        # 导入Config并设置系统路径
        from config import Config
        self.config = Config()
        
        # 添加系统路径
        for path in self.config.SYSTEM_PATHS:
            if path not in sys.path:
                sys.path.append(path)
                self.logger.info(f"添加系统路径: {path}")
        
        try:
            # 导入并初始化音频分离模型
            from models.ClearerVoice.clearvoice import ClearVoice
            
            self.model_name = model_name
            self.clearvoice = ClearVoice(
                task='speech_enhancement',
                model_names=[model_name]
            )
            
            self.logger.info("音频分离模型加载完成")
        except Exception as e:
            self.logger.error(f"音频分离模型加载失败: {str(e)}")
            raise
    
    async def separate_audio(self, input_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """执行音频分离"""
        try:
            self.logger.info(f"开始分离音频: {input_path}")
            
            # 在异步方法中使用run_sync包装同步操作
            def do_separate():
                return self.clearvoice(
                    input_path=input_path,
                    online_write=False,
                    extract_noise=True
                )
            
            # 使用Ray的concurrency.run_sync或asyncio.to_thread
            from utils import concurrency
            enhanced_audio, background_audio = await concurrency.run_sync(do_separate)
            
            # 根据模型名称确定采样率
            if self.model_name.endswith('16K'):
                sr = 16000
            elif self.model_name.endswith('48K'):
                sr = 48000
            else:
                sr = 48000
                
            self.logger.info(f"音频分离完成: {input_path}")
            return enhanced_audio, background_audio, sr
            
        except Exception as e:
            self.logger.error(f"音频分离失败: {str(e)}")
            raise

# 只保留工厂函数，移除默认实例创建
def create_audio_separator(num_gpus=0.1, model_name='MossFormer2_SE_48K'):
    """创建音频分离器Actor实例"""
    return AudioSeparatorActor.options(
        num_gpus=num_gpus,
        name="audio_separator"
    ).remote(model_name) 