import ray
import sys
import logging
from config import Config
from ray import serve

@serve.deployment(
    name="asr_model"
)
class ASRModel:
    """
    ASR模型，负责语音识别
    """
    def __init__(self):
        """初始化ASR模型"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化ASR模型Actor")
        self.config = Config()
        
        # 添加系统路径
        for path in self.config.SYSTEM_PATHS:
            if path not in sys.path:
                sys.path.append(path)
                self.logger.info(f"添加系统路径: {path}")
        
        try:
            # 导入并初始化ASR模型
            from core.auto_sense import SenseAutoModel
            
            # 直接使用硬编码参数，与原始代码保持一致
            self.model = SenseAutoModel(
                config=self.config,
                model="iic/SenseVoiceSmall",
                remote_code="./models/SenseVoice/model.py",
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                vad_kwargs={"max_single_segment_time": 30000},
                spk_model="cam++",
                trust_remote_code=True,
                disable_update=True,
                device="cuda"
            )
            
            self.logger.info("ASR模型加载完成")
        except Exception as e:
            self.logger.error(f"ASR模型加载失败: {str(e)}")
            raise
    
    async def generate_async(self, input, **kwargs):
        """
        执行与原始SenseAutoModel.generate_async相同的方法
        这是ASR的主要接口方法
        """
        try:
            self.logger.info(f"开始ASR识别音频: {input if isinstance(input, str) else '(已加载音频)'}")
            # 直接调用model的原始方法
            result = await self.model.generate_async(input, **kwargs)
            self.logger.info(f"ASR识别完成，获得 {len(result)} 个句子")
            return result
        except Exception as e:
            self.logger.error(f"ASR识别失败: {str(e)}")
            raise 