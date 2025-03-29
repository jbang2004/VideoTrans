import ray
import sys
import logging
import asyncio  # 添加 asyncio 导入
from config import Config
from ray import serve

@serve.deployment(
    name="asr_model",
    health_check_timeout_s=120,  # 将健康检查超时时间从默认的30秒增加到120秒
    health_check_period_s=30     # 将健康检查周期从默认的10秒增加到30秒
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
    
    # 改为异步方法
    async def generate(self, input, **kwargs):
        """
        执行模型生成方法（异步版本）
        """
        try:
            self.logger.info(f"开始ASR识别音频: {input if isinstance(input, str) else '(已加载音频)'}")
            # 使用 asyncio.to_thread 包装同步调用
            result = await asyncio.to_thread(self.model.generate, input, **kwargs)
            self.logger.info(f"ASR识别完成，获得 {len(result)} 个句子")
            return result
        except Exception as e:
            self.logger.error(f"ASR识别失败: {str(e)}")
            raise 