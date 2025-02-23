import logging
from typing import List, Any
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from .duration_aligner import DurationAligner
from workers.modelin_worker.model_in import ModelIn
from workers.tts_worker.tts_token_gener import TTSTokenGenerator
from workers.translation_worker.translation.translator import Translator
from workers.translation_worker.translation.deepseek_client import DeepSeekClient
from workers.translation_worker.translation.gemini_client import GeminiClient
from services.cosyvoice.client import CosyVoiceClient

logger = logging.getLogger(__name__)

class DurationWorker:
    """
    时长对齐 Worker：调用 DurationAligner 对句子进行时长调整。
    """

    def __init__(self, config):
        """初始化 DurationWorker"""
        self.config = config
        self.logger = logger
        
        # 初始化 CosyVoiceClient
        cosyvoice_address = f"{config.COSYVOICE_SERVICE_HOST}:{config.COSYVOICE_SERVICE_PORT}"
        cosyvoice_client = CosyVoiceClient(address=cosyvoice_address)
        
        # 初始化依赖组件
        model_in = ModelIn(cosyvoice_client=cosyvoice_client, max_concurrent_tasks=config.MAX_PARALLEL_SEGMENTS)
        tts_token_generator = TTSTokenGenerator(cosyvoice_client=cosyvoice_client)
        
        # 初始化翻译客户端
        translation_model = config.TRANSLATION_MODEL.lower()
        if translation_model == "deepseek":
            client = DeepSeekClient(api_key=config.DEEPSEEK_API_KEY)
        elif translation_model == "gemini":
            client = GeminiClient(api_key=config.GEMINI_API_KEY)
        else:
            raise ValueError(f"不支持的翻译模型: {translation_model}")
        simplifier = Translator(translation_client=client)
        
        # 直接初始化 DurationAligner
        self.duration_aligner = DurationAligner(
            model_in=model_in,
            simplifier=simplifier,
            tts_token_gener=tts_token_generator,
            max_speed=1.2
        )

    @worker_decorator(
        input_queue_attr='duration_align_queue',
        next_queue_attr='audio_gen_queue',
        worker_name='时长对齐 Worker'
    )
    async def run(self, sentences_batch: List[Any], task_state: TaskState):
        if not sentences_batch:
            return
        self.logger.debug(f"[时长对齐 Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.duration_aligner.align_durations(sentences_batch)
        return sentences_batch

if __name__ == '__main__':
    print("Duration Worker 模块加载成功")