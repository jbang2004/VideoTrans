import logging
from typing import List, Any
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from .tts_token_gener import TTSTokenGenerator
from services.cosyvoice.client import CosyVoiceClient

logger = logging.getLogger(__name__)

class TTSTokenWorker:
    """
    TTS Token 生成 Worker：调用 TTSTokenGenerator 为句子生成 TTS token。
    """

    def __init__(self, config):
        """初始化 TTSTokenWorker"""
        self.config = config
        self.logger = logger
        
        # 初始化 CosyVoiceClient
        cosyvoice_address = f"{config.COSYVOICE_SERVICE_HOST}:{config.COSYVOICE_SERVICE_PORT}"
        cosyvoice_client = CosyVoiceClient(address=cosyvoice_address)
        self.tts_token_generator = TTSTokenGenerator(cosyvoice_client=cosyvoice_client)

    @worker_decorator(
        input_queue_attr='tts_token_queue',
        next_queue_attr='duration_align_queue',
        worker_name='TTS Token生成 Worker'
    )
    async def run(self, sentences_batch: List[Any], task_state: TaskState):
        if not sentences_batch:
            return
        self.logger.debug(f"[TTS Token生成 Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.tts_token_generator.tts_token_maker(sentences_batch)
        return sentences_batch

if __name__ == '__main__':
    print("TTS Token Worker 模块加载成功")