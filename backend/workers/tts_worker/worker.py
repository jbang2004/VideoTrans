import logging
import asyncio
from typing import List, Any
from utils.redis_decorators import redis_worker_decorator
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

    @redis_worker_decorator(
        input_queue='tts_token_queue',
        next_queue='duration_align_queue',
        worker_name='TTS Token生成 Worker'
    )
    async def run(self, item, task_state: TaskState):
        sentences_batch = item.get('data', item)  # 兼容直接传入数据或包含data字段的情况
        if not sentences_batch:
            return
        self.logger.debug(f"[TTS Token生成 Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.tts_token_generator.tts_token_maker(sentences_batch)
        return sentences_batch

async def start():
    """启动 Worker"""
    config_module = __import__('config')
    config = config_module.Config()
    worker = TTSTokenWorker(config)
    await worker.run()

if __name__ == '__main__':
    asyncio.run(start())