import logging
import asyncio
from typing import List, Any
from utils.worker_decorators import redis_worker_decorator
from utils.task_state import TaskState
from .model_in import ModelIn
from services.cosyvoice.client import CosyVoiceClient

logger = logging.getLogger(__name__)

class ModelInWorker:
    """
    模型输入 Worker：调用 ModelIn 对句子进行 speaker 特征更新、文本处理等。
    """

    def __init__(self, config):
        """初始化 ModelInWorker"""
        self.config = config
        self.logger = logger
        
        # 初始化 CosyVoiceClient
        cosyvoice_address = f"{config.COSYVOICE_SERVICE_HOST}:{config.COSYVOICE_SERVICE_PORT}"
        cosyvoice_client = CosyVoiceClient(address=cosyvoice_address)
        self.model_in = ModelIn(cosyvoice_client=cosyvoice_client)

    @redis_worker_decorator(
        input_queue='modelin_queue',
        next_queue='tts_token_queue',
        worker_name='模型输入 Worker',
        mode='stream',
        serialization_mode='msgpack'
    )
    async def run(self, item, task_state: TaskState):
        sentences_batch = item.get('data', item) if isinstance(item, dict) else item
        if not sentences_batch:
            return
        self.logger.debug(f"[模型输入 Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        async for processed_batch in self.model_in.modelin_maker(
            sentences_batch,
            reuse_speaker=False,
            batch_size=self.config.MODELIN_BATCH_SIZE
        ):
            yield processed_batch

async def start():
    """启动 Worker"""
    config_module = __import__('config')
    config = config_module.Config()
    worker = ModelInWorker(config)
    await worker.run()

if __name__ == '__main__':
    asyncio.run(start())