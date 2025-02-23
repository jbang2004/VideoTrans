import logging
from typing import List, Any
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from core.tts_token_gener import TTSTokenGenerator
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class TTSTokenWorker:
    """
    TTS Token 生成 Worker：调用 TTSTokenGenerator 为句子生成 TTS token。
    """

    def __init__(self, config):
        """初始化 TTSTokenWorker"""
        self.config = config
        self.logger = logger
        
        # 获取模型管理器实例
        model_manager = ModelManager()
        model_manager.initialize_models(config)
        
        # 获取 TTS Token 生成器实例
        self.tts_token_generator = model_manager.tts_generator

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
