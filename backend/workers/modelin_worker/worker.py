import logging
from typing import List, Any
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from core.model_in import ModelIn
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class ModelInWorker:
    """
    模型输入 Worker：调用 ModelIn 对句子进行 speaker 特征更新、文本处理等。
    """

    def __init__(self, config):
        """初始化 ModelInWorker"""
        self.config = config
        self.logger = logger
        
        # 获取模型管理器实例
        model_manager = ModelManager()
        model_manager.initialize_models(config)
        
        # 获取 ModelIn 实例
        self.model_in = model_manager.model_in

    @worker_decorator(
        input_queue_attr='modelin_queue',
        next_queue_attr='tts_token_queue',
        worker_name='模型输入 Worker',
        mode='stream'
    )
    async def run(self, sentences_batch: List[Any], task_state: TaskState):
        if not sentences_batch:
            return
        self.logger.debug(f"[模型输入 Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        async for updated_batch in self.model_in.modelin_maker(
            sentences_batch,
            reuse_speaker=False,
            batch_size=self.config.MODELIN_BATCH_SIZE
        ):
            yield updated_batch

if __name__ == '__main__':
    print("ModelIn Worker 模块加载成功")
