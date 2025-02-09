import logging
from typing import List, Any
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from core.timeadjust.duration_aligner import DurationAligner
from models.model_manager import ModelManager

logger = logging.getLogger(__name__)

class DurationWorker:
    """
    时长对齐 Worker：调用 DurationAligner 对句子进行时长调整。
    """

    def __init__(self, config):
        """初始化 DurationWorker"""
        self.config = config
        self.logger = logger
        
        # 获取模型管理器实例
        model_manager = ModelManager()
        model_manager.initialize_models(config)
        
        # 获取时长对齐器实例
        self.duration_aligner = model_manager.duration_aligner

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
