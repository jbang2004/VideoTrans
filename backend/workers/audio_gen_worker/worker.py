import logging
from typing import List, Any
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from core.audio_gener import AudioGenerator
from models.model_manager import ModelManager
from .timestamp_adjuster import TimestampAdjuster

logger = logging.getLogger(__name__)

class AudioGenWorker:
    """
    音频生成 Worker：利用 AudioGenerator 对句子生成合成音频。
    """

    def __init__(self, config):
        """初始化 AudioGenWorker"""
        self.config = config
        self.logger = logger
        
        # 获取模型管理器实例
        model_manager = ModelManager()
        model_manager.initialize_models(config)
        
        # 获取音频生成器实例
        self.audio_generator = model_manager.audio_generator
        
        # 直接实例化时间戳调整器
        self.timestamp_adjuster = TimestampAdjuster(config=config)

    @worker_decorator(
        input_queue_attr='audio_gen_queue',
        next_queue_attr='mixing_queue',
        worker_name='音频生成 Worker'
    )
    async def run(self, sentences_batch: List[Any], task_state: TaskState):
        if not sentences_batch:
            return
        self.logger.debug(f"[音频生成 Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.audio_generator.vocal_audio_maker(sentences_batch)
        task_state.current_time = self.timestamp_adjuster.update_timestamps(sentences_batch, start_time=task_state.current_time)
        valid = self.timestamp_adjuster.validate_timestamps(sentences_batch)
        if not valid:
            self.logger.warning(f"[音频生成 Worker] 检测到时间戳不连续, TaskID={task_state.task_id}")
        return sentences_batch

if __name__ == '__main__':
    print("Audio Generation Worker 模块加载成功")
