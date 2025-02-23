import logging
from typing import List, Any
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from .audio_gener import AudioGenerator
from .timestamp_adjuster import TimestampAdjuster
from services.cosyvoice.client import CosyVoiceClient


logger = logging.getLogger(__name__)

class AudioGenWorker:
    """
    音频生成 Worker：利用 AudioGenerator 对句子生成合成音频。
    """

    def __init__(self, config):
        """初始化 AudioGenWorker"""
        self.config = config
        self.logger = logger
        
        # 初始化 CosyVoiceClient
        cosyvoice_address = f"{config.COSYVOICE_SERVICE_HOST}:{config.COSYVOICE_SERVICE_PORT}"
        cosyvoice_client = CosyVoiceClient(address=cosyvoice_address)
        self.audio_generator = AudioGenerator(
            cosyvoice_client=cosyvoice_client,
            sample_rate=config.SAMPLE_RATE
        )
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