# -----------------------------------------
# backend/pipeline_scheduler.py (节选完整示例)
# -----------------------------------------
import asyncio
import logging
from utils.task_state import TaskState

logger = logging.getLogger(__name__)

class PipelineScheduler:
    """
    流水线调度器：负责协调各个 Worker 阶段，通过各个队列传递数据，
    启动所有 worker 长循环任务，然后在停止时发送终止信号。
    """

    def __init__(
        self,
        segment_worker,    # workers.segment_worker.Worker 实例
        asr_worker,        # workers.asr_worker.Worker 实例
        translation_worker,# workers.translation_worker.Worker 实例
        modelin_worker,    # workers.modelin_worker.Worker 实例
        tts_token_worker,  # workers.tts_worker.Worker 实例
        duration_worker,   # workers.duration_worker.Worker 实例
        audio_gen_worker,  # workers.audio_gen_worker.Worker 实例
        mixer_worker,      # workers.mixer_worker.Worker 实例
        config
    ):
        self.logger = logging.getLogger(__name__)
        self.segment_worker = segment_worker
        self.asr_worker = asr_worker
        self.translation_worker = translation_worker
        self.modelin_worker = modelin_worker
        self.tts_token_worker = tts_token_worker
        self.duration_worker = duration_worker
        self.audio_gen_worker = audio_gen_worker
        self.mixer_worker = mixer_worker
        self.config = config
        self._workers = []

    async def start_workers(self, task_state: TaskState):
        self.logger.info(f"[PipelineScheduler] 启动所有 Worker -> TaskID={task_state.task_id}")
        self._workers = [
            asyncio.create_task(self.segment_worker.run_init(task_state)),
            asyncio.create_task(self.segment_worker.run_extract(task_state)),
            asyncio.create_task(self.asr_worker.run(task_state)),
            asyncio.create_task(self.translation_worker.run(task_state)),
            asyncio.create_task(self.modelin_worker.run(task_state)),
            asyncio.create_task(self.tts_token_worker.run(task_state)),
            asyncio.create_task(self.duration_worker.run(task_state)),
            asyncio.create_task(self.audio_gen_worker.run(task_state)),
            asyncio.create_task(self.mixer_worker.run(task_state))
        ]

    async def stop_workers(self, task_state: TaskState):
        self.logger.info(f"[PipelineScheduler] 停止所有 Worker -> TaskID={task_state.task_id}")
        # 发送终止信号（在最上游队列中发送 None）
        await task_state.segment_init_queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self.logger.info(f"[PipelineScheduler] 所有 Worker 已结束 -> TaskID={task_state.task_id}")
