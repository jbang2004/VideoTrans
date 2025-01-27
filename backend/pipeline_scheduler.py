# -----------------------------------------
# backend/pipeline_scheduler.py (节选完整示例)
# -----------------------------------------
import asyncio
import logging
from typing import List
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from core.sentence_tools import Sentence

logger = logging.getLogger(__name__)

class PipelineScheduler:
    """
    多个Worker的调度器，负责翻译->model_in->tts_token->时长对齐->音频生成->混音 ...
    """

    def __init__(
        self,
        translator,
        model_in,
        tts_token_generator,
        duration_aligner,
        audio_generator,
        timestamp_adjuster,
        mixer,
        config
    ):
        self.logger = logging.getLogger(__name__)
        self.translator = translator
        self.model_in = model_in
        self.tts_token_generator = tts_token_generator
        self.duration_aligner = duration_aligner
        self.audio_generator = audio_generator
        self.timestamp_adjuster = timestamp_adjuster
        self.mixer = mixer
        self.config = config

        self._workers = []

    async def start_workers(self, task_state: TaskState):
        self.logger.info(f"[PipelineScheduler] start_workers -> TaskID={task_state.task_id}")
        self._workers = [
            asyncio.create_task(self._translation_worker(task_state)),
            asyncio.create_task(self._modelin_worker(task_state)),
            asyncio.create_task(self._tts_token_worker(task_state)),
            asyncio.create_task(self._duration_align_worker(task_state)),
            asyncio.create_task(self._audio_generation_worker(task_state)),
            asyncio.create_task(self._mixing_worker(task_state))
        ]

    async def stop_workers(self, task_state: TaskState):
        self.logger.info(f"[PipelineScheduler] stop_workers -> TaskID={task_state.task_id}")
        # 向translation_queue发送None，让整条流水线停止
        await task_state.translation_queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self.logger.info(f"[PipelineScheduler] 所有Worker已结束 -> TaskID={task_state.task_id}")

    async def push_sentences_to_pipeline(self, task_state: TaskState, sentences: List[Sentence]):
        self.logger.debug(f"[push_sentences_to_pipeline] 放入 {len(sentences)} 个句子到 translation_queue, TaskID={task_state.task_id}")
        await task_state.translation_queue.put(sentences)

    # ------------------------------
    # 各 Worker 的实现
    # ------------------------------

    @worker_decorator(
        input_queue_attr='translation_queue',
        next_queue_attr='modelin_queue',
        worker_name='翻译Worker',
        mode='stream'
    )
    async def _translation_worker(self, sentences_list: List[Sentence], task_state: TaskState):
        if not sentences_list:
            return
        self.logger.debug(f"[翻译Worker] 收到 {len(sentences_list)} 句子, TaskID={task_state.task_id}")

        async for translated_batch in self.translator.translate_sentences(
            sentences_list,
            batch_size=self.config.TRANSLATION_BATCH_SIZE,
            target_language=task_state.target_language
        ):
            yield translated_batch

    @worker_decorator(
        input_queue_attr='modelin_queue',
        next_queue_attr='tts_token_queue',
        worker_name='模型输入Worker',
        mode='stream'
    )
    async def _modelin_worker(self, sentences_batch: List[Sentence], task_state: TaskState):
        if not sentences_batch:
            return
        self.logger.debug(f"[模型输入Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        async for updated_batch in self.model_in.modelin_maker(
            sentences_batch,
            reuse_speaker=False,
            reuse_uuid=False,
            batch_size=self.config.MODELIN_BATCH_SIZE
        ):
            yield updated_batch

    @worker_decorator(
        input_queue_attr='tts_token_queue',
        next_queue_attr='duration_align_queue',
        worker_name='TTS Token生成Worker'
    )
    async def _tts_token_worker(self, sentences_batch: List[Sentence], task_state: TaskState):
        if not sentences_batch:
            return
        self.logger.debug(f"[TTS Token生成Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.tts_token_generator.tts_token_maker(sentences_batch, reuse_uuid=False)
        return sentences_batch

    @worker_decorator(
        input_queue_attr='duration_align_queue',
        next_queue_attr='audio_gen_queue',
        worker_name='时长对齐Worker'
    )
    async def _duration_align_worker(self, sentences_batch: List[Sentence], task_state: TaskState):
        if not sentences_batch:
            return
        self.logger.debug(f"[时长对齐Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.duration_aligner.align_durations(sentences_batch)
        return sentences_batch

    @worker_decorator(
        input_queue_attr='audio_gen_queue',
        next_queue_attr='mixing_queue',
        worker_name='音频生成Worker'
    )
    async def _audio_generation_worker(self, sentences_batch: List[Sentence], task_state: TaskState):
        if not sentences_batch:
            return
        self.logger.debug(f"[音频生成Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.audio_generator.vocal_audio_maker(sentences_batch)
        task_state.current_time = self.timestamp_adjuster.update_timestamps(sentences_batch, start_time=task_state.current_time)
        valid = self.timestamp_adjuster.validate_timestamps(sentences_batch)
        if not valid:
            self.logger.warning(f"[音频生成Worker] 检测到时间戳不连续, TaskID={task_state.task_id}")
        return sentences_batch

    @worker_decorator(
        input_queue_attr='mixing_queue',
        worker_name='混音Worker'
    )
    async def _mixing_worker(self, sentences_batch: List[Sentence], task_state: TaskState):
        if not sentences_batch:
            return
        seg_index = sentences_batch[0].segment_index
        self.logger.debug(f"[混音Worker] 收到 {len(sentences_batch)} 句, segment={seg_index}, TaskID={task_state.task_id}")

        output_path = task_state.task_paths.segments_dir / f"segment_{task_state.batch_counter}.mp4"

        # ========== (修改) ============
        # 将task_state.generate_subtitle 传给MediaMixer
        success = await self.mixer.mixed_media_maker(
            sentences=sentences_batch,
            task_state=task_state,
            output_path=str(output_path),
            generate_subtitle=task_state.generate_subtitle
        )

        if success and task_state.hls_manager:
            await task_state.hls_manager.add_segment(str(output_path), task_state.batch_counter)
            self.logger.info(f"[混音Worker] 分段 {task_state.batch_counter} 已加入 HLS, TaskID={task_state.task_id}")

            task_state.merged_segments.append(str(output_path))

        task_state.batch_counter += 1
        return None
