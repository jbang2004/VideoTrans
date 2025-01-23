import asyncio
import logging
from typing import List
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from core.sentence_tools import Sentence

logger = logging.getLogger(__name__)

class PipelineScheduler:
    """
    多个Worker(翻译、模型输入、TTS Token、时长对齐、音频生成、混音)的调度器。
    每个任务对应一个TaskState，用task_state.*_queue来存放数据。
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

    async def start_workers(self, task_state: "TaskState"):
        """启动worker并记录日志"""
        self.logger.info(f"[PipelineScheduler] start_workers -> TaskID={task_state.task_id}")
        self._workers = [
            asyncio.create_task(self._translation_worker(task_state)),
            asyncio.create_task(self._modelin_worker(task_state)),
            asyncio.create_task(self._tts_token_worker(task_state)),
            asyncio.create_task(self._duration_align_worker(task_state)),
            asyncio.create_task(self._audio_generation_worker(task_state)),
            asyncio.create_task(self._mixing_worker(task_state))
        ]

    async def stop_workers(self, task_state: "TaskState"):
        """停止worker并等待结束"""
        self.logger.info(f"[PipelineScheduler] stop_workers -> TaskID={task_state.task_id}")
        # 放置 None 到最上游队列
        await task_state.translation_queue.put(None)
        # 等待全部worker
        await asyncio.gather(*self._workers, return_exceptions=True)
        self.logger.info(f"[PipelineScheduler] 所有Worker已结束 -> TaskID={task_state.task_id}")

    async def push_sentences_to_pipeline(self, task_state: "TaskState", sentences: List[Sentence], is_first_segment=False):
        """将ASR得到的句子推入翻译队列，并记录日志"""
        if is_first_segment:
            batch_size = self.config.TRANSLATION_BATCH_SIZE
            total = len(sentences)
            task_state.first_segment_batch_count = (total + batch_size - 1) // batch_size
            self.logger.info(f"[push_sentences_to_pipeline] 第一段预估批次数: {task_state.first_segment_batch_count} (TaskID={task_state.task_id})")

        self.logger.debug(f"[push_sentences_to_pipeline] 放入 {len(sentences)} 个句子到 translation_queue, TaskID={task_state.task_id}")
        await task_state.translation_queue.put(sentences)

    async def wait_first_segment_done(self, task_state: "TaskState"):
        """等待第一段完成"""
        self.logger.info(f"等待第一段所有batch完成 -> TaskID={task_state.task_id}")
        await task_state.mixing_complete.get()  # 阻塞等待
        self.logger.info(f"第一段完成 -> TaskID={task_state.task_id}")

    # ------------------- Worker 实现 -------------------

    @worker_decorator(
        input_queue_attr='translation_queue',
        next_queue_attr='modelin_queue',
        worker_name='翻译Worker',
        mode='stream'
    )
    async def _translation_worker(self, sentences_list: List[Sentence], task_state: "TaskState"):
        """翻译Worker，使用 translator.translate_sentences 做流式翻译"""
        if not sentences_list:
            return
        self.logger.debug(f"[翻译Worker] 收到 {len(sentences_list)} 句子, TaskID={task_state.task_id}")

        async for translated_batch in self.translator.translate_sentences(
            sentences_list,
            batch_size=self.config.TRANSLATION_BATCH_SIZE,
            target_language=task_state.target_language
        ):
            self.logger.debug(f"[翻译Worker] 翻译完成一批 -> size={len(translated_batch)}, TaskID={task_state.task_id}")
            yield translated_batch

    @worker_decorator(
        input_queue_attr='modelin_queue',
        next_queue_attr='tts_token_queue',
        worker_name='模型输入Worker',
        mode='stream'
    )
    async def _modelin_worker(self, sentences_batch: List[Sentence], task_state: "TaskState"):
        """对翻译好的句子更新模型输入特征"""
        if not sentences_batch:
            return
        self.logger.debug(f"[模型输入Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        async for updated_batch in self.model_in.modelin_maker(
            sentences_batch,
            reuse_speaker=False,
            reuse_uuid=False,
            batch_size=self.config.MODELIN_BATCH_SIZE
        ):
            self.logger.debug(f"[模型输入Worker] 处理完成一批 -> size={len(updated_batch)}, TaskID={task_state.task_id}")
            yield updated_batch

    @worker_decorator(
        input_queue_attr='tts_token_queue',
        next_queue_attr='duration_align_queue',
        worker_name='TTS Token生成Worker'
    )
    async def _tts_token_worker(self, sentences_batch: List[Sentence], task_state: "TaskState"):
        """批量生成 TTS token"""
        if not sentences_batch:
            return
        self.logger.debug(f"[TTS Token生成Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.tts_token_generator.tts_token_maker(sentences_batch, reuse_uuid=False)
        self.logger.debug(f"[TTS Token生成Worker] 已为本批次生成token -> size={len(sentences_batch)}, TaskID={task_state.task_id}")
        return sentences_batch

    @worker_decorator(
        input_queue_attr='duration_align_queue',
        next_queue_attr='audio_gen_queue',
        worker_name='时长对齐Worker'
    )
    async def _duration_align_worker(self, sentences_batch: List[Sentence], task_state: "TaskState"):
        """对句子时长进行对齐修正"""
        if not sentences_batch:
            return
        self.logger.debug(f"[时长对齐Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.duration_aligner.align_durations(sentences_batch)
        self.logger.debug(f"[时长对齐Worker] 对齐完成 -> size={len(sentences_batch)}, TaskID={task_state.task_id}")
        return sentences_batch

    @worker_decorator(
        input_queue_attr='audio_gen_queue',
        next_queue_attr='mixing_queue',
        worker_name='音频生成Worker'
    )
    async def _audio_generation_worker(self, sentences_batch: List[Sentence], task_state: "TaskState"):
        """生成语音音频并更新时间戳"""
        if not sentences_batch:
            return
        self.logger.debug(f"[音频生成Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        await self.audio_generator.vocal_audio_maker(sentences_batch)
        task_state.current_time = self.timestamp_adjuster.update_timestamps(
            sentences_batch, start_time=task_state.current_time
        )
        if not self.timestamp_adjuster.validate_timestamps(sentences_batch):
            self.logger.warning(f"[音频生成Worker] 检测到时间戳不连续, TaskID={task_state.task_id}")

        self.logger.debug(f"[音频生成Worker] 音频生成完成 -> size={len(sentences_batch)}, TaskID={task_state.task_id}")
        return sentences_batch

    @worker_decorator(
        input_queue_attr='mixing_queue',
        worker_name='混音Worker'
    )
    async def _mixing_worker(self, sentences_batch: List[Sentence], task_state: "TaskState"):
        """混音 + HLS 分段"""
        if not sentences_batch:
            return
        seg_index = sentences_batch[0].segment_index
        self.logger.debug(f"[混音Worker] 收到 {len(sentences_batch)} 句, segment={seg_index}, TaskID={task_state.task_id}")

        output_path = task_state.task_paths.segments_dir / f"segment_{task_state.batch_counter}.mp4"

        success = await self.mixer.mixed_media_maker(
            sentences=sentences_batch,
            task_state=task_state,
            output_path=str(output_path)
        )

        if success and task_state.hls_manager:
            await task_state.hls_manager.add_segment(str(output_path), task_state.batch_counter)
            self.logger.info(f"[混音Worker] 分段 {task_state.batch_counter} 已加入 HLS, TaskID={task_state.task_id}")

        # 如果是第一段
        if seg_index == 0:
            task_state.first_segment_processed_count += 1
            self.logger.info(
                f"[混音Worker] 第一段处理: {task_state.first_segment_processed_count}/{task_state.first_segment_batch_count}, "
                f"TaskID={task_state.task_id}"
            )
            if task_state.first_segment_processed_count >= task_state.first_segment_batch_count and task_state.mixing_complete:
                await task_state.mixing_complete.put(True)

        task_state.batch_counter += 1
        self.logger.debug(f"[混音Worker] 本批次混音完成 -> batch_counter={task_state.batch_counter}, TaskID={task_state.task_id}")
        return None
