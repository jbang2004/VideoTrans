import asyncio
import logging
import ray
from typing import List
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from core.sentence_tools import Sentence
from core.tts_token_gener import generate_tts_tokens
from utils import concurrency

logger = logging.getLogger(__name__)

class PipelineScheduler:
    """
    多个Worker的调度器，负责翻译->model_in->TTS token生成->时长对齐->音频生成->混音 ...
    使用 Ray 的依赖传递重构翻译、模型输入和TTS Token生成部分，后续部分仍使用队列。
    """

    def __init__(
        self,
        translator_actor,  # TranslatorActor
        model_in_actor,    # ModelInActor
        cosyvoice_actor,   # CosyVoiceModelActor (替代tts_token_generator)
        duration_aligner,
        audio_generator,
        timestamp_adjuster,
        mixer,
        config
    ):
        self.logger = logging.getLogger(__name__)
        self.translator_actor = translator_actor  # TranslatorActor引用
        self.model_in_actor = model_in_actor      # ModelInActor引用
        self.cosyvoice_actor = cosyvoice_actor    # CosyVoice模型Actor
        self.duration_aligner = duration_aligner
        self.audio_generator = audio_generator
        self.timestamp_adjuster = timestamp_adjuster
        self.mixer = mixer
        self.config = config

        self._workers = []

    async def start_workers(self, task_state: TaskState):
        """
        启动后续的异步 worker，不包括翻译、模型输入和TTS Token生成部分（它们通过依赖传递执行）。
        """
        self.logger.info(f"[PipelineScheduler] start_workers -> TaskID={task_state.task_id}")
        self._workers = [
            asyncio.create_task(self._duration_align_worker(task_state)),
            asyncio.create_task(self._audio_generation_worker(task_state)),
            asyncio.create_task(self._mixing_worker(task_state))
        ]

    async def stop_workers(self, task_state: TaskState):
        """
        停止所有 worker，通过向 duration_align_queue 发送 None 信号。
        """
        self.logger.info(f"[PipelineScheduler] stop_workers -> TaskID={task_state.task_id}")
        await task_state.duration_align_queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self.logger.info(f"[PipelineScheduler] 所有Worker已结束 -> TaskID={task_state.task_id}")
        
        # 清理资源，但保留说话人特征
        await self.cleanup_resources(task_state)

    async def cleanup_resources(self, task_state: TaskState):
        """
        清理任务相关的资源，但保留说话人特征
        """
        from utils import concurrency
        
        try:
            # 使用concurrency.run_sync调用Actor方法
            await concurrency.run_sync(
                lambda: ray.get(self.model_in_actor.cosyvoice_actor.cleanup_feature_cache.remote(
                    cache_ids=None,  # 清理所有缓存
                    skip_speaker_features=True  # 跳过说话人特征
                ))
            )
            self.logger.info(f"[PipelineScheduler] 已清理资源（保留说话人特征） -> TaskID={task_state.task_id}")
        except Exception as e:
            self.logger.error(f"[PipelineScheduler] 清理资源失败: {e} -> TaskID={task_state.task_id}")

    async def push_sentences_to_pipeline(self, task_state: TaskState, sentences: List[Sentence]):
        """
        将句子推送到流水线，使用 Ray 的依赖传递执行翻译、模型输入和TTS Token生成，最后放入 duration_align_queue。
        """
        self.logger.debug(f"[push_sentences_to_pipeline] 处理 {len(sentences)} 个句子, TaskID={task_state.task_id}")
        
        # 迭代获取翻译的 ObjectRef
        for translated_ref in self.translator_actor.translate_sentences.remote(
            sentences,
            target_language=task_state.target_language,
            batch_size=self.config.TRANSLATION_BATCH_SIZE
        ):
            # 使用 model_in_actor 处理翻译后的句子
            for modelin_ref in self.model_in_actor.modelin_maker.remote(
                translated_ref,
                reuse_speaker=False,
                batch_size=self.config.MODELIN_BATCH_SIZE
            ):
                # 使用 generate_tts_tokens task 处理模型输入后的句子
                self.logger.info(f"TTS token生成开始")
                tts_token_ref = generate_tts_tokens.remote(
                    modelin_ref,  # 直接传递引用，不使用ray.get()
                    self.cosyvoice_actor
                )
                
                # 获取处理后的句子并放入下一阶段队列
                tts_processed_batch = ray.get(tts_token_ref)
                self.logger.info(f"TTS token生成完成: {len(tts_processed_batch)}个句子")
                await task_state.duration_align_queue.put(tts_processed_batch)

    # ------------------------------
    # 后续 Worker 
    # ------------------------------

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