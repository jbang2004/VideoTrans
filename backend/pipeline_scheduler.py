import asyncio
import logging
import ray
from typing import List
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from core.sentence_tools import Sentence
from core.tts_token_gener import generate_tts_tokens
from utils import concurrency
from core.timeadjust.duration_aligner import align_durations
from core.audio_gener import generate_audio
from core.timeadjust.timestamp_adjuster import adjust_timestamps

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
        simplifier,        # 简化器（通常是TranslatorActor）
        mixer,
        config,
        sample_rate=None,  # 采样率，如果为None则使用cosyvoice_actor的采样率
        max_speed=1.1      # 最大语速阈值
    ):
        self.logger = logging.getLogger(__name__)
        self.translator_actor = translator_actor  # TranslatorActor引用
        self.model_in_actor = model_in_actor      # ModelInActor引用
        self.cosyvoice_actor = cosyvoice_actor    # CosyVoice模型Actor
        self.simplifier = simplifier
        self.mixer = mixer
        self.config = config
        self.sample_rate = sample_rate
        self.max_speed = max_speed

        self._workers = []

    async def start_workers(self, task_state: TaskState):
        """
        启动后续的异步 worker，只包含混音部分。
        时长对齐和音频生成现在直接在push_sentences_to_pipeline中处理。
        """
        self.logger.info(f"[PipelineScheduler] start_workers -> TaskID={task_state.task_id}")
        self._workers = [
            # 移除了_audio_generation_worker，只保留_mixing_worker
            asyncio.create_task(self._mixing_worker(task_state))
        ]

    async def stop_workers(self, task_state: TaskState):
        """
        停止所有 worker，通过向 mixing_queue 发送 None 信号。
        """
        self.logger.info(f"[PipelineScheduler] stop_workers -> TaskID={task_state.task_id}")
        await task_state.mixing_queue.put(None)
        await asyncio.gather(*self._workers, return_exceptions=True)
        self.logger.info(f"[PipelineScheduler] 所有Worker已结束 -> TaskID={task_state.task_id}")
        
        # 清理资源，但保留说话人特征
        await self.cleanup_resources(task_state)

    async def cleanup_resources(self, task_state: TaskState):
        """
        清理任务相关的资源，但保留说话人特征（可重用）
        """
        self.logger.info(f"[PipelineScheduler] cleanup_resources -> TaskID={task_state.task_id}")
        try:
            # 清理各种特征缓存，但保留说话人特征
            await ray.get(self.cosyvoice_actor.cleanup_text_features.remote())
            await ray.get(self.cosyvoice_actor.cleanup_tts_tokens.remote())
            await ray.get(self.cosyvoice_actor.cleanup_processed_audio.remote())
            self.logger.info(f"[PipelineScheduler] 已清理资源（保留说话人特征） -> TaskID={task_state.task_id}")
        except Exception as e:
            self.logger.error(f"[PipelineScheduler] 清理资源失败: {e} -> TaskID={task_state.task_id}")

    async def push_sentences_to_pipeline(self, task_state: TaskState, sentences: List[Sentence]):
        """
        将句子推送到流水线，使用 Ray 的依赖传递执行翻译、模型输入、TTS Token生成、时长对齐和音频生成，最后放入 mixing_queue。
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
                    modelin_ref,  # 参数名已改为sentences，但这里仍传递modelin_ref引用
                    self.cosyvoice_actor
                )
                
                # 直接创建时长对齐任务，传递TTS token生成任务的引用
                self.logger.info(f"创建时长对齐任务")
                aligned_ref = align_durations.remote(
                    tts_token_ref,  # 参数名已改为sentences，但这里仍传递tts_token_ref引用
                    self.simplifier,
                    self.model_in_actor,
                    self.cosyvoice_actor,
                    self.max_speed
                )
                
                # 直接创建音频生成任务，传递时长对齐任务的引用
                self.logger.info(f"创建音频生成任务")
                audio_ref = generate_audio.remote(
                    aligned_ref,  # 直接传递aligned_ref引用
                    self.cosyvoice_actor,
                    self.sample_rate
                )
                
                # 创建时间戳调整任务，传递音频生成任务的引用
                self.logger.info(f"创建时间戳调整任务")
                timestamp_ref = adjust_timestamps.remote(
                    audio_ref,  # 直接传递audio_ref引用
                    self.sample_rate,  # 直接使用sample_rate，不再需要timestamp_adjuster
                    task_state.current_time
                )
                
                # 获取时间戳调整任务结果
                sentences_with_timestamps = ray.get(timestamp_ref)
                
                # 更新当前时间（使用最后一个句子的结束时间）
                if sentences_with_timestamps:
                    last_sentence = sentences_with_timestamps[-1]
                    task_state.current_time = last_sentence.adjusted_start + last_sentence.adjusted_duration
                
                self.logger.info(f"时间戳调整任务完成: {len(sentences_with_timestamps) if sentences_with_timestamps else 0}个句子")
                
                # 直接放入混音队列
                await task_state.mixing_queue.put(sentences_with_timestamps)

    # ------------------------------
    # 后续 Worker 
    # ------------------------------

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