import asyncio
import logging
import ray
from typing import List
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from core.sentence_tools import Sentence
from core.tts_token_gener import generate_tts_tokens
from core.timeadjust.duration_aligner import align_durations
from core.audio_gener import generate_audio
from core.timeadjust.timestamp_adjuster import adjust_timestamps
from core.media_mixer import MediaMixerActor

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
        config,
        sample_rate=None,  # 采样率，如果为None则使用cosyvoice_actor的采样率
        max_speed=1.1,      # 最大语速阈值
        hls_manager=None    # HLS管理器实例（可选）
    ):
        self.logger = logging.getLogger(__name__)
        self.translator_actor = translator_actor  # TranslatorActor引用
        self.model_in_actor = model_in_actor      # ModelInActor引用
        self.cosyvoice_actor = cosyvoice_actor    # CosyVoice模型Actor
        self.simplifier = simplifier
        self.config = config
        self.sample_rate = sample_rate if sample_rate else ray.get(cosyvoice_actor.get_sample_rate.remote())
        self.max_speed = max_speed
        self.hls_manager = hls_manager
        
        # 初始化MediaMixerActor
        self.media_mixer_actor = MediaMixerActor.remote(config, self.sample_rate)
        self.logger.info(f"PipelineScheduler初始化完成，采样率={self.sample_rate}")

    async def start_workers(self, task_state: TaskState):
        """启动异步工作器"""
        self.logger.info(f"[PipelineScheduler] 启动工作器 -> TaskID={task_state.task_id}")
        # 不再需要启动mixing_worker，因为使用Ray Actor处理

    async def stop_workers(self, task_state: TaskState):
        """停止异步工作器"""
        self.logger.info(f"[PipelineScheduler] 停止工作器 -> TaskID={task_state.task_id}")
        # 不再需要停止mixing_worker，因为使用Ray Actor处理

    async def cleanup_resources(self, task_state: TaskState):
        """清理资源"""
        try:
            # 清理各种特征缓存，但保留说话人特征
            await ray.get(self.cosyvoice_actor.cleanup_feature_cache.remote(cache_ids=None, skip_speaker_features=True))
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
                    modelin_ref,
                    self.cosyvoice_actor
                )
                
                # 直接创建时长对齐任务，传递TTS token生成任务的引用
                self.logger.info(f"创建时长对齐任务")
                aligned_ref = align_durations.remote(
                    tts_token_ref,
                    self.simplifier,
                    self.model_in_actor,
                    self.cosyvoice_actor,
                    self.max_speed
                )
                
                # 直接创建音频生成任务，传递时长对齐任务的引用
                self.logger.info(f"创建音频生成任务")
                audio_ref = generate_audio.remote(
                    aligned_ref,
                    self.cosyvoice_actor,
                    self.sample_rate
                )
                
                # 创建时间戳调整任务，传递音频生成任务的引用
                self.logger.info(f"创建时间戳调整任务")
                timestamp_ref = adjust_timestamps.remote(
                    audio_ref,
                    self.sample_rate,
                    task_state.current_time
                )
                
                sentences_with_timestamps = ray.get(timestamp_ref)
                
                # 更新当前时间（使用最后一个句子的结束时间）
                if sentences_with_timestamps:
                    last_sentence = sentences_with_timestamps[-1]
                    task_state.current_time = last_sentence.adjusted_start + last_sentence.adjusted_duration
                
                self.logger.info(f"时间戳调整任务完成: {len(sentences_with_timestamps) if sentences_with_timestamps else 0}个句子")
                
                # 使用MediaMixerActor处理媒体混合，直接传递时间戳调整任务的引用
                self.logger.info(f"调用MediaMixerActor进行媒体混合")
                mix_ref = self.media_mixer_actor.mix_media.remote(
                    timestamp_ref,
                    task_state
                )
                
                # 获取混合任务的结果
                output_path = ray.get(mix_ref)
                if output_path:
                    # 记录已处理的片段
                    task_state.merged_segments.append(output_path)
                    
                    # 如果处理成功且有HLS管理器，添加到HLS流
                    if self.hls_manager:
                        try:
                            await self.hls_manager.add_segment(output_path, task_state.batch_counter)
                            self.logger.info(f"分段 {task_state.batch_counter} 已加入 HLS -> TaskID={task_state.task_id}")
                        except Exception as e:
                            self.logger.error(f"添加HLS片段失败: {e} -> TaskID={task_state.task_id}")
                    
                    # 更新批次计数器
                    task_state.batch_counter += 1

