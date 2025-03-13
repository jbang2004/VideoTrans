import ray
import logging
import asyncio
from typing import List, Optional
from config import Config
from core.sentence_tools import Sentence
from core.translation.translator_actor import TranslatorActor
from core.model_in_actor import ModelInActor
from core.media_mixer_actor import MediaMixerActor
from core.tts_token_gener import generate_tts_tokens
from core.timeadjust.duration_aligner import align_durations
from core.audio_gener import generate_audio
from core.timeadjust.timestamp_adjuster import adjust_timestamps
from utils.task_state import TaskState
from utils.media_utils import extract_segment_task

logger = logging.getLogger(__name__)

class PipelineScheduler:
    """
    流水线调度器，负责协调翻译、模型输入、TTS token生成、时长对齐、音频生成和混音等处理步骤。
    使用Ray的依赖传递机制来执行所有处理步骤，通过Actor模式处理媒体混合和HLS流生成。
    """

    def __init__(
        self,
        translator_actor,  # TranslatorActor
        model_in_actor,    # ModelInActor
        cosyvoice_actor,   # CosyVoiceModelActor
        simplifier,        # 简化器（通常是TranslatorActor）
        config,
        sample_rate=None,  # 采样率，如果为None则使用config.TARGET_SR
        max_speed=1.1,     # 最大语速阈值
        hls_manager_actor=None,  # HLS管理器Actor引用（可选）
        audio_separator_actor=None  # AudioSeparatorActor引用（可选）
    ):
        self.logger = logging.getLogger(__name__)
        self.translator_actor = translator_actor  # TranslatorActor引用
        self.model_in_actor = model_in_actor      # ModelInActor引用
        self.cosyvoice_actor = cosyvoice_actor    # CosyVoice模型Actor
        self.simplifier = simplifier
        self.config = config
        self.sample_rate = sample_rate if sample_rate else config.TARGET_SR
        self.max_speed = max_speed
        self.hls_manager_actor = hls_manager_actor  # HLSManagerActor引用
        self.audio_separator_actor = audio_separator_actor  # AudioSeparatorActor引用
        
        # 初始化MediaMixerActor
        self.media_mixer_actor = MediaMixerActor.options(
            num_cpus=config.MEDIA_MIXER_ACTOR_NUM_CPUS
        ).remote(config, self.sample_rate)
        self.logger.info(f"PipelineScheduler初始化完成，采样率={self.sample_rate}")

    async def cleanup_resources(self, task_state: TaskState):
        """清理资源"""
        try:
            # 清理各种特征缓存，但保留说话人特征
            await self.cosyvoice_actor.cleanup_feature_cache.remote(cache_ids=None, skip_speaker_features=True)
            self.logger.info(f"[PipelineScheduler] 已清理资源（保留说话人特征） -> TaskID={task_state.task_id}")
        except Exception as e:
            self.logger.error(f"[PipelineScheduler] 清理资源失败: {e} -> TaskID={task_state.task_id}")

    async def push_sentences_to_pipeline(
        self, 
        task_state: TaskState, 
        segment_index: int, 
        segment_start: float, 
        sense_model_actor,
        segment_duration: float = None
    ):
        """
        将句子推送到流水线进行处理。
        执行ASR识别，并处理结果。
        
        Args:
            task_state: 任务状态对象
            segment_index: 分段索引
            segment_start: 分段开始时间
            sense_model_actor: ASR模型Actor引用
            segment_duration: 分段持续时间（可选）
        """
        try:
            # 1. 提取并分离人声/背景 - 使用Ray任务
            media_files = await extract_segment_task.remote(
                video_path=task_state.video_path,
                start=segment_start,
                duration=segment_duration,
                output_dir=str(task_state.task_paths.processing_dir),
                segment_index=segment_index,
                audio_separator_actor=self.audio_separator_actor,
                target_sr=self.sample_rate
            )
            task_state.segment_media_files[segment_index] = media_files
            
            # 获取该分段的媒体文件信息
            media_files = task_state.segment_media_files.get(segment_index)
            if not media_files or 'vocals' not in media_files:
                self.logger.error(f"[push_sentences_to_pipeline] 找不到分段 {segment_index} 的vocals文件, TaskID={task_state.task_id}")
                return
            
            # 执行ASR识别
            sentences = await sense_model_actor.generate_async.remote(
                input=media_files['vocals'],
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False
            )
            
            self.logger.info(f"[push_sentences_to_pipeline] ASR识别完成: {len(sentences)} 条句子, seg={segment_index}, TaskID={task_state.task_id}")
            
            if not sentences:
                self.logger.warning(f"[push_sentences_to_pipeline] ASR结果为空, seg={segment_index}, TaskID={task_state.task_id}")
                return
            
            # 为句子添加元数据
            for s in sentences:
                s.segment_index = segment_index
                s.segment_start = segment_start
                s.task_id = task_state.task_id
                s.sentence_id = task_state.sentence_counter
                task_state.sentence_counter += 1
                
        except Exception as e:
            self.logger.error(f"[push_sentences_to_pipeline] ASR处理失败: {str(e)}, seg={segment_index}, TaskID={task_state.task_id}")
            return
            
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
                
                # 修改为异步调用
                sentences_with_timestamps = await timestamp_ref
                
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
                
                # 获取混合任务的结果，修改为异步调用
                output_path = await mix_ref
                if output_path:
                    # 记录已处理的片段
                    task_state.merged_segments.append(output_path)
                    
                    # 如果处理成功且有HLS管理器，添加到HLS流
                    if self.hls_manager_actor:
                        try:
                            # 使用Ray Actor方式调用add_segment方法，修改为异步调用
                            add_segment_ref = self.hls_manager_actor.add_segment.remote(output_path, task_state.batch_counter)
                            success = await add_segment_ref
                            if success:
                                self.logger.info(f"分段 {task_state.batch_counter} 已加入 HLS -> TaskID={task_state.task_id}")
                                # 成功添加第一个分段后，设置HLS就绪状态
                                if not task_state.hls_ready and task_state.batch_counter == 0:
                                    task_state.hls_ready = True
                                    self.logger.info(f"HLS播放列表已就绪 -> TaskID={task_state.task_id}")
                            else:
                                self.logger.error(f"添加HLS片段失败 -> TaskID={task_state.task_id}")
                        except Exception as e:
                            self.logger.error(f"添加HLS片段失败: {e} -> TaskID={task_state.task_id}")
                    
                    # 更新批次计数器
                    task_state.batch_counter += 1

