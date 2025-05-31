# ---------------------------------------------------
# backend/core/media_mixer.py (精简版)
# ---------------------------------------------------
import numpy as np
import logging
from typing import List, Optional, Tuple
from ray import serve
import asyncio

# Ray tasks imported directly
from utils.audio_utils import apply_fade_effect, mix_with_background, normalize_audio
from utils.video_utils import add_video_segment 
from config import Config
from core.sentence_tools import Sentence
from utils.task_storage import TaskPaths
from core.supabase_client import SupabaseClient

# 使用全局日志配置，直接获取 logger
logger = logging.getLogger(__name__)

@serve.deployment(
    name="media_mixer",
    ray_actor_options={"num_cpus": 1},
    num_replicas=2,  # 添加多个实例以提高吞吐量
    logging_config={"log_level": "INFO"}
)
class MediaMixer:
    """
    媒体混合，负责混合音频和视频
    """
    def __init__(self):
        self.config = Config()
        self.sample_rate = self.config.TARGET_SR
        self.max_val = 0.8  # 音频最大值
        self.logger = logging.getLogger(__name__)
        self.supabase_client = SupabaseClient(config=self.config)
        self.logger.info(f"MediaMixerActor初始化完成，采样率={self.sample_rate}")
        self.full_audio_buffer = np.array([], dtype=np.float32)  # 保留音频缓冲区，用于平滑过渡
    
    async def mix_media(
        self,
        sentences_batch: List[Sentence],
        task_paths: TaskPaths,
        batch_counter: int,
        task_id: str
    ) -> Optional[str]:
        """处理一批句子并返回处理后的视频片段路径"""
        
        try:
            # If this is the first batch processed by the mixer for this task, update status to 'mixing'
            if batch_counter == 0 and self.supabase_client:
                try:
                    self.logger.info(f"[{task_id}] MediaMixer: First batch (batch_counter=0), updating status to 'mixing'.")
                    asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'mixing'}))
                except Exception as e_update_status:
                    self.logger.error(f"[{task_id}] MediaMixer: Failed to update status to 'mixing': {e_update_status}")

            if not sentences_batch:
                logger.warning(f"[{task_id}] mix_media: 收到空的句子列表")
                return None

            # 获取 media_files 和 target_language
            media_files = {}
            target_language = None # 初始化
            generate_subtitle = False # 初始化
            if self.supabase_client:
                task_data = await self.supabase_client.get_task(task_id)
                if task_data:
                    media_files['silent_video_path'] = task_data.get('silent_video_path')
                    media_files['vocals_audio_path'] = task_data.get('vocals_audio_path')
                    media_files['background_audio_path'] = task_data.get('background_audio_path')
                    target_language = task_data.get('target_language') # <<< 获取 target_language
                    generate_subtitle = task_data.get('generate_subtitle', False) # <<< 获取 generate_subtitle
                    # 获取视频尺寸
                    video_width = task_data.get('video_width', -1) # Default to -1 if not found
                    video_height = task_data.get('video_height', -1) # Default to -1 if not found
                    media_files['video_width'] = video_width
                    media_files['video_height'] = video_height
                    
                    if video_width == -1 or video_height == -1:
                        self.logger.warning(f"[{task_id}] MediaMixer: video_width or video_height not found or invalid in task_data. Subtitle scaling might be affected.")
                else:
                    self.logger.error(f"[{task_id}] MediaMixer: 无法从数据库获取任务信息")
                    return None
            else:
                self.logger.error(f"[{task_id}] MediaMixer: Supabase客户端未初始化")
                return None

            if not media_files.get('silent_video_path') or not media_files.get('vocals_audio_path'):
                self.logger.error(f"[{task_id}] MediaMixer: 数据库中缺少 silent_video_path 或 vocals_audio_path")
                return None
            
            # 只有在生成字幕时才需要target_language
            if generate_subtitle and not target_language:
                self.logger.error(f"[{task_id}] MediaMixer: 生成字幕时缺少 target_language")
                return None
            
            logger.info(f"[{task_id}] 开始处理批次 {batch_counter}, 句子数 {len(sentences_batch)}, 目标语言: {target_language}")
            
            output_path = task_paths.segments_dir / f"segment_{batch_counter}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            max_val = 1.0
            
            success, updated_buffer = await create_mixed_segment(
                sentences=sentences_batch,
                media_files=media_files,
                task_paths=task_paths,
                output_path=str(output_path),
                generate_subtitle=generate_subtitle,
                config=self.config,
                sample_rate=self.sample_rate,
                max_val=max_val,
                full_audio_buffer=self.full_audio_buffer,
                task_id=task_id,
                target_language=target_language
            )
            
            if not success:
                logger.error(f"[{task_id}] 批次 {batch_counter} 处理失败")
                return None
            
            if len(updated_buffer) > self.sample_rate * 5:
                preserve_samples = min(len(updated_buffer), int(self.sample_rate * 5))
                self.full_audio_buffer = updated_buffer[-preserve_samples:]
            else:
                self.full_audio_buffer = updated_buffer
                
            logger.info(f"[{task_id}] 更新音频缓冲区, 批次 {batch_counter}, 句子数 {len(sentences_batch)}")
            
            return str(output_path)
                
        except Exception as e:
            logger.exception(f"[{task_id}] mix_media 执行出错: {str(e)}")
            return None
        finally:
            if 'updated_buffer' in locals() and 'success' in locals() and success and updated_buffer is not self.full_audio_buffer:
                del updated_buffer

async def create_mixed_segment(
    sentences: List[Sentence],
    media_files: dict,
    task_paths: TaskPaths,
    output_path: str,
    generate_subtitle: bool,
    config: Config,
    sample_rate: int,
    max_val: float,
    full_audio_buffer: np.ndarray,
    task_id: str,
    target_language: str
) -> Tuple[bool, np.ndarray]:
    """
    将一批句子的合成音频与原视频片段混合，并可生成带字幕的视频。
    返回(成功标志, 更新后的音频缓冲区)
    """
    full_audio = None
    updated_audio_buffer = None
    audio_data = None
    
    try:
        if not sentences:
            logger.warning(f"[{task_id}] create_mixed_segment: 收到空的句子列表")
            return False, full_audio_buffer

        full_audio = await asyncio.to_thread(_concat_audio_segments, sentences, full_audio_buffer, config.AUDIO_OVERLAP)
        if len(full_audio) == 0:
            logger.error(f"[{task_id}] create_mixed_segment: 没有有效的合成音频数据")
            return False, full_audio_buffer

        start_time_param, duration = await asyncio.to_thread(_calculate_time_params, sentences)

        if not media_files:
            logger.error(f"[{task_id}] create_mixed_segment: 找不到媒体文件信息")
            return False, full_audio_buffer

        background_audio_path = media_files.get('background_audio_path')
        if background_audio_path:
            audio_data = await _process_background_audio(
                background_audio_path, 
                start_time_param, 
                duration, 
                full_audio,
                sample_rate, 
                config.VOCALS_VOLUME, 
                config.BACKGROUND_VOLUME, 
                max_val
            )
            if audio_data is not None:
                full_audio = audio_data
                del audio_data
                audio_data = None

        video_path = media_files.get('silent_video_path')
        if not video_path:
            logger.warning(f"[{task_id}] create_mixed_segment: 本片段无video_path可用")
            return False, full_audio_buffer
            
        # Extract video_width and video_height from media_files
        video_width = media_files.get('video_width', -1)
        video_height = media_files.get('video_height', -1)
        if video_width == -1 or video_height == -1:
            logger.warning(f"[{task_id}] create_mixed_segment: video_width or video_height not found or invalid in media_files. Defaulting or skipping scaling.")
            # Potentially set to a default or handle error, for now, it will pass -1

        updated_audio_buffer = np.concatenate((full_audio_buffer, full_audio))

        await add_video_segment(
            video_path=video_path,
            start_time=start_time_param,
            duration=duration,
            audio_data=full_audio,
            output_path=output_path,
            sentences=sentences,
            generate_subtitle=generate_subtitle,
            target_language=target_language,
            sample_rate=sample_rate,
            video_width=video_width,      # Pass video_width
            video_height=video_height    # Pass video_height
        )
        
        if len(updated_audio_buffer) > sample_rate * 5:
            preserve_samples = min(len(updated_audio_buffer), int(sample_rate * 5))
            updated_audio_buffer = updated_audio_buffer[-preserve_samples:]
        
        return True, updated_audio_buffer
        
    except Exception as e:
        logger.exception(f"[{task_id}] create_mixed_segment 执行出错，错误: {e}")
        return False, full_audio_buffer
    finally:
        if 'full_audio' in locals() and full_audio is not None and full_audio is not updated_audio_buffer: 
            del full_audio

def _concat_audio_segments(sentences: List[Sentence], full_audio_buffer: np.ndarray, overlap: float) -> np.ndarray:
    """拼接所有句子的合成音频"""
    full_audio = np.array([], dtype=np.float32)
    for sentence in sentences:
        if sentence.generated_audio is not None and len(sentence.generated_audio) > 0:
            audio_data = np.asarray(sentence.generated_audio, dtype=np.float32)
            if len(full_audio) > 0:
                audio_data = apply_fade_effect(audio_data, full_audio_buffer, overlap)
            full_audio = np.concatenate((full_audio, audio_data))
        else:
            logger.warning(
                "句子音频生成失败或为空: text=%r, UUID=%s",
                sentence.raw_text,
                sentence.model_input.get("uuid", "unknown")
            )
    return full_audio

def _calculate_time_params(sentences: List[Sentence]) -> tuple:
    """计算时间参数"""

    start_time = 0.0
    if not sentences[0].is_first:
        start_time = sentences[0].adjusted_start / 1000.0
    duration = sum(s.adjusted_duration for s in sentences) / 1000.0
    return start_time, duration

async def _process_background_audio(
    bg_path: str, start_time: float, duration: float, audio_data: np.ndarray,
    sample_rate: int, vocals_volume: float, background_volume: float, max_val: float
) -> np.ndarray:
    """处理背景音频 - 异步版本"""
    try:
        mixed_audio = await mix_with_background(
            bg_path=bg_path,
            start_time=start_time,
            duration=duration,
            audio_data=audio_data,
            sample_rate=sample_rate,
            vocals_volume=vocals_volume,
            background_volume=background_volume
        )
        return normalize_audio(mixed_audio, max_val)
    finally:
        if 'mixed_audio' in locals():
            del mixed_audio
