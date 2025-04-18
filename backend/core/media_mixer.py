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
from utils.task_state import TaskState

# 使用全局日志配置，直接获取 logger
logger = logging.getLogger(__name__)

@serve.deployment(
    name="media_mixer",
    ray_actor_options={"num_cpus": 1},
    num_replicas=2  # 添加多个实例以提高吞吐量
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
        self.logger.info(f"MediaMixerActor初始化完成，采样率={self.sample_rate}")
        self.full_audio_buffer = np.array([], dtype=np.float32)  # 保留音频缓冲区，用于平滑过渡
    
    async def mix_media(
        self,
        sentences_batch: List[Sentence],
        task_state: TaskState,
    ) -> Optional[str]:
        """处理一批句子并返回处理后的视频片段路径"""
        
        try:
            if not sentences_batch:
                logger.warning("[MediaMixerActor] mix_media: 收到空的句子列表")
                return None
            
            seg_index = sentences_batch[0].segment_index
            batch_counter = task_state.batch_counter
            logger.info(f"[MediaMixerActor] 开始处理分段 {seg_index}, 批次 {batch_counter}, 句子数 {len(sentences_batch)}")
            
            # 生成输出路径
            output_path = task_state.task_paths.segments_dir / f"segment_{batch_counter}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 初始化参数
            max_val = 1.0
            
            # 处理音视频，使用Actor中保存的full_audio_buffer
            success, updated_buffer = await create_mixed_segment(
                sentences=sentences_batch,
                task_state=task_state,
                output_path=str(output_path),
                generate_subtitle=task_state.generate_subtitle,
                config=self.config,
                sample_rate=self.sample_rate,
                max_val=max_val,
                full_audio_buffer=self.full_audio_buffer
            )
            
            if not success:
                logger.error(f"[MediaMixerActor] 分段 {batch_counter} 处理失败, TaskID={task_state.task_id}")
                return None
            
            # 更新Actor中的音频缓冲区，但限制大小以节省内存
            # 只保留最后5秒的音频用于下一批次的平滑过渡
            if len(updated_buffer) > self.sample_rate * 5:
                preserve_samples = min(len(updated_buffer), int(self.sample_rate * 5))
                self.full_audio_buffer = updated_buffer[-preserve_samples:]
            else:
                self.full_audio_buffer = updated_buffer
                
            logger.info(f"[MediaMixerActor] 更新音频缓冲区, 分段 {batch_counter}, 句子数 {len(sentences_batch)}")
            
            # 返回处理后的视频片段路径
            return str(output_path)
                
        except Exception as e:
            logger.exception(f"[MediaMixerActor] mix_media 执行出错: {str(e)}")
            return None
        finally:
            # 清理不再需要的变量
            if 'updated_buffer' in locals() and 'success' in locals() and success and updated_buffer is not self.full_audio_buffer:
                del updated_buffer

async def create_mixed_segment(
    sentences: List[Sentence],
    task_state: TaskState,
    output_path: str,
    generate_subtitle: bool,
    config: Config,
    sample_rate: int,
    max_val: float,
    full_audio_buffer: np.ndarray
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
            logger.warning("[MediaMixer] create_mixed_segment: 收到空的句子列表")
            return False, full_audio_buffer

        # 1. 拼接所有句子的合成音频 - 直接用asyncio.to_thread包装
        full_audio = await asyncio.to_thread(_concat_audio_segments, sentences, full_audio_buffer, config.AUDIO_OVERLAP)
        if len(full_audio) == 0:
            logger.error("[MediaMixer] create_mixed_segment: 没有有效的合成音频数据")
            return False, full_audio_buffer

        # 2. 计算时间参数 - 直接用asyncio.to_thread包装
        start_time_param, duration = await asyncio.to_thread(_calculate_time_params, sentences)

        # 3. 背景音乐混合
        segment_index = sentences[0].segment_index
        segment_files = task_state.segment_media_files.get(segment_index)
        if not segment_files:
            logger.error(f"[MediaMixer] 找不到分段 {segment_index} 对应的媒体文件信息")
            return False, full_audio_buffer

        background_audio_path = segment_files.get('background')
        if background_audio_path:
            # 调用异步函数 _process_background_audio
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
            # 更新音频数据
            if audio_data is not None:
                full_audio = audio_data
                # 显式删除不再需要的引用
                del audio_data
                audio_data = None

        # 4. 更新全局音频缓冲区 - 保留用于音频平滑过渡
        updated_audio_buffer = np.concatenate((full_audio_buffer, full_audio))

        # 5. 处理视频
        video_path = segment_files.get('video')
        if not video_path:
            logger.warning("[MediaMixer] create_mixed_segment: 本片段无video_path可用")
            return False, full_audio_buffer

        await add_video_segment(
            video_path=video_path,
            start_time=start_time_param,
            duration=duration,
            audio_data=full_audio,
            output_path=output_path,
            sentences=sentences,
            generate_subtitle=generate_subtitle,
            task_state=task_state,
            sample_rate=sample_rate
        )
        
        # 优化：仅保留最后N秒的音频用于下一批次的过渡，而不是整个缓冲区
        if len(updated_audio_buffer) > sample_rate * 5:  # 例如仅保留最后5秒
            preserve_samples = min(len(updated_audio_buffer), int(sample_rate * 5))
            updated_audio_buffer = updated_audio_buffer[-preserve_samples:]
        
        return True, updated_audio_buffer
        
    except Exception as e:
        logger.exception(f"[MediaMixer] create_mixed_segment 执行出错，错误: {e}")
        return False, full_audio_buffer
    finally:
        # 现有的清理逻辑很好
        if 'full_audio' in locals() and full_audio is not None and full_audio is not updated_audio_buffer: 
            del full_audio
        if 'audio_data' in locals() and audio_data is not None:
            del audio_data

def _concat_audio_segments(sentences: List[Sentence], full_audio_buffer: np.ndarray, overlap: float) -> np.ndarray:
    """拼接所有句子的合成音频"""
    full_audio = np.array([], dtype=np.float32)
    for sentence in sentences:
        if sentence.generated_audio is not None and len(sentence.generated_audio) > 0:
            audio_data = np.asarray(sentence.generated_audio, dtype=np.float32)
            # 关键部分: 应用音频淡入淡出效果以实现平滑过渡
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
        start_time = (sentences[0].adjusted_start - sentences[0].segment_start * 1000) / 1000.0
    duration = sum(s.adjusted_duration for s in sentences) / 1000.0
    return start_time, duration

async def _process_background_audio(
    bg_path: str, start_time: float, duration: float, audio_data: np.ndarray,
    sample_rate: int, vocals_volume: float, background_volume: float, max_val: float
) -> np.ndarray:
    """处理背景音频 - 异步版本"""
    try:
        # 调用异步函数 mix_with_background
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
        # 显式释放大型中间变量
        if 'mixed_audio' in locals():
            del mixed_audio
