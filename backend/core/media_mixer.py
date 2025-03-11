# ---------------------------------------------------
# backend/core/media_mixer.py (精简版)
# ---------------------------------------------------
import numpy as np
import logging
import asyncio
from typing import List, Optional
from pathlib import Path

from utils.decorators import handle_errors
from utils.ffmpeg_utils import FFmpegTool
from utils.audio_utils import apply_fade_effect, mix_with_background, normalize_audio
from utils.subtitle_utils import generate_subtitles_for_segment
from utils.video_utils import add_video_segment, concat_video_segments
from config import Config
from core.sentence_tools import Sentence
from utils.task_state import TaskState

logger = logging.getLogger(__name__)

class MediaMixer:
    """
    用于将多段句子的合成音频与原视频片段混合，并可生成带字幕的视频。
    支持:
      - 音频淡入淡出
      - 背景音乐混合
      - 基于 pysubs2 生成 .ass 字幕("YouTube风格")
      - 按语言自动决定单行最大长度
    """
    def __init__(self, config: Config, sample_rate: int, ffmpeg_tool=None):
        self.config = config
        self.sample_rate = sample_rate
        self.max_val = 1.0
        self.overlap = self.config.AUDIO_OVERLAP
        self.vocals_volume = self.config.VOCALS_VOLUME
        self.background_volume = self.config.BACKGROUND_VOLUME
        self.full_audio_buffer = np.array([], dtype=np.float32)
        self.ffmpeg_tool = ffmpeg_tool or FFmpegTool()

    @handle_errors(logger)
    async def mixed_media_maker(
        self,
        sentences: List[Sentence],
        task_state: TaskState,
        output_path: str,
        generate_subtitle: bool = False
    ) -> bool:
        """
        主入口: 处理一批句子的音频与视频，输出一段带音频的 MP4。
        根据 generate_subtitle 决定是否烧制字幕。
        """
        if not sentences:
            logger.warning("mixed_media_maker: 收到空的句子列表")
            return False

        segment_index = sentences[0].segment_index
        segment_files = task_state.segment_media_files.get(segment_index)
        if not segment_files:
            logger.error(f"找不到分段 {segment_index} 对应的媒体文件信息")
            return False

        # 1. 拼接所有句子的合成音频
        full_audio = self._concat_audio_segments(sentences)
        if len(full_audio) == 0:
            logger.error("mixed_media_maker: 没有有效的合成音频数据")
            return False

        # 2. 计算时间参数
        start_time, duration = self._calculate_time_params(sentences)

        # 3. 背景音乐混合
        background_audio_path = segment_files.get('background')
        if background_audio_path:
            full_audio = self._process_background_audio(
                background_audio_path, start_time, duration, full_audio
            )

        # 4. 更新全局音频缓冲区
        self.full_audio_buffer = np.concatenate((self.full_audio_buffer, full_audio))

        # 5. 处理视频
        video_path = segment_files.get('video')
        if not video_path:
            logger.warning("mixed_media_maker: 本片段无video_path可用")
            return False
            
        await add_video_segment(
            video_path=video_path,
            start_time=start_time,
            duration=duration,
            audio_data=full_audio,
            output_path=output_path,
            sentences=sentences,
            generate_subtitle=generate_subtitle,
            task_state=task_state,
            sample_rate=self.sample_rate,
            ffmpeg_tool=self.ffmpeg_tool
        )
        return True

    def _concat_audio_segments(self, sentences: List[Sentence]) -> np.ndarray:
        """拼接所有句子的合成音频"""
        full_audio = np.array([], dtype=np.float32)
        for sentence in sentences:
            if sentence.generated_audio is not None:
                audio_data = np.asarray(sentence.generated_audio, dtype=np.float32)
                if len(full_audio) > 0:
                    audio_data = apply_fade_effect(audio_data, self.full_audio_buffer, self.overlap)
                full_audio = np.concatenate((full_audio, audio_data))
            else:
                logger.warning(
                    "句子音频生成失败: text=%r, UUID=%s",
                    sentence.raw_text,
                    sentence.model_input.get("uuid", "unknown")
                )
        return full_audio

    def _calculate_time_params(self, sentences: List[Sentence]) -> tuple:
        """计算时间参数"""
        start_time = 0.0
        if not sentences[0].is_first:
            start_time = (sentences[0].adjusted_start - sentences[0].segment_start * 1000) / 1000.0
        duration = sum(s.adjusted_duration for s in sentences) / 1000.0
        return start_time, duration

    def _process_background_audio(
        self, bg_path: str, start_time: float, duration: float, audio_data: np.ndarray
    ) -> np.ndarray:
        """处理背景音频"""
        audio_data = mix_with_background(
            bg_path=bg_path,
            start_time=start_time,
            duration=duration,
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            vocals_volume=self.vocals_volume,
            background_volume=self.background_volume
        )
        return normalize_audio(audio_data, self.max_val)

    @handle_errors(logger)
    async def process_and_add_segment(
        self,
        sentences_batch: List[Sentence],
        task_state: TaskState,
        hls_manager=None
    ) -> bool:
        """处理一批句子并添加到HLS流中"""
        if not sentences_batch:
            logger.warning("process_and_add_segment: 收到空的句子列表")
            return False
            
        try:
            seg_index = sentences_batch[0].segment_index
            batch_counter = task_state.batch_counter
            logger.info(f"[MediaMixer] 开始处理分段 {seg_index}, 批次 {batch_counter}, 句子数 {len(sentences_batch)}")
            
            # 生成输出路径
            output_path = task_state.task_paths.segments_dir / f"segment_{batch_counter}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 处理音视频
            start_time = asyncio.get_event_loop().time()
            success = await self.mixed_media_maker(
                sentences=sentences_batch,
                task_state=task_state,
                output_path=str(output_path),
                generate_subtitle=task_state.generate_subtitle
            )
            
            if not success:
                logger.error(f"[MediaMixer] 分段 {batch_counter} 处理失败, TaskID={task_state.task_id}")
                return False
                
            # 处理成功后的操作
            processing_time = asyncio.get_event_loop().time() - start_time
            
            # 添加到HLS流
            if hls_manager:
                hls_start_time = asyncio.get_event_loop().time()
                await hls_manager.add_segment(str(output_path), batch_counter)
                hls_time = asyncio.get_event_loop().time() - hls_start_time
                logger.info(
                    f"[MediaMixer] 分段 {batch_counter} 已加入 HLS, "
                    f"处理耗时: {processing_time:.2f}s, HLS耗时: {hls_time:.2f}s, "
                    f"TaskID={task_state.task_id}"
                )
            else:
                logger.info(
                    f"[MediaMixer] 分段 {batch_counter} 处理完成 (无HLS), "
                    f"处理耗时: {processing_time:.2f}s, TaskID={task_state.task_id}"
                )
            
            # 更新任务状态
            task_state.merged_segments.append(str(output_path))
            task_state.batch_counter += 1
            return True
                
        except Exception as e:
            logger.exception(f"[MediaMixer] 处理分段时发生异常: {str(e)}, TaskID={task_state.task_id}")
            return False

    async def reset(self):
        """重置 full_audio_buffer"""
        self.full_audio_buffer = np.array([], dtype=np.float32)
        logger.debug("MediaMixer 已重置 full_audio_buffer")

    @handle_errors(logger)
    async def concat_segments(self, task_state: TaskState, hls_manager=None) -> Optional[Path]:
        """合并所有处理后的视频分段"""
        if not task_state.merged_segments:
            logger.warning(f"[MediaMixer] 无可合并的视频分段, TaskID={task_state.task_id}")
            return None
            
        # 创建最终输出路径
        final_path = task_state.task_paths.output_dir / f"final_{task_state.task_id}.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 调用视频合并工具函数
        return await concat_video_segments(
            task_state=task_state,
            output_path=final_path,
            ffmpeg_tool=self.ffmpeg_tool,
            hls_manager=hls_manager
        )
