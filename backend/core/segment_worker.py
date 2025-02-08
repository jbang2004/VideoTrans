import logging
import asyncio
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Any

from utils.ffmpeg_utils import FFmpegTool
from utils.task_state import TaskState
from utils.decorators import handle_errors, worker_decorator
from utils import concurrency
from core.audio_separator import ClearVoiceSeparator
from core.video_segmenter import VideoSegmenter

logger = logging.getLogger(__name__)

class SegmentWorker:
    """视频分段处理Worker"""
    
    def __init__(
        self,
        config,
        audio_separator: ClearVoiceSeparator,
        video_segmenter: VideoSegmenter,
        ffmpeg_tool: FFmpegTool,
        target_sr: int = 24000
    ):
        self.config = config
        self.target_sr = target_sr
        self.audio_separator = audio_separator
        self.video_segmenter = video_segmenter
        self.ffmpeg_tool = ffmpeg_tool
        self.logger = logger

    @worker_decorator(
        input_queue_attr='segment_init_queue',
        next_queue_attr='segment_queue',
        worker_name='分段初始化Worker'
    )
    async def segment_init_maker(self, task_info: Dict[str, Any], task_state: TaskState) -> bool:
        """处理视频分段初始化"""
        try:
            # 获取视频时长
            duration = await self.video_segmenter.get_video_duration(task_state.video_path)
            if duration <= 0:
                raise ValueError(f"无效的视频时长: {duration}s")

            # 计算分段
            segments = await self.video_segmenter.get_audio_segments(duration)
            if not segments:
                raise ValueError("无法获取有效分段")

            self.logger.info(
                f"[分段初始化Worker] 视频总长={duration:.2f}s, "
                f"分段数={len(segments)}, TaskID={task_state.task_id}"
            )

            # 将分段信息放入处理队列
            for i, (start, duration) in enumerate(segments):
                await task_state.segment_queue.put({
                    'index': i,
                    'start': start,
                    'duration': duration
                })
            return True

        except Exception as e:
            self.logger.error(f"[分段初始化Worker] 处理失败: {e} -> TaskID={task_state.task_id}")
            task_state.errors.append({
                'stage': 'segment_initialization',
                'error': str(e)
            })
            return False

    @worker_decorator(
        input_queue_attr='segment_queue',
        next_queue_attr='asr_queue',
        worker_name='分段提取Worker'
    )
    async def segment_media_maker(self, segment_info: Dict[str, Any], task_state: TaskState) -> Dict[str, Any]:
        """
        处理单个视频分段：
          1. 并发提取音频和视频。
          2. 调用音频分离器分离人声和背景音乐（同步函数通过 concurrency.run_sync 调用）。
          3. 使用同步方式进行重采样与写文件操作，同样通过 concurrency.run_sync 调用。
          4. 最后删除临时生成的全音频文件，并构造下游所需的媒体文件信息。
        """
        try:
            segment_index = segment_info['index']
            start = segment_info['start']
            duration = segment_info['duration']
            
            self.logger.debug(
                f"[分段提取Worker] 开始处理分段 {segment_index}, "
                f"start={start:.2f}s, duration={duration:.2f}s -> "
                f"TaskID={task_state.task_id}"
            )
            
            # 构造输出文件路径
            silent_video = str(task_state.task_paths.processing_dir / f"video_silent_{segment_index}.mp4")
            full_audio = str(task_state.task_paths.processing_dir / f"audio_full_{segment_index}.wav")
            vocals_audio = str(task_state.task_paths.processing_dir / f"vocals_{segment_index}.wav")
            background_audio = str(task_state.task_paths.processing_dir / f"background_{segment_index}.wav")
            
            # (1) 并发提取音频和视频（ffmpeg工具本身为异步实现）
            await asyncio.gather(
                self.ffmpeg_tool.extract_audio(task_state.video_path, full_audio, start, duration),
                self.ffmpeg_tool.extract_video(task_state.video_path, silent_video, start, duration)
            )
            
            # (2) 分离人声：采用 concurrency.run_sync 封装同步调用
            vocals, background, sr = await concurrency.run_sync(
                lambda: self.audio_separator.separate_audio(full_audio)
            )
            
            # (3) 重采样背景音乐：为保证与目标采样率一致，调用同步重采样函数
            background = await concurrency.run_sync(
                lambda: self._resample_audio_sync(sr, background, self.target_sr)
            )
            
            # (4) 写入音频文件：分别存储人声和背景，均使用同步写文件方法
            await asyncio.gather(
                concurrency.run_sync(lambda: self._write_audio_file(vocals_audio, vocals, sr)),
                concurrency.run_sync(lambda: self._write_audio_file(background_audio, background, self.target_sr))
            )
            
            # 删除临时生成的全音频文件
            Path(full_audio).unlink(missing_ok=True)
            
            # 准备输出信息
            media_files = {
                'video': silent_video,
                'vocals': vocals_audio,
                'background': background_audio,
                'duration': len(vocals) / sr
            }
            
            # 存储分段文件信息
            task_state.segment_media_files[segment_index] = media_files
            
            # 返回ASR所需信息
            return {
                'segment_index': segment_index,
                'vocals_path': vocals_audio,
                'start': start,
                'duration': media_files['duration']
            }
            
        except Exception as e:
            self.logger.error(
                f"[分段提取Worker] 分段 {segment_info['index']} 处理失败: {e} -> "
                f"TaskID={task_state.task_id}"
            )
            task_state.errors.append({
                'segment_index': segment_info['index'],
                'stage': 'segment_extraction',
                'error': str(e)
            })
            return None

    def _resample_audio_sync(self, fs: int, audio: np.ndarray, target_sr: int) -> np.ndarray:
        """
        同步方式进行音频归一化和重采样（参考 media_utils 中 normalize_and_resample 的实现）
          - audio: 待处理音频数据
          - fs: 源采样率
          - target_sr: 目标采样率
        """
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
            
        # 如果是多通道，则转换为单通道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=-1)
            
        if fs != target_sr:
            audio = np.ascontiguousarray(audio)
            resampler = torchaudio.transforms.Resample(
                orig_freq=fs,
                new_freq=target_sr,
                dtype=torch.float32
            )
            audio = resampler(torch.from_numpy(audio)[None, :])[0].numpy()
            
        return audio
        
    def _write_audio_file(self, file_path: str, audio: np.ndarray, sr: int) -> None:
        """
        同步写入音频文件（依赖 soundfile 库），类型和采样率由传入参数控制。
        """
        sf.write(file_path, audio, sr, subtype='FLOAT') 