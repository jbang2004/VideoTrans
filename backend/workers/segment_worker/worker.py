import logging
import asyncio
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional
from utils.ffmpeg_utils import FFmpegTool
from utils.task_state import TaskState
from utils.decorators import worker_decorator
from utils import concurrency
from .audio_separator import ClearVoiceSeparator
from .video_segmenter import VideoSegmenter

logger = logging.getLogger(__name__)

class SegmentWorker:
    """
    视频分段处理 Worker：负责分段初始化以及媒体提取（音频、视频和后续音频分离）。
    """

    def __init__(self, config):
        """初始化 SegmentWorker 及其依赖"""
        self.config = config
        self.target_sr = 24000  # 从config获取
        
        # 初始化依赖
        self.audio_separator = ClearVoiceSeparator(model_name='MossFormer2_SE_48K')
        self.ffmpeg_tool = FFmpegTool()
        self.video_segmenter = VideoSegmenter(config=config, ffmpeg_tool=self.ffmpeg_tool)
        self.logger = logger

    @worker_decorator(
        input_queue_attr='segment_init_queue',
        next_queue_attr='segment_queue',
        worker_name='分段初始化 Worker',
        mode='stream'
    )
    async def run_init(self, item, task_state: TaskState):
        """处理视频分段初始化任务"""
        try:
            video_path = item['video_path']
            duration = await self.video_segmenter.get_video_duration(video_path)
            if duration <= 0:
                raise ValueError(f"无效的视频时长: {duration}s")
            segments = await self.video_segmenter.get_audio_segments(duration)
            if not segments:
                raise ValueError("无法获取有效分段")
            self.logger.info(
                f"[分段初始化 Worker] 视频总长={duration:.2f}s, 分段数={len(segments)}, TaskID={task_state.task_id}"
            )
            for i, (start, seg_duration) in enumerate(segments):
                yield {
                    'index': i,
                    'start': start,
                    'duration': seg_duration,
                    'video_path': video_path  # 传递视频路径给下游
                }
        except Exception as e:
            self.logger.error(f"[分段初始化 Worker] 处理失败: {e} -> TaskID={task_state.task_id}", exc_info=True)
            task_state.errors.append({
                'stage': 'segment_initialization',
                'error': str(e)
            })

    @worker_decorator(
        input_queue_attr='segment_queue',
        next_queue_attr='asr_queue',
        worker_name='分段提取 Worker'
    )
    async def run_extract(self, item, task_state: TaskState) -> Dict[str, Any]:
        """
        处理单个视频分段，执行：
          1. 并发提取音频与视频；
          2. 分离人声和背景音；
          3. 重采样与写文件；
          4. 清理临时文件，并返回提取信息。
        """
        try:
            if item is None:
                return None

            index = item['index']
            start = item['start']
            duration = item['duration']
            video_path = item['video_path']  # 从队列消息中获取视频路径

            self.logger.debug(
                f"[分段提取 Worker] 开始处理分段 {index}, start={start:.2f}s, duration={duration:.2f}s -> TaskID={task_state.task_id}"
            )
            silent_video = str(task_state.task_paths.processing_dir / f"video_silent_{index}.mp4")
            full_audio = str(task_state.task_paths.processing_dir / f"audio_full_{index}.wav")
            vocals_audio = str(task_state.task_paths.processing_dir / f"vocals_{index}.wav")
            background_audio = str(task_state.task_paths.processing_dir / f"background_{index}.wav")

            # 并发提取音频与视频
            await asyncio.gather(
                self.ffmpeg_tool.extract_audio(video_path, full_audio, start, duration),
                self.ffmpeg_tool.extract_video(video_path, silent_video, start, duration)
            )

            # 分离人声与背景（同步调用用 asyncio.to_thread 包装）
            vocals, background, sr = await asyncio.to_thread(self.audio_separator.separate_audio, full_audio)
            background = await asyncio.to_thread(self._resample_audio_sync, sr, background, self.target_sr)

            await asyncio.gather(
                asyncio.to_thread(sf.write, vocals_audio, vocals, sr, subtype='FLOAT'),
                asyncio.to_thread(sf.write, background_audio, background, self.target_sr, subtype='FLOAT')
            )

            Path(full_audio).unlink(missing_ok=True)

            media_files = {
                'video': silent_video,
                'vocals': vocals_audio,
                'background': background_audio,
                'duration': len(vocals) / sr
            }
            task_state.segment_media_files[index] = media_files

            return {
                'segment_index': index,
                'vocals_path': vocals_audio,
                'start': start,
                'duration': media_files['duration']
            }
        except Exception as e:
            self.logger.error(
                f"[分段提取 Worker] 分段处理失败: {e} -> TaskID={task_state.task_id}",
                exc_info=True
            )
            task_state.errors.append({
                'stage': 'segment_extraction',
                'error': str(e)
            })
            return None

    def _resample_audio_sync(self, fs: int, audio: np.ndarray, target_sr: int) -> np.ndarray:
        """
        同步方式进行音频归一化和重采样
          - audio: 待处理音频数据
          - fs: 源采样率
          - target_sr: 目标采样率
        """
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

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

if __name__ == '__main__':
    print("Segment Worker 模块加载成功")
