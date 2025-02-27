import logging
import asyncio
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import json
import aioredis
from typing import Dict, Any, Optional
from utils.ffmpeg_utils import FFmpegTool
from utils.task_state import TaskState
from utils.redis_decorators import redis_worker_decorator
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

    @redis_worker_decorator(
        input_queue='segment_init_queue',
        next_queue='segment_queue',
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
                
            # 更新总分段数
            task_state.total_segments = len(segments)
            
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

    @redis_worker_decorator(
        input_queue='segment_queue',
        next_queue='asr_queue',
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

            data = item.get('data', item)  # 兼容直接传入数据或包含data字段的情况
            index = data['index']
            start = data['start']
            duration = data['duration']
            video_path = data['video_path']  # 从队列消息中获取视频路径

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

            # 分离人声和背景音
            await self.audio_separator.separate(
                input_path=full_audio,
                vocals_output=vocals_audio,
                background_output=background_audio
            )

            # 保存分段媒体文件信息
            segment_info = {
                'segment_index': index,
                'start': start,
                'duration': duration,
                'silent_video_path': silent_video,
                'full_audio_path': full_audio,
                'vocals_path': vocals_audio,
                'background_path': background_audio
            }
            task_state.segment_media_files[index] = segment_info

            self.logger.info(
                f"[分段提取 Worker] 分段 {index} 处理完成 -> TaskID={task_state.task_id}"
            )
            return segment_info

        except Exception as e:
            self.logger.error(
                f"[分段提取 Worker] 分段 {index if 'index' in locals() else '?'} 处理失败: {e} -> TaskID={task_state.task_id}",
                exc_info=True
            )
            task_state.errors.append({
                'segment_index': index if 'index' in locals() else None,
                'stage': 'segment_extraction',
                'error': str(e)
            })
            return None

async def start():
    """启动 Worker"""
    config_module = __import__('config')
    config = config_module.Config()
    worker = SegmentWorker(config)
    
    # 启动两个并行任务
    await asyncio.gather(
        worker.run_init(),
        worker.run_extract()
    )

if __name__ == '__main__':
    asyncio.run(start())
