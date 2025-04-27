import logging
import asyncio
from typing import List, Tuple, Dict
from pathlib import Path
import time
import ray
from ray import serve

from utils.ffmpeg_utils import get_duration
from config import Config

# 配置日志
logger = logging.getLogger(__name__)

@serve.deployment(
    name="video_segmenter",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}
)
class VideoSegmenter:
    """
    视频分段器：负责获取视频时长并划分为多个时间片段
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()

    async def segment_video(self, video_path: str) -> Dict:
        """
        分析视频并划分为多个时间片段（已简化：始终作为整段处理）
        """
        try:
            start_time = time.time()
            # 始终获取完整视频时长并作为单段处理
            duration = await self._get_video_duration(video_path)
            segments = [(0, duration)]
            elapsed = time.time() - start_time
            self.logger.info(f"视频分段跳过，处理整段视频：总长度={duration:.2f}s, 段数=1, 耗时={elapsed:.2f}s")
            return {"status": "success", "duration": duration, "segments": segments}
        except Exception as e:
            self.logger.exception(f"视频分段失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _get_video_duration(self, video_path: str) -> float:
        """
        获取视频时长
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频时长（秒）
        """
        try:
            # 使用异步函数获取视频时长
            duration = await get_duration(video_path)
            self.logger.debug(f"获取到视频时长: {duration:.2f}秒")
            return duration
        except Exception as e:
            self.logger.error(f"获取视频时长失败: {e}")
            raise
    
    def _get_audio_segments(self, duration: float, segment_minutes: float, min_segment_minutes: float) -> List[Tuple[float, float]]:
        """
        将视频按照配置的时长分割为多个时间片段 - 同步方法
        
        Args:
            duration: 视频总时长（秒）
            segment_minutes: 每个分段的分钟数
            min_segment_minutes: 最小分段分钟数
            
        Returns:
            分段列表，每个元素为(开始时间, 持续时间)的元组
        """
        try:
            segment_length = segment_minutes * 60
            min_length = min_segment_minutes * 60

            if duration <= min_length:
                return [(0, duration)]

            segments = []
            current_pos = 0.0

            while current_pos < duration:
                remaining_duration = duration - current_pos

                if remaining_duration <= segment_length:
                    # 如果剩余片段过短且已有片段，和前一个合并
                    if remaining_duration < min_length and segments:
                        start = segments[-1][0]
                        new_duration = duration - start
                        segments[-1] = (start, new_duration)
                    else:
                        segments.append((current_pos, remaining_duration))
                    break

                segments.append((current_pos, segment_length))
                current_pos += segment_length

            self.logger.debug(f"视频分段完成: {len(segments)}个分段")
            return segments
        except Exception as e:
            self.logger.error(f"视频分段失败: {e}")
            raise 