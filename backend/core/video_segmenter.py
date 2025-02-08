import logging
from typing import List, Tuple
from utils.ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class VideoSegmenter:
    """视频分段计算器，负责计算视频应该如何分段"""
    
    def __init__(self, config, ffmpeg_tool: FFmpegTool):
        self.config = config
        self.ffmpeg_tool = ffmpeg_tool
        self.logger = logger
        
    async def get_video_duration(self, video_path: str) -> float:
        """获取视频时长"""
        return await self.ffmpeg_tool.get_duration(video_path)
        
    async def get_audio_segments(self, duration: float) -> List[Tuple[float, float]]:
        """
        按配置分割时间片：
        segment_length: 每段的理想长度
        min_length: 最小允许的分段时长，若最后一个分段不足该时长则与前一段合并
        """
        segment_length = self.config.SEGMENT_MINUTES * 60
        min_length = self.config.MIN_SEGMENT_MINUTES * 60

        if duration <= min_length:
            return [(0, duration)]

        segments = []
        current_pos = 0.0

        while current_pos < duration:
            remaining_duration = duration - current_pos
            
            if remaining_duration <= segment_length:
                if remaining_duration < min_length and segments:
                    # 合并到上一段
                    start = segments[-1][0]
                    new_duration = duration - start
                    segments[-1] = (start, new_duration)
                else:
                    segments.append((current_pos, remaining_duration))
                break

            segments.append((current_pos, segment_length))
            current_pos += segment_length

        return segments 