# core/hls_manager.py
import logging
import asyncio
from pathlib import Path
import m3u8
from typing import List, Union
from utils.decorators import handle_errors
from utils.task_storage import TaskPaths
import os.path
import shutil

# 引入统一的 FFmpegTool
from utils.ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class HLSManager:
    """处理 HLS 流媒体相关的功能"""
    def __init__(self, config, task_id: str, task_paths: TaskPaths):
        self.config = config
        self.task_id = task_id
        self.task_paths = task_paths
        self.logger = logging.getLogger(__name__)

        self.playlist_path = Path(task_paths.playlist_path)
        self.segments_dir = Path(task_paths.segments_dir)
        self.sequence_number = 0
        self._lock = asyncio.Lock()

        self.playlist = m3u8.M3U8()
        self.playlist.version = 3
        self.playlist.target_duration = 20
        self.playlist.media_sequence = 0
        self.playlist.playlist_type = 'VOD'
        self.playlist.is_endlist = False

        # 引入统一 ffmpeg 工具
        self.ffmpeg_tool = FFmpegTool()

        self.has_segments = False

        self._save_playlist()

    def _save_playlist(self) -> None:
        """保存播放列表到文件"""
        try:
            for segment in self.playlist.segments:
                # 确保 URI 带有斜杠
                if segment.uri is not None and not segment.uri.startswith('/'):
                    segment.uri = '/' + segment.uri

            with open(self.playlist_path, 'w', encoding='utf-8') as f:
                f.write(self.playlist.dumps())
            self.logger.info("播放列表已更新")
        except Exception as e:
            self.logger.error(f"保存播放列表失败: {e}")
            raise

    @handle_errors(None)
    async def add_segment(self, video_path: Union[str, Path], part_index: int) -> None:
        """添加新的视频片段到播放列表"""
        async with self._lock:
            try:
                self.segments_dir.mkdir(parents=True, exist_ok=True)

                segment_filename = f'segment_{self.sequence_number:04d}_%03d.ts'
                segment_pattern = str(self.segments_dir / segment_filename)
                temp_playlist_path = self.task_paths.processing_dir / f'temp_{part_index}.m3u8'

                # 统一使用 ffmpeg_tool.hls_segment
                await self.ffmpeg_tool.hls_segment(
                    input_path=str(video_path),
                    segment_pattern=segment_pattern,
                    playlist_path=str(temp_playlist_path),
                    hls_time=10
                )

                # 加入分段
                temp_m3u8 = m3u8.load(str(temp_playlist_path))
                discontinuity_segment = m3u8.Segment(discontinuity=True)
                self.playlist.add_segment(discontinuity_segment)

                for segment in temp_m3u8.segments:
                    segment.uri = f"segments/{self.task_id}/{Path(segment.uri).name}"
                    self.playlist.segments.append(segment)

                self.sequence_number += len(temp_m3u8.segments)
                self.has_segments = True
                self._save_playlist()

            finally:
                if os.path.exists(str(temp_playlist_path)):
                    try:
                        os.unlink(str(temp_playlist_path))
                    except Exception as e:
                        self.logger.warning(f"清理临时文件失败: {e}")

    async def finalize_playlist(self) -> None:
        """标记播放列表为完成状态"""
        if self.has_segments:
            self.playlist.is_endlist = True
            self._save_playlist()
            self.logger.info("播放列表已保存，并标记为完成状态")
        else:
            self.logger.warning("播放列表为空，不标记为结束状态")
