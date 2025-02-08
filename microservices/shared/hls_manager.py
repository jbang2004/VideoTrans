import logging
import asyncio
import m3u8
from pathlib import Path

from .ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class HLSManager:
    """
    用于将多个混音后的视频片段转为 HLS 分段并生成 playlist。
    """
    def __init__(self, task_id: str, segments_dir: Path, playlist_path: Path):
        self.task_id = task_id
        self.segments_dir = segments_dir
        self.playlist_path = playlist_path
        self.ffmpeg_tool = FFmpegTool()
        self._lock = asyncio.Lock()

        self.playlist = m3u8.M3U8()
        self.playlist.version = 3
        self.playlist.target_duration = 20
        self.playlist.playlist_type = 'VOD'
        self.playlist.is_endlist = False

        self.has_segments = False
        self.sequence_number = 0
        self._save_playlist()

    def _save_playlist(self):
        for segment in self.playlist.segments:
            if segment.uri and not segment.uri.startswith('/'):
                segment.uri = '/' + segment.uri
        self.playlist.dump(str(self.playlist_path))

    async def add_segment(self, video_path: str, part_index: int):
        """
        将单段 MP4 转为 TS 并添加到playlist
        """
        async with self._lock:
            self.segments_dir.mkdir(parents=True, exist_ok=True)
            segment_filename = f'segment_{self.sequence_number:04d}_%03d.ts'
            segment_pattern = str(self.segments_dir / segment_filename)
            temp_playlist_path = self.segments_dir / f'temp_{part_index}.m3u8'

            await self.ffmpeg_tool.hls_segment(
                input_path=video_path,
                segment_pattern=segment_pattern,
                playlist_path=str(temp_playlist_path),
                hls_time=10
            )

            temp_m3u8 = m3u8.load(str(temp_playlist_path))
            for seg in temp_m3u8.segments:
                seg.uri = f"segments/{self.task_id}/{Path(seg.uri).name}"
                self.playlist.segments.append(seg)

            self.sequence_number += len(temp_m3u8.segments)
            self.has_segments = True
            self._save_playlist()

    async def finalize_playlist(self):
        async with self._lock:
            if self.has_segments:
                self.playlist.is_endlist = True
                self._save_playlist()
                logger.info(f"HLS playlist finalized: {self.playlist_path}")
            else:
                logger.warning("No segments to finalize for HLS")
