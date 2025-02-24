import m3u8
import logging
from pathlib import Path
from typing import Optional, Dict
from .storage import StorageService
from utils.ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class HLSService:
    """HLS 流媒体服务"""
    
    def __init__(self, config, storage_service: StorageService):
        self.config = config
        self.storage = storage_service
        self.playlists: Dict[str, m3u8.M3U8] = {}
        self.logger = logger
        self.ffmpeg_tool = FFmpegTool()
        
    def _create_playlist(self, task_id: str) -> m3u8.M3U8:
        """创建新的播放列表"""
        playlist = m3u8.M3U8()
        playlist.version = 3
        playlist.target_duration = self.config.HLS_SEGMENT_DURATION
        playlist.media_sequence = 0
        playlist.playlist_type = 'VOD'
        playlist.is_endlist = False
        
        self.playlists[task_id] = playlist
        return playlist
        
    async def init_task(self, task_id: str) -> None:
        """初始化任务的HLS资源"""
        playlist = self._create_playlist(task_id)
        await self.storage.update_playlist(task_id, playlist.dumps())
        
    async def add_segment(self, task_id: str, segment_path: Path, segment_index: int) -> bool:
        """添加新的视频片段"""
        try:
            # 1. 使用 FFmpeg 进行 HLS 分段
            segment_filename = f'segment_{segment_index}_%03d.ts'
            target_dir = self.config.PUBLIC_DIR / "segments" / task_id
            target_dir.mkdir(parents=True, exist_ok=True)
            
            segment_pattern = str(target_dir / segment_filename)
            temp_playlist_path = target_dir / f'temp_{segment_index}.m3u8'

            # 调用 FFmpeg 进行分段，添加更多参数以确保准确切片
            await self.ffmpeg_tool.hls_segment(
                input_path=str(segment_path),               
                hls_time=self.config.HLS_TIME,
                segment_pattern=segment_pattern,
                playlist_path=str(temp_playlist_path),
            )

            # 2. 读取临时播放列表并更新主播放列表
            temp_m3u8 = m3u8.load(str(temp_playlist_path))
            
            playlist = self.playlists.get(task_id)
            if not playlist:
                playlist = self._create_playlist(task_id)

            # 添加不连续标记
            discontinuity_segment = m3u8.Segment(discontinuity=True)
            playlist.segments.append(discontinuity_segment)

            # 添加分片
            for segment in temp_m3u8.segments:
                segment.uri = f"/segments/{task_id}/{Path(segment.uri).name}"
                playlist.segments.append(segment)

            # 3. 保存更新后的播放列表
            await self.storage.update_playlist(
                task_id,
                playlist.dumps()
            )
            
            # 4. 清理临时播放列表
            if temp_playlist_path.exists():
                temp_playlist_path.unlink()

            self.logger.debug(f"已添加分片 {segment_index} 到任务 {task_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加分片失败: {e}")
            return False
            
    async def finalize_task(self, task_id: str) -> None:
        """完成任务的HLS处理"""
        try:
            playlist = self.playlists.get(task_id)
            if playlist:
                playlist.is_endlist = True
                await self.storage.update_playlist(
                    task_id,
                    playlist.dumps()
                )
                # 清理缓存
                self.playlists.pop(task_id, None)
                self.logger.info(f"任务 {task_id} 的HLS流已完成")
        except Exception as e:
            self.logger.error(f"完成HLS流失败: {e}")
            
    async def cleanup_task(self, task_id: str) -> None:
        """清理任务的所有HLS资源"""
        try:
            await self.storage.cleanup_task(task_id)
            self.playlists.pop(task_id, None)
            self.logger.info(f"已清理任务 {task_id} 的HLS资源")
        except Exception as e:
            self.logger.error(f"清理HLS资源失败: {e}") 