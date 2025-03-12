# core/hls_manager_actor.py
import logging
import m3u8
import os.path
import shutil
import time
from pathlib import Path
from typing import Union, Optional
import ray

from utils.ffmpeg_utils import FFmpegTool
from utils.task_storage import TaskPaths
from config import Config

logger = logging.getLogger(__name__)

@ray.remote
class HLSManagerActor:
    """处理 HLS 流媒体相关的功能"""
    def __init__(self, config, task_id: str, task_paths: TaskPaths):
        self.config = config
        self.task_id = task_id
        self.task_paths = task_paths
        self.logger = logging.getLogger(__name__)

        self.playlist_path = Path(task_paths.playlist_path)
        self.segments_dir = Path(task_paths.segments_dir)
        self.sequence_number = 0
        # Ray Actor 已提供并发控制

        self.playlist = m3u8.M3U8()
        self.playlist.version = 3
        self.playlist.target_duration = 20
        self.playlist.media_sequence = 0
        self.playlist.playlist_type = 'VOD'
        # 不设置is_endlist，让播放器可以实时加载新分段
        self.playlist.is_endlist = False

        # 添加segment_time属性，默认为10秒
        self.segment_time = 10

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

            self.logger.info(f"保存播放列表到: {self.playlist_path}")
            # 确保目录存在
            self.playlist_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.playlist_path, 'w', encoding='utf-8') as f:
                content = self.playlist.dumps()
                f.write(content)
                
            self.logger.info(f"播放列表已更新，总计{len(self.playlist.segments)}个分段")
        except Exception as e:
            self.logger.error(f"保存播放列表失败: {e}")
            raise

    def add_segment(self, video_path: Union[str, Path], part_index: int) -> bool:
        """添加新的视频片段到播放列表（同步方法）"""
        start_time = time.time()
        try:
            self.segments_dir.mkdir(parents=True, exist_ok=True)

            segment_filename = f'segment_{self.sequence_number:04d}_%03d.ts'
            segment_pattern = str(self.segments_dir / segment_filename)
            temp_playlist_path = self.task_paths.processing_dir / f'temp_{part_index}.m3u8'

            # 直接使用同步版本的hls_segment方法
            self.ffmpeg_tool.hls_segment(
                input_path=str(video_path),
                segment_pattern=segment_pattern,
                playlist_path=str(temp_playlist_path),
                hls_time=self.segment_time
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
            
            # 确保播放列表不标记为结束，以便实时加载
            self.playlist.is_endlist = False
            
            self._save_playlist()
            
            # 删除临时播放列表
            if temp_playlist_path.exists():
                temp_playlist_path.unlink()
            
            elapsed = time.time() - start_time
            self.logger.info(f"已添加片段 {part_index} 到HLS流，耗时 {elapsed:.2f}s")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"添加HLS片段失败: {e}，耗时 {elapsed:.2f}s")
            return False

    def finalize_playlist(self) -> bool:
        """标记播放列表为完成状态"""
        try:
            if self.has_segments:
                self.playlist.is_endlist = True
                self._save_playlist()
                self.logger.info("播放列表已保存，并标记为完成状态")
                return True
            else:
                self.logger.warning("播放列表为空，不标记为结束状态")
                return False
        except Exception as e:
            self.logger.error(f"完成播放列表失败: {e}")
            return False

    def get_has_segments(self):
        """获取has_segments属性的值"""
        return self.has_segments 