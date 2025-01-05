import logging
import asyncio
from pathlib import Path
import m3u8
from typing import List, Union
from utils.decorators import handle_errors
from utils.task_storage import TaskPaths
import os.path
import shutil

logger = logging.getLogger(__name__)

class HLSManager:
    """处理 HLS 流媒体相关的功能"""
    def __init__(self, config, task_id: str, task_paths: TaskPaths):
        self.config = config
        self.task_id = task_id
        self.task_paths = task_paths
        self.logger = logging.getLogger(__name__)
        
        # 设置路径 - 确保是 Path 对象
        self.playlist_path = Path(task_paths.playlist_path)
        self.segments_dir = Path(task_paths.segments_dir)  # 确保是 Path 对象
        self.sequence_number = 0
        self._lock = asyncio.Lock()
        
        # 初始化主播放列表
        self.playlist = m3u8.M3U8()
        self.playlist.version = 3
        self.playlist.target_duration = 20
        self.playlist.media_sequence = 0
        self.playlist.playlist_type = 'VOD'
        self.playlist.is_endlist = False
        
        # 新增属性
        self.has_segments = False
        
        # 立即保存初始播放列表
        self._save_playlist()
    
    def _save_playlist(self) -> None:
        """保存播放列表到文件"""
        try:
            # 确保片段路径以斜杠开头
            for segment in self.playlist.segments:
                if segment.uri is not None and not segment.uri.startswith('/'):
                    segment.uri = '/' + segment.uri
            
            with open(self.playlist_path, 'w', encoding='utf-8') as f:
                f.write(self.playlist.dumps())
            logger.info(f"播放列表已保存到: {self.playlist_path}")
            logger.debug(f"播放列表内容: {self.playlist.dumps()}")
        except Exception as e:
            logger.error(f"保存播放列表失败: {e}")
            raise
    
    @handle_errors(None)
    async def add_segment(self, video_path: Union[str, Path], part_index: int) -> None:
        """添加新的视频片段到播放列表"""
        async with self._lock:
            try:
                # 确保目录存在
                if isinstance(self.segments_dir, str):
                    self.segments_dir = Path(self.segments_dir)
                self.segments_dir.mkdir(parents=True, exist_ok=True)
                
                # 生成文件名和路径
                segment_filename = f'segment_{self.sequence_number:04d}_%03d.ts'
                # 使用 processing_segments_dir 来存储临时片段
                segment_pattern = str(self.task_paths.processing_segments_dir / segment_filename)
                temp_playlist_path = self.task_paths.processing_dir / f'temp_{part_index}.m3u8'
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(video_path),
                    '-c', 'copy',
                    '-f', 'hls',
                    '-hls_time', '10',
                    '-hls_list_size', '0',
                    '-hls_segment_type', 'mpegts',
                    '-hls_segment_filename', segment_pattern,
                    str(temp_playlist_path)
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    self.logger.error(f"FFmpeg 错误: {stderr.decode()}")
                    raise RuntimeError(f"FFmpeg 错误: {stderr.decode()}")
                
                # 检查输出文件
                expected_first_segment = str(self.task_paths.processing_segments_dir / f'segment_{self.sequence_number:04d}_000.ts')
                if not os.path.exists(expected_first_segment):
                    self.logger.error(f"片段文件未生成: {expected_first_segment}")
                    raise FileNotFoundError(f"片段文件未生成: {expected_first_segment}")

                # 加载并处理临时播放列表
                temp_m3u8 = m3u8.load(str(temp_playlist_path))
                
                # 添加不连续标记
                discontinuity_segment = m3u8.Segment(discontinuity=True)
                self.playlist.add_segment(discontinuity_segment)
                
                # 修改片段 URI 为相对路径并移动文件
                for segment in temp_m3u8.segments:
                    segment_name = Path(segment.uri).name
                    # 源文件和目标文件路径
                    src_path = self.task_paths.processing_segments_dir / segment_name
                    dst_path = self.segments_dir / segment_name
                    # 移动文件到公共目录
                    shutil.move(str(src_path), str(dst_path))
                    # 更新播放列表中的路径
                    segment.uri = f"segments/{self.task_id}/{segment_name}"
                    # 在添加片段时检查 URI 是否为 None
                    if segment.uri is None:
                        logger.error(f"添加片段 {segment_name} 时 URI 为 None。")
                    self.playlist.segments.append(segment)
                
                self.sequence_number += len(temp_m3u8.segments)
                self.has_segments = True
                self._save_playlist()
                
            finally:
                # 清理临时文件
                temp_playlist = self.task_paths.processing_dir / f'temp_{part_index}.m3u8'
                if os.path.exists(str(temp_playlist)):
                    try:
                        os.unlink(str(temp_playlist))
                    except Exception as e:
                        self.logger.warning(f"清理临时文件失败: {e}")
    
    async def finalize_playlist(self) -> None:
        """标记播放列表为完成状态"""
        # 只有当播放列表中有片段时才标记为结束
        if self.has_segments:
            self.playlist.is_endlist = True
            self._save_playlist()
            self.logger.info("播放列表已保存，并标记为完成状态")
        else:
            self.logger.warning("播放列表为空，不标记为结束状态")