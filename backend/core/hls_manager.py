# core/hls_manager.py
import logging
import m3u8
import os.path
import shutil
import time
import asyncio
from pathlib import Path
from typing import Union, Optional, Dict
from ray import serve

from utils.ffmpeg_utils import hls_segment
from utils.task_storage import TaskPaths
from config import Config

logger = logging.getLogger(__name__)

@serve.deployment(
    name="hls_manager"
)
class HLSManager:
    """HLS流媒体管理器 - 支持多任务管理"""
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        
        # 存储每个任务的HLS管理信息
        self.task_managers = {}
        
        # 并发锁，每个任务一个锁
        self.locks = {}
        
        self.logger.info("HLS管理器已初始化")

    async def create_manager(self, task_id: str, task_paths: TaskPaths) -> Dict:
        """
        为特定任务创建HLS管理器
        
        Args:
            task_id: 任务ID
            task_paths: 任务路径信息
            
        Returns:
            Dict: 包含状态信息的字典
        """
        # 如果任务已存在，直接返回
        if task_id in self.task_managers:
            self.logger.info(f"任务 {task_id} 的HLS管理器已存在")
            return {"status": "success", "message": "任务HLS管理器已存在"}
        
        # 创建任务锁
        self.locks[task_id] = asyncio.Lock()
        
        async with self.locks[task_id]:
            try:
                playlist_path = Path(task_paths.playlist_path)
                segments_dir = Path(task_paths.segments_dir)
                
                # 创建播放列表
                playlist = m3u8.M3U8()
                playlist.version = 3
                playlist.target_duration = 20
                playlist.media_sequence = 0
                playlist.playlist_type = 'EVENT'
                playlist.is_endlist = False
                
                # 存储任务管理信息
                self.task_managers[task_id] = {
                    "task_paths": task_paths,
                    "playlist_path": playlist_path,
                    "segments_dir": segments_dir,
                    "playlist": playlist,
                    "sequence_number": 0,
                    "has_segments": False,
                    "segment_time": 10,  # 默认分段时间为10秒
                    "created_at": time.time()
                }
                
                # 保存初始播放列表
                await self._save_playlist(task_id)
                
                self.logger.info(f"已为任务 {task_id} 创建HLS管理器")
                return {"status": "success", "message": "HLS管理器创建成功"}
            except Exception as e:
                self.logger.error(f"为任务 {task_id} 创建HLS管理器失败: {e}")
                # 清理已创建的部分资源
                if task_id in self.task_managers:
                    del self.task_managers[task_id]
                if task_id in self.locks:
                    del self.locks[task_id]
                return {"status": "error", "message": f"创建HLS管理器失败: {str(e)}"}

    async def _save_playlist(self, task_id: str) -> None:
        """
        保存特定任务的播放列表到文件
        
        Args:
            task_id: 任务ID
        """
        if task_id not in self.task_managers:
            raise ValueError(f"任务 {task_id} 的HLS管理器不存在")
            
        manager = self.task_managers[task_id]
        playlist = manager["playlist"]
        playlist_path = manager["playlist_path"]
        
        try:
            for segment in playlist.segments:
                # 确保 URI 带有斜杠
                if segment.uri is not None and not segment.uri.startswith('/'):
                    segment.uri = '/' + segment.uri

            self.logger.info(f"保存播放列表到: {playlist_path}, 任务ID={task_id}")
            # 确保目录存在
            playlist_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用asyncio.to_thread避免阻塞
            await asyncio.to_thread(self._write_playlist, playlist, playlist_path)
                
            self.logger.info(f"播放列表已更新，总计{len(playlist.segments)}个分段, 任务ID={task_id}")
        except Exception as e:
            self.logger.error(f"保存播放列表失败: {e}, 任务ID={task_id}")
            raise
    
    def _write_playlist(self, playlist, playlist_path):
        """同步写入播放列表文件"""
        with open(playlist_path, 'w', encoding='utf-8') as f:
            content = playlist.dumps()
            f.write(content)

    async def add_segment(self, task_id: str, video_path: Union[str, Path], part_index: int) -> Dict:
        """
        添加新的视频片段到播放列表
        
        Args:
            task_id: 任务ID
            video_path: 视频片段路径
            part_index: 部分索引
            
        Returns:
            Dict: 包含状态信息的字典
        """
        if task_id not in self.task_managers:
            self.logger.error(f"任务 {task_id} 的HLS管理器不存在")
            return {"status": "error", "message": "任务HLS管理器不存在"}
            
        if task_id not in self.locks:
            self.locks[task_id] = asyncio.Lock()
            
        start_time = time.time()
        
        async with self.locks[task_id]:
            try:
                manager = self.task_managers[task_id]
                segments_dir = manager["segments_dir"]
                playlist = manager["playlist"]
                sequence_number = manager["sequence_number"]
                segment_time = manager["segment_time"]
                task_paths = manager["task_paths"]
                
                self.logger.info(f"开始处理HLS片段 {part_index}, 任务ID={task_id}")
                segments_dir.mkdir(parents=True, exist_ok=True)

                segment_filename = f'segment_{sequence_number:04d}_%03d.ts'
                segment_pattern = str(segments_dir / segment_filename)
                temp_playlist_path = task_paths.processing_dir / f'temp_{part_index}.m3u8'

                # 使用异步函数
                await hls_segment(
                    input_path=str(video_path),
                    segment_pattern=segment_pattern,
                    playlist_path=str(temp_playlist_path),
                    hls_time=segment_time
                )

                # 加入分段
                # 使用asyncio.to_thread避免阻塞
                temp_m3u8 = await asyncio.to_thread(m3u8.load, str(temp_playlist_path))
                
                discontinuity_segment = m3u8.Segment(discontinuity=True)
                playlist.add_segment(discontinuity_segment)

                for segment in temp_m3u8.segments:
                    segment.uri = f"segments/{task_id}/{Path(segment.uri).name}"
                    playlist.segments.append(segment)

                # 更新序列号
                manager["sequence_number"] += len(temp_m3u8.segments)
                manager["has_segments"] = True
                
                # 确保播放列表不标记为结束，以便实时加载
                playlist.is_endlist = False
                
                await self._save_playlist(task_id)
                
                # 删除临时播放列表
                if temp_playlist_path.exists():
                    await asyncio.to_thread(temp_playlist_path.unlink)
                
                elapsed = time.time() - start_time
                self.logger.info(f"已添加片段 {part_index} 到HLS流，耗时 {elapsed:.2f}s, 任务ID={task_id}")
                return {"status": "success", "part_index": part_index}
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"添加HLS片段失败: {e}，耗时 {elapsed:.2f}s, 任务ID={task_id}")
                return {"status": "error", "message": f"添加HLS片段失败: {str(e)}"}

    async def finalize_playlist(self, task_id: str) -> Dict:
        """
        标记播放列表为完成状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 包含状态信息的字典
        """
        if task_id not in self.task_managers:
            self.logger.error(f"任务 {task_id} 的HLS管理器不存在")
            return {"status": "error", "message": "任务HLS管理器不存在"}
            
        if task_id not in self.locks:
            self.locks[task_id] = asyncio.Lock()
            
        async with self.locks[task_id]:
            try:
                manager = self.task_managers[task_id]
                playlist = manager["playlist"]
                has_segments = manager["has_segments"]
                
                if has_segments:
                    playlist.is_endlist = True
                    await self._save_playlist(task_id)
                    self.logger.info(f"播放列表已保存，并标记为完成状态, 任务ID={task_id}")
                    return {"status": "success", "message": "播放列表已标记为完成"}
                else:
                    self.logger.warning(f"播放列表为空，不标记为结束状态, 任务ID={task_id}")
                    return {"status": "warning", "message": "播放列表为空，未标记为完成"}
            except Exception as e:
                self.logger.error(f"完成播放列表失败: {e}, 任务ID={task_id}")
                return {"status": "error", "message": f"完成播放列表失败: {str(e)}"}

    async def get_has_segments(self, task_id: str) -> Dict:
        """
        获取任务是否有分段
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 包含状态和has_segments值的字典
        """
        if task_id not in self.task_managers:
            return {"status": "error", "message": "任务HLS管理器不存在", "has_segments": False}
            
        manager = self.task_managers[task_id]
        return {"status": "success", "has_segments": manager["has_segments"]}
    
    async def clean_old_tasks(self, max_age_hours: int = 24) -> Dict:
        """
        清理旧任务资源
        
        Args:
            max_age_hours: 最大保留小时数，默认24小时
            
        Returns:
            Dict: 包含清理信息的字典
        """
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        tasks_to_clean = []
        
        # 标识需要清理的任务
        for task_id, manager in self.task_managers.items():
            if now - manager["created_at"] > max_age_seconds:
                tasks_to_clean.append(task_id)
        
        # 执行清理
        cleaned_count = 0
        for task_id in tasks_to_clean:
            if task_id in self.locks:
                async with self.locks[task_id]:
                    if task_id in self.task_managers:
                        del self.task_managers[task_id]
                        cleaned_count += 1
                del self.locks[task_id]
        
        self.logger.info(f"已清理 {cleaned_count} 个过期任务的HLS资源")
        return {"status": "success", "cleaned_count": cleaned_count} 