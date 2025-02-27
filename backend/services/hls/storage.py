from abc import ABC, abstractmethod
from pathlib import Path
import logging
import shutil
import aiofiles
from typing import Optional

logger = logging.getLogger(__name__)

class StorageService(ABC):
    """存储服务抽象基类"""
    
    @abstractmethod
    async def upload_segment(self, task_id: str, segment_path: Path, segment_index: int) -> str:
        """
        上传分片并返回访问路径
        Args:
            task_id: 任务ID
            segment_path: 分片文件路径
            segment_index: 分片索引
        Returns:
            str: 分片访问路径
        """
        pass
    
    @abstractmethod
    async def update_playlist(self, task_id: str, playlist_content: str) -> str:
        """
        更新播放列表并返回访问路径
        Args:
            task_id: 任务ID
            playlist_content: m3u8内容
        Returns:
            str: 播放列表访问路径
        """
        pass
        
    @abstractmethod
    async def cleanup_task(self, task_id: str) -> None:
        """清理任务相关的存储资源"""
        pass

class LocalStorageService(StorageService):
    """本地文件系统存储实现"""
    
    def __init__(self, config):
        self.config = config
        self.public_dir = config.PUBLIC_DIR
        
    async def upload_segment(self, task_id: str, segment_path: Path, segment_index: int) -> str:
        """本地存储实现 - 复制文件到公共目录"""
        target_dir = self.public_dir / "segments" / task_id
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / f"segment_{segment_index}.ts"
        shutil.copy2(str(segment_path), str(target_path))
        
        return f"/segments/{task_id}/segment_{segment_index}.ts"
        
    async def update_playlist(self, task_id: str, playlist_content: str) -> str:
        """本地存储实现 - 写入m3u8文件"""
        playlist_path = self.public_dir / "playlists" / f"{task_id}.m3u8"
        async with aiofiles.open(playlist_path, 'w') as f:
            await f.write(playlist_content)
        return f"/playlists/{task_id}.m3u8"
        
    async def cleanup_task(self, task_id: str) -> None:
        """清理任务文件"""
        # 清理分片
        segment_dir = self.public_dir / "segments" / task_id
        if segment_dir.exists():
            shutil.rmtree(str(segment_dir))
            
        # 清理播放列表    
        playlist_path = self.public_dir / "playlists" / f"{task_id}.m3u8"
        if playlist_path.exists():
            playlist_path.unlink() 