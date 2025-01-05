from pathlib import Path
import shutil
import logging
import os.path
from typing import Dict, Optional
from config import Config

logger = logging.getLogger(__name__)

class TaskPaths:
    """任务路径信息类"""
    def __init__(self, config: Config, task_id: str):
        # 任务基本目录
        self.task_dir = config.TASKS_DIR / task_id
        
        # 工作目录
        self.input_dir = self.task_dir / "input"              # 输入文件
        self.processing_dir = self.task_dir / "processing"    # 处理过程文件
        self.output_dir = self.task_dir / "output"           # 输出文件
        
        # 公共访问路径
        self.playlist_path = config.PUBLIC_DIR / "playlists" / f"playlist_{task_id}.m3u8"
        self.segments_dir = config.PUBLIC_DIR / "segments" / task_id
        
        # 具体工作目录
        self.media_dir = self.processing_dir / "media"        # 媒体处理目录
        self.processing_segments_dir = self.processing_dir / "segments"  # 重命名为 processing_segments_dir
        
    def create_directories(self):
        """创建所有必要的目录"""
        dirs = [
            self.task_dir,
            self.input_dir,
            self.processing_dir,
            self.output_dir,
            self.media_dir,
            self.processing_segments_dir
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
    async def cleanup(self, keep_output: bool = False):
        """清理任务目录"""
        try:
            if keep_output:
                # 只保留输出文件
                dirs_to_clean = [
                    self.input_dir,
                    self.processing_dir
                ]
                for dir_path in dirs_to_clean:
                    if os.path.exists(str(dir_path)):
                        shutil.rmtree(str(dir_path))
            else:
                # 清理所有文件
                if os.path.exists(str(self.task_dir)):
                    shutil.rmtree(str(self.task_dir))
                if os.path.exists(str(self.processing_segments_dir)):
                    shutil.rmtree(str(self.processing_segments_dir))
        except Exception as e:
            logger.error(f"清理任务目录失败: {e}")
            raise 