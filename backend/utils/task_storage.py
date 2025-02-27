import shutil
import logging
import os
from pathlib import Path
from config import Config
from typing import Optional, Union

logger = logging.getLogger(__name__)

class TaskPaths:
    def __init__(
        self, 
        task_dir: Union[str, Path],
        processing_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        segments_dir: Optional[Union[str, Path]] = None,
        audio_dir: Optional[Union[str, Path]] = None,
        subtitle_dir: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None,
        task_id: Optional[str] = None
    ):
        """
        初始化任务路径
        可以通过两种方式初始化：
        1. 提供 config 和 task_id，自动生成所有路径
        2. 直接提供各个路径，用于从 Redis 恢复
        """
        # 将所有路径转换为 Path 对象
        self.task_dir = Path(task_dir)
        
        # 如果提供了 config 和 task_id，使用配置生成路径
        if config and task_id:
            self.config = config
            self.task_id = task_id

            self.task_dir = config.TASKS_DIR / task_id
            self.input_dir = self.task_dir / "input"
            self.processing_dir = self.task_dir / "processing"
            self.output_dir = self.task_dir / "output"

            self.segments_dir = config.PUBLIC_DIR / "segments" / task_id
            self.playlist_path = config.PUBLIC_DIR / "playlists" / f"playlist_{task_id}.m3u8"

            self.media_dir = self.processing_dir / "media"
            self.processing_segments_dir = self.processing_dir / "segments"
            self.audio_dir = self.processing_dir / "audio"
            self.subtitle_dir = self.processing_dir / "subtitle"
        else:
            # 直接使用提供的路径
            self.processing_dir = Path(processing_dir) if processing_dir else self.task_dir / "processing"
            self.output_dir = Path(output_dir) if output_dir else self.task_dir / "output"
            self.segments_dir = Path(segments_dir) if segments_dir else self.task_dir / "segments"
            self.audio_dir = Path(audio_dir) if audio_dir else self.processing_dir / "audio"
            self.subtitle_dir = Path(subtitle_dir) if subtitle_dir else self.processing_dir / "subtitle"
            
            # 其他可能需要的目录
            self.input_dir = self.task_dir / "input"
            self.media_dir = self.processing_dir / "media"
            self.processing_segments_dir = self.processing_dir / "segments"
            
            # 尝试从 task_dir 提取 task_id
            if not task_id:
                self.task_id = self.task_dir.name
            else:
                self.task_id = task_id

    def create_directories(self):
        """创建所有必要的目录"""
        dirs = [
            self.task_dir,
            self.input_dir,
            self.processing_dir,
            self.output_dir,
            self.segments_dir,
            self.media_dir,
            self.processing_segments_dir,
            self.audio_dir,
            self.subtitle_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug(f"[TaskPaths] 创建目录: {d}")

    async def cleanup(self, keep_output: bool = False):
        """清理任务目录"""
        try:
            if keep_output:
                logger.info(f"[TaskPaths] 保留输出目录, 即将清理输入/processing/segments")
                dirs_to_clean = [self.input_dir, self.processing_dir, self.segments_dir]
                for d in dirs_to_clean:
                    if d.exists():
                        shutil.rmtree(d)
                        logger.debug(f"[TaskPaths] 已清理: {d}")
            else:
                logger.info(f"[TaskPaths] 全量清理任务目录: {self.task_dir}")
                if self.task_dir.exists():
                    shutil.rmtree(str(self.task_dir))
                if hasattr(self, 'segments_dir') and self.segments_dir.exists():
                    shutil.rmtree(str(self.segments_dir))
                if hasattr(self, 'playlist_path') and self.playlist_path.exists():
                    os.remove(str(self.playlist_path))
        except Exception as e:
            logger.error(f"[TaskPaths] 清理失败: {e}")
