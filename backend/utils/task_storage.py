import shutil
import logging
import os
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)

class TaskPaths:
    def __init__(self, config: Config, task_id: str):
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

    def create_directories(self):
        dirs = [
            self.task_dir,
            self.input_dir,
            self.processing_dir,
            self.output_dir,
            self.segments_dir,
            self.media_dir,
            self.processing_segments_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            logger.debug(f"[TaskPaths] 创建目录: {d}")

    async def cleanup(self, keep_output: bool = False):
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
                    logger.debug(f"[TaskPaths] 已删除: {self.task_dir}")

                if self.segments_dir.exists():
                    shutil.rmtree(str(self.segments_dir))
                    logger.debug(f"[TaskPaths] 已删除: {self.segments_dir}")
        except Exception as e:
            logger.error(f"[TaskPaths] 清理任务目录失败: {e}", exc_info=True)
            raise
