import logging
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)

class TempFileManager:
    """管理临时文件，方便任务结束时统一清理"""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.temp_files: Set[Path] = set()
    
    def add_file(self, file_path: Path) -> None:
        self.temp_files.add(Path(file_path))
    
    async def cleanup(self) -> None:
        for file_path in self.temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Deleted temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {file_path}: {e}")
        self.temp_files.clear()
