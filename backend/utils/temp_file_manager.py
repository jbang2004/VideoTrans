import logging
from pathlib import Path
from typing import Set

logger = logging.getLogger(__name__)

class TempFileManager:
    """临时文件管理器"""
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.temp_files: Set[Path] = set()
    
    def add_file(self, file_path: Path) -> None:
        """添加临时文件到管理器"""
        self.temp_files.add(Path(file_path))
    
    async def cleanup(self) -> None:
        """清理所有临时文件"""
        for file_path in self.temp_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"已删除临时文件: {file_path}")
            except Exception as e:
                logger.warning(f"清理临时文件失败: {file_path}, 错误: {e}")
        self.temp_files.clear()