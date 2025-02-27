# ---------------------------------
# backend/utils/task_state.py (完整可复制版本)
# ---------------------------------
from dataclasses import dataclass, field
from typing import Any, Dict
from utils.task_storage import TaskPaths

@dataclass
class TaskState:
    """
    每个任务的独立状态：包括处理进度、分段信息等
    """
    task_id: str
    task_paths: TaskPaths
    target_language: str = "zh"
    generate_subtitle: bool = False

    # 已处理到的句子计数
    sentence_counter: int = 0

    # 时间戳记录
    current_time: float = 0

    # 第几个 HLS 批次 (混音后输出)
    batch_counter: int = 0

    # 每个分段对应的媒体文件信息
    segment_media_files: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # 记录 mixing_worker 产出的每个 segment_xxx.mp4
    merged_segments: list = field(default_factory=list)

    # 记录总分段数
    total_segments: int = 0

    # 错误记录
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """将 TaskState 转换为字典，用于 Redis 存储"""
        return {
            "task_id": self.task_id,
            "task_paths": {
                "task_dir": str(self.task_paths.task_dir),
                "processing_dir": str(self.task_paths.processing_dir),
                "output_dir": str(self.task_paths.output_dir),
                "segments_dir": str(self.task_paths.segments_dir),
                "audio_dir": str(self.task_paths.audio_dir),
                "subtitle_dir": str(self.task_paths.subtitle_dir),
            },
            "target_language": self.target_language,
            "generate_subtitle": self.generate_subtitle,
            "sentence_counter": self.sentence_counter,
            "current_time": self.current_time,
            "batch_counter": self.batch_counter,
            "segment_media_files": self.segment_media_files,
            "merged_segments": self.merged_segments,
            "total_segments": self.total_segments,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TaskState':
        """从字典创建 TaskState 实例，用于从 Redis 恢复"""
        task_paths = TaskPaths(
            task_dir=data["task_paths"]["task_dir"],
            processing_dir=data["task_paths"]["processing_dir"],
            output_dir=data["task_paths"]["output_dir"],
            segments_dir=data["task_paths"]["segments_dir"],
            audio_dir=data["task_paths"]["audio_dir"],
            subtitle_dir=data["task_paths"]["subtitle_dir"],
        )
        
        task_state = cls(
            task_id=data["task_id"],
            task_paths=task_paths,
            target_language=data["target_language"],
            generate_subtitle=data["generate_subtitle"],
        )
        
        task_state.sentence_counter = data["sentence_counter"]
        task_state.current_time = data["current_time"]
        task_state.batch_counter = data["batch_counter"]
        task_state.segment_media_files = data["segment_media_files"]
        task_state.merged_segments = data["merged_segments"]
        task_state.total_segments = data["total_segments"]
        task_state.errors = data["errors"]
        
        return task_state
        
    def all_segments_processed(self) -> bool:
        """检查是否所有分段都已处理完成"""
        if self.total_segments == 0:
            return False
        return len(self.merged_segments) >= self.total_segments
