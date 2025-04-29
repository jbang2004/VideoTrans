# ---------------------------------
# backend/utils/task_state.py (完整可复制版本)
# ---------------------------------
from dataclasses import dataclass, field
from typing import Any, Dict, List
import asyncio
from utils.task_storage import TaskPaths

@dataclass
class TaskState:
    """
    每个任务的独立状态：包括队列、处理进度等
    """
    task_id: str
    video_path: str
    task_paths: TaskPaths
    target_language: str = "zh"

    # 时间戳记录
    current_time: float = 0

    # 第几个 HLS 批次 (混音后输出)
    batch_counter: int = 0

    # HLS就绪状态
    hls_ready: bool = False

    # 媒体文件信息（音频、视频等）
    media_files: Dict[str, Any] = field(default_factory=dict)

    # 记录 mixing_worker 产出的每个 segment_xxx.mp4
    merged_segments: List[str] = field(default_factory=list)

    # =========== (新增) ===========
    # 用户是否选择烧制字幕
    generate_subtitle: bool = False
