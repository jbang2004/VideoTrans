from dataclasses import dataclass, field
from typing import Optional, Any
import asyncio
from utils.task_storage import TaskPaths

@dataclass
class TaskState:
    """
    每个任务的独立状态：包括队列、处理进度、分段信息等
    """
    task_id: str
    video_path: str
    task_paths: TaskPaths
    hls_manager: Any = None
    target_language: str = "zh"

    sentence_counter: int = 0
    current_time: float = 0
    batch_counter: int = 0
    segment_media_files: dict = field(default_factory=dict)

    # 每个任务的队列
    translation_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    modelin_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    tts_token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    duration_align_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    audio_gen_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    mixing_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # 第一段处理完成的同步
    mixing_complete: asyncio.Queue = field(default_factory=asyncio.Queue)
    first_segment_batch_count: int = 0
    first_segment_processed_count: int = 0