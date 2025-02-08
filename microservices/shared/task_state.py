from dataclasses import dataclass, field
from typing import Any, Dict
import asyncio
from .task_storage import TaskPaths

@dataclass
class TaskState:
    task_id: str
    video_path: str
    task_paths: TaskPaths
    hls_manager: Any = None
    target_language: str = "zh"
    sentence_counter: int = 0
    current_time: float = 0
    batch_counter: int = 0
    segment_media_files: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    translation_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    modelin_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    tts_token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    duration_align_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    audio_gen_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    mixing_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    merged_segments: list = field(default_factory=list)
    generate_subtitle: bool = False
