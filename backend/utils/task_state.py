from dataclasses import dataclass, field
from typing import Any, Dict
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

    # 已处理到的句子计数
    sentence_counter: int = 0

    # 时间戳记录
    current_time: float = 0

    # 第几个 HLS 批次 (混音后输出)
    batch_counter: int = 0

    # 每个分段对应的媒体文件信息
    segment_media_files: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # 各个异步队列 (翻译->模型输入->tts_token->时长对齐->音频生成->混音)
    translation_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    modelin_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    tts_token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    duration_align_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    audio_gen_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    mixing_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # 新增：记录 _mixing_worker 产出的每个 segment_xxx.mp4
    merged_segments: list = field(default_factory=list)