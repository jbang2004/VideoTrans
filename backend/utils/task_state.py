# ---------------------------------
# backend/utils/task_state.py (完整可复制版本)
# ---------------------------------
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
    generate_subtitle: bool = False

    # 已处理到的句子计数
    sentence_counter: int = 0

    # 时间戳记录
    current_time: float = 0

    # 第几个 HLS 批次 (混音后输出)
    batch_counter: int = 0

    # 每个分段对应的媒体文件信息
    segment_media_files: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # 各个异步队列 (按处理流程顺序排列)
    segment_init_queue: asyncio.Queue = field(default_factory=asyncio.Queue)  # 分段初始化队列
    segment_queue: asyncio.Queue = field(default_factory=asyncio.Queue)       # 分段提取队列
    asr_queue: asyncio.Queue = field(default_factory=asyncio.Queue)          # ASR处理队列
    translation_queue: asyncio.Queue = field(default_factory=asyncio.Queue)   # 翻译队列
    modelin_queue: asyncio.Queue = field(default_factory=asyncio.Queue)      # 模型输入队列
    tts_token_queue: asyncio.Queue = field(default_factory=asyncio.Queue)    # TTS Token队列
    duration_align_queue: asyncio.Queue = field(default_factory=asyncio.Queue) # 时长对齐队列
    audio_gen_queue: asyncio.Queue = field(default_factory=asyncio.Queue)    # 音频生成队列
    mixing_queue: asyncio.Queue = field(default_factory=asyncio.Queue)       # 混音队列

    # 记录 mixing_worker 产出的每个 segment_xxx.mp4
    merged_segments: list = field(default_factory=list)

    # 错误记录
    errors: list = field(default_factory=list)
