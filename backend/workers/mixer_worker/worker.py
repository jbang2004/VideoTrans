import logging
from typing import List, Any
from pathlib import Path
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from .media_mixer import MediaMixer
from services.hls import HLSClient

logger = logging.getLogger(__name__)

class MixerWorker:
    """
    混音 Worker：调用 MediaMixer 将生成的音频与视频混合，生成最终输出段视频。
    """

    def __init__(self, config, hls_service: HLSClient):
        """初始化 MixerWorker"""
        self.config = config
        self.hls_service = hls_service
        self.logger = logger
        
        # 直接实例化 MediaMixer
        self.mixer = MediaMixer(config=config)

    @worker_decorator(
        input_queue_attr='mixing_queue',
        worker_name='混音 Worker'
    )
    async def run(self, sentences_batch: List[Any], task_state: TaskState):
        if not sentences_batch:
            return
        seg_index = sentences_batch[0].segment_index
        self.logger.debug(f"[混音 Worker] 收到 {len(sentences_batch)} 句, segment={seg_index}, TaskID={task_state.task_id}")

        output_path = task_state.task_paths.segments_dir / f"segment_{task_state.batch_counter}.mp4"

        success = await self.mixer.mixed_media_maker(
            sentences=sentences_batch,
            task_state=task_state,
            output_path=str(output_path),
            generate_subtitle=task_state.generate_subtitle
        )

        if success:
            # 使用 HLS 服务
            added = await self.hls_service.add_segment(
                task_state.task_id,
                output_path,
                task_state.batch_counter
            )
            if added:
                self.logger.info(f"[混音 Worker] 分段 {task_state.batch_counter} 已加入 HLS, TaskID={task_state.task_id}")
                task_state.merged_segments.append(str(output_path))
            else:
                self.logger.error(f"[混音 Worker] 分段 {task_state.batch_counter} 添加到 HLS 流失败, TaskID={task_state.task_id}")

        task_state.batch_counter += 1
        return None

if __name__ == '__main__':
    print("Mixer Worker 模块加载成功")
