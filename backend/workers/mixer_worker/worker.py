import asyncio
import json
from typing import List, Any
from pathlib import Path
from utils.worker_decorators import redis_worker_decorator
from utils.task_state import TaskState
from utils.redis_utils import load_task_state
from utils.log_config import get_logger
from .media_mixer import MediaMixer
from services.hls import HLSClient
import aioredis

logger = get_logger(__name__)

class MixerWorker:
    """
    混音 Worker：调用 MediaMixer 将生成的音频与视频混合，生成最终输出段视频。
    """

    def __init__(self, config):
        """初始化 MixerWorker"""
        self.config = config
        self.logger = logger
        self.hls_service = None  # 异步初始化
        
        # 直接实例化 MediaMixer
        self.mixer = MediaMixer(config=config)

    async def initialize(self):
        """异步初始化 HLS 服务"""
        self.hls_service = await HLSClient.create(self.config)
        return self

    @redis_worker_decorator(
        input_queue='mixing_queue',
        worker_name='混音 Worker',
        serialization_mode='msgpack'
    )
    async def run(self, item, task_state: TaskState):
        sentences_batch = item.get('data', item) if isinstance(item, dict) else item
        if not sentences_batch:
            return
        self.logger.debug(f"[混音 Worker] 收到 {len(sentences_batch)} 句子, TaskID={task_state.task_id}")

        # 确保 HLS 服务已初始化
        if not self.hls_service:
            await self.initialize()

        # 处理输出路径
        output_path = task_state.task_paths.segments_dir / f"segment_{task_state.batch_counter}.mp4"

        # 混音处理
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
                
                # 检查是否所有分段都已处理完成
                if task_state.all_segments_processed():
                    await self._notify_task_completion(task_state)
            else:
                self.logger.error(f"[混音 Worker] 分段 {task_state.batch_counter} 添加到 HLS 流失败, TaskID={task_state.task_id}")

        task_state.batch_counter += 1
        return None
        
    async def _notify_task_completion(self, task_state: TaskState):
        """通知任务完成"""
        redis = await aioredis.create_redis_pool('redis://localhost')
        try:
            # 获取最终视频路径
            final_video_path = await self.hls_service.finalize_task(task_state.task_id)
            
            # 发送完成信号
            completion_key = f"task_completion:{task_state.task_id}"
            completion_data = {
                "status": "success",
                "final_video_path": str(final_video_path) if final_video_path else ""
            }
            await redis.rpush(completion_key, json.dumps(completion_data))
            
            self.logger.info(f"[混音 Worker] 任务 {task_state.task_id} 已完成，发送完成信号")
        except Exception as e:
            self.logger.error(f"[混音 Worker] 发送完成信号失败: {e}")
        finally:
            redis.close()
            await redis.wait_closed()

async def start():
    """启动 Worker"""
    config_module = __import__('config')
    config = config_module.Config()
    worker = await MixerWorker(config).initialize()
    await worker.run()

if __name__ == '__main__':
    asyncio.run(start())
