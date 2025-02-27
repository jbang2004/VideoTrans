import logging
from typing import Dict, Any
from pathlib import Path
import asyncio
import json
import aioredis
from config import Config
from utils.task_storage import TaskPaths
from utils.task_state import TaskState
from utils.redis_utils import save_task_state, push_to_queue
from services.hls import HLSClient
from utils.ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class ViTranslator:
    def __init__(self, config: Config):
        self.config = config
        self.hls_client = None  # 异步初始化
        self.logger = logger

    async def trans_video(
        self,
        video_path: str,
        task_id: str,
        task_paths: TaskPaths,
        target_language: str = "zh",
        generate_subtitle: bool = False,
    ) -> Dict[str, Any]:
        """翻译视频并返回结果"""
        try:
            # 异步初始化 HLS 客户端
            self.hls_client = await HLSClient.create(self.config)

            # 初始化任务状态并存储
            task_state = TaskState(
                task_id=task_id,
                task_paths=task_paths,
                target_language=target_language,
                generate_subtitle=generate_subtitle
            )
            await save_task_state(task_state)

            # 初始化 HLS
            if not await self.hls_client.init_task(task_id):
                raise RuntimeError("HLS任务初始化失败")

            # 推送初始任务到队列
            await push_to_queue('segment_init_queue', {
                'task_id': task_id,
                'video_path': video_path
            })

            # 等待任务完成
            final_video_path = await self._wait_task_completion(task_id)
            if final_video_path and final_video_path.exists():
                return {
                    "status": "success",
                    "message": "视频翻译完成",
                    "final_video_path": str(final_video_path)
                }
            return {"status": "error", "message": "任务未完成"}
        except Exception as e:
            self.logger.exception(f"处理失败: {e}")
            if self.hls_client:
                await self.hls_client.cleanup_task(task_id)
            return {"status": "error", "message": str(e)}
        finally:
            if self.hls_client:
                await self.hls_client.close()

    async def _wait_task_completion(self, task_id: str):
        """等待任务完成并返回最终视频路径"""
        redis = await aioredis.create_redis_pool('redis://localhost')
        try:
            # 等待完成信号
            completion_key = f"task_completion:{task_id}"
            result = await redis.blpop(completion_key, timeout=3600)  # 1小时超时
            
            if result:
                # 解析结果
                result_data = json.loads(result[1].decode('utf-8'))
                if result_data.get("status") == "success":
                    return Path(result_data.get("final_video_path", ""))
            return None
        finally:
            redis.close()
            await redis.wait_closed()

if __name__ == "__main__":
    # 示例用法
    async def main():
        config = Config()
        translator = ViTranslator(config)
        # 示例调用
        result = await translator.trans_video(
            video_path="/path/to/video.mp4",
            task_id="test_task",
            task_paths=TaskPaths(
                task_dir=Path("/tmp/tasks/test_task"),
                processing_dir=Path("/tmp/tasks/test_task/processing"),
                output_dir=Path("/tmp/tasks/test_task/output")
            )
        )
        print(result)

    # asyncio.run(main())