import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import aioredis

from config import Config
from services.hls.client import HLSClient
from utils.task_storage import TaskPaths
from utils.task_state import TaskState
from utils.redis_utils import save_task_state, push_to_queue
from utils.ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class ViTranslator:
    """视频翻译器"""
    def __init__(self, config: Config):
        self.config = config
        self.hls_client = None  # 异步初始化
        self.redis = None  # 异步初始化
        self.logger = logger

    async def initialize(self):
        """异步初始化"""
        self.redis = await aioredis.create_redis_pool("redis://localhost")
        return self

    async def trans_video(
        self,
        video_path: str,
        task_id: str,
        task_paths: TaskPaths,
        target_language: str = "zh",
        generate_subtitle: bool = False,
    ):
        """
        翻译视频
        
        Args:
            video_path: 视频路径
            task_id: 任务ID
            task_paths: 任务路径
            target_language: 目标语言
            generate_subtitle: 是否生成字幕
        
        Returns:
            最终视频路径
        """
        try:
            # 初始化 HLS 客户端
            if not self.hls_client:
                self.hls_client = await HLSClient.create(self.config)
            
            # 初始化任务
            try:
                if not await self.hls_client.init_task(task_id):
                    logger.error(f"HLS任务初始化失败")
                    raise RuntimeError("HLS任务初始化失败")
            except Exception as e:
                logger.error(f"HLS任务初始化失败: {e}", exc_info=True)
                raise RuntimeError("HLS任务初始化失败")
            
            # 创建并保存任务状态
            task_state = TaskState(
                task_id=task_id,
                task_paths=task_paths,
                target_language=target_language,
                generate_subtitle=generate_subtitle
            )
            await save_task_state(task_state)
            
            # 推送初始化任务
            segment_init_data = {
                "task_id": task_id,
                "task_dir": str(task_paths.task_dir),
                "video_path": video_path,
            }
            await self.redis.rpush("segment_init_queue", json.dumps(segment_init_data))
            
            # 等待任务完成
            try:
                final_video_path = await self._wait_task_completion(task_id)
                if final_video_path:
                    return final_video_path
                else:
                    logger.error(f"任务 {task_id} 处理失败或超时")
                    raise RuntimeError("视频处理失败或超时")
            except Exception as e:
                logger.error(f"等待任务完成时出错: {e}", exc_info=True)
                raise RuntimeError(f"处理失败: {str(e)}")
        except Exception as e:
            # 清理任务
            if self.hls_client:
                try:
                    await self.hls_client.cleanup_task(task_id)
                except Exception as cleanup_error:
                    logger.error(f"清理任务失败: {cleanup_error}")
            
            # 重新抛出异常
            raise RuntimeError(f"处理失败: {str(e)}")
        finally:
            if self.hls_client:
                await self.hls_client.close()

    async def _wait_task_completion(self, task_id: str):
        """等待任务完成并返回最终视频路径"""
        try:
            # 等待完成信号
            completion_key = f"task_completion:{task_id}"
            
            # 检查是否已经有结果
            existing_result = await self.redis.lrange(completion_key, 0, -1)
            if existing_result:
                # 如果已经有结果，直接返回
                result_data = json.loads(existing_result[0].decode('utf-8'))
                if result_data.get("status") == "success":
                    return Path(result_data.get("final_video_path", ""))
                return None
            
            # 设置更短的超时时间，避免长时间阻塞
            timeout = 300  # 5分钟超时
            try:
                result = await asyncio.wait_for(
                    self.redis.blpop(completion_key),
                    timeout=timeout
                )
                
                if result:
                    # 解析结果
                    result_data = json.loads(result[1].decode('utf-8'))
                    if result_data.get("status") == "success":
                        return Path(result_data.get("final_video_path", ""))
                return None
            except asyncio.TimeoutError:
                logger.warning(f"等待任务 {task_id} 完成超时（{timeout}秒）")
                return None
        except Exception as e:
            logger.error(f"等待任务完成时出错: {e}", exc_info=True)
            return None

if __name__ == "__main__":
    # 示例用法
    async def main():
        config = Config()
        translator = await ViTranslator(config).initialize()
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