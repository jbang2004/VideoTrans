#!/usr/bin/env python3
"""
测试向 Redis 队列推送任务
"""
import asyncio
import logging
import sys
import uuid
import json
from pathlib import Path
from utils.redis_utils import push_to_queue, get_queue_length, save_task_state
from utils.task_state import TaskState
from utils.task_storage import TaskPaths

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def push_test_task():
    """推送测试任务到队列"""
    # 创建测试任务
    task_id = f"test_{uuid.uuid4().hex[:8]}"
    task_dir = Path(f"/tmp/tasks/{task_id}")
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建任务路径
    task_paths = TaskPaths(
        task_dir=str(task_dir),
        processing_dir=str(task_dir / "processing"),
        audio_dir=str(task_dir / "audio"),
        subtitle_dir=str(task_dir / "subtitle")
    )
    
    # 创建任务状态
    task_state = TaskState(
        task_id=task_id,
        task_paths=task_paths,
        total_segments=0,
        errors=[]
    )
    
    # 保存任务状态到 Redis
    await save_task_state(task_state)
    
    # 推送到分段初始化队列
    queue_name = "segment_init_queue"
    task_data = {
        "task_id": task_id,
        "task_dir": str(task_dir),
        "video_path": "test_video.mp4"
    }
    
    logger.info(f"推送任务到队列 {queue_name}: {task_data}")
    await push_to_queue(queue_name, task_data)
    
    # 检查队列长度
    length = await get_queue_length(queue_name)
    logger.info(f"队列 {queue_name} 当前长度: {length}")
    
    return task_id, task_dir

async def main():
    """主函数"""
    try:
        task_id, task_dir = await push_test_task()
        logger.info(f"测试任务已推送，TaskID: {task_id}, TaskDir: {task_dir}")
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
