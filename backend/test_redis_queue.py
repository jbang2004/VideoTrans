#!/usr/bin/env python3
"""
Redis 队列系统测试脚本
"""
import asyncio
import logging
import sys
import uuid
import json
from pathlib import Path
from config import Config
from utils.task_storage import TaskPaths
from utils.task_state import TaskState
from utils.redis_utils import (
    save_task_state, load_task_state, push_to_queue, 
    get_redis_connection, get_queue_length
)

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def test_task_paths():
    """测试 TaskPaths 类的两种初始化方式"""
    task_id = f"test_{uuid.uuid4().hex[:8]}"
    task_dir = Path(f"/tmp/tasks/{task_id}")
    
    # 方式1：直接提供路径
    task_paths1 = TaskPaths(
        task_dir=task_dir,
        processing_dir=task_dir / "processing",
        output_dir=task_dir / "output",
        segments_dir=task_dir / "segments"
    )
    
    # 方式2：使用 config 和 task_id
    config = Config()
    task_paths2 = TaskPaths(
        task_dir=task_dir,
        config=config,
        task_id=task_id
    )
    
    # 验证
    assert task_paths1.task_dir == task_dir
    assert task_paths1.task_id == task_id
    
    logger.info(f"TaskPaths 测试通过: {task_id}")
    return task_paths1

async def test_task_state():
    """测试任务状态的保存和加载"""
    task_id = f"test_{uuid.uuid4().hex[:8]}"
    task_dir = Path(f"/tmp/tasks/{task_id}")
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建任务路径
    task_paths = TaskPaths(
        task_dir=task_dir,
        processing_dir=task_dir / "processing",
        output_dir=task_dir / "output",
        segments_dir=task_dir / "segments"
    )
    
    # 创建任务状态
    task_state = TaskState(
        task_id=task_id,
        task_paths=task_paths,
        target_language="zh",
        generate_subtitle=True
    )
    
    # 保存任务状态
    logger.info(f"保存任务状态: {task_id}")
    await save_task_state(task_state)
    
    # 加载任务状态
    logger.info(f"加载任务状态: {task_id}")
    loaded_task_state = await load_task_state(task_id, task_dir)
    
    # 验证
    assert loaded_task_state.task_id == task_id
    assert loaded_task_state.target_language == "zh"
    assert loaded_task_state.generate_subtitle == True
    
    # 测试 to_dict 和 from_dict
    task_dict = task_state.to_dict()
    task_state_from_dict = TaskState.from_dict(task_dict)
    
    assert task_state_from_dict.task_id == task_id
    assert task_state_from_dict.target_language == "zh"
    assert task_state_from_dict.generate_subtitle == True
    
    logger.info(f"任务状态测试通过: {task_id}")
    return task_state

async def test_redis_queue(task_state):
    """测试 Redis 队列的推送和获取"""
    # 测试队列名
    test_queue = "test_queue"
    
    # 连接 Redis
    redis = await get_redis_connection()
    
    try:
        # 清空测试队列
        await redis.delete(test_queue)
        
        # 推送测试数据
        test_data = {
            "task_id": task_state.task_id,
            "task_dir": str(task_state.task_paths.task_dir),
            "test_value": "Hello Redis Queue"
        }
        
        logger.info(f"推送数据到队列: {test_queue}")
        await push_to_queue(test_queue, test_data)
        
        # 获取队列长度
        length = await get_queue_length(test_queue)
        assert length == 1
        logger.info(f"队列长度: {length}")
        
        # 从队列获取数据
        logger.info(f"从队列获取数据: {test_queue}")
        item_json = await redis.blpop(test_queue, timeout=1)
        
        if item_json:
            item_data = json.loads(item_json[1].decode('utf-8'))
            assert item_data["task_id"] == task_state.task_id
            assert item_data["test_value"] == "Hello Redis Queue"
            logger.info(f"获取的数据: {item_data}")
        else:
            raise Exception("从队列获取数据失败")
        
        logger.info("Redis 队列测试通过")
    finally:
        # 清理
        await redis.delete(test_queue)
        redis.close()
        await redis.wait_closed()

async def main():
    """主函数"""
    try:
        logger.info("开始测试 Redis 队列系统")
        
        # 测试 TaskPaths
        await test_task_paths()
        
        # 测试任务状态
        task_state = await test_task_state()
        
        # 测试 Redis 队列
        await test_redis_queue(task_state)
        
        logger.info("所有测试通过")
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
