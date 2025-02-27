#!/usr/bin/env python3
"""
简单的 Redis 队列系统测试脚本
"""
import asyncio
import logging
import sys
import uuid
import json
from pathlib import Path
from utils.redis_utils import (
    get_redis_connection, push_to_queue, get_queue_length
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

async def test_producer(queue_name: str, num_messages: int = 5):
    """测试消息生产者"""
    logger.info(f"开始向队列 {queue_name} 推送 {num_messages} 条消息")
    
    for i in range(num_messages):
        message = {
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "content": f"测试消息 #{i+1}",
            "timestamp": f"{i+1}"
        }
        
        await push_to_queue(queue_name, message)
        logger.info(f"已推送消息 {i+1}/{num_messages}: {message}")
        
        # 暂停一下，便于观察
        await asyncio.sleep(0.5)
    
    length = await get_queue_length(queue_name)
    logger.info(f"推送完成，当前队列长度: {length}")

async def test_consumer(queue_name: str, num_messages: int = 5):
    """测试消息消费者"""
    logger.info(f"开始从队列 {queue_name} 消费消息")
    
    redis = await get_redis_connection()
    try:
        for i in range(num_messages):
            logger.info(f"等待消息 {i+1}/{num_messages}...")
            
            # 设置超时，避免无限等待
            item_json = await redis.blpop(queue_name, timeout=5)
            
            if item_json:
                item_data = json.loads(item_json[1].decode('utf-8'))
                logger.info(f"收到消息: {item_data}")
            else:
                logger.warning(f"等待超时，没有收到消息 {i+1}")
                break
        
        length = await get_queue_length(queue_name)
        logger.info(f"消费完成，当前队列长度: {length}")
    finally:
        redis.close()
        await redis.wait_closed()

async def main():
    """主函数"""
    try:
        # 测试队列名
        test_queue = "test_simple_queue"
        
        # 连接 Redis
        redis = await get_redis_connection()
        try:
            # 清空测试队列
            await redis.delete(test_queue)
            logger.info(f"已清空队列: {test_queue}")
        finally:
            redis.close()
            await redis.wait_closed()
        
        # 创建生产者和消费者任务
        producer = asyncio.create_task(test_producer(test_queue, 5))
        consumer = asyncio.create_task(test_consumer(test_queue, 5))
        
        # 等待任务完成
        await asyncio.gather(producer, consumer)
        
        logger.info("测试完成")
    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
