import json
import logging
import aioredis
from pathlib import Path
import pickle
from utils.task_state import TaskState

logger = logging.getLogger(__name__)

# Redis 连接配置
REDIS_URL = 'redis://localhost'
REDIS_TASK_STATE_PREFIX = 'task_state:'
REDIS_TASK_STATE_EXPIRY = 86400 * 7  # 7天过期时间

async def get_redis_connection():
    """获取 Redis 连接"""
    return await aioredis.create_redis_pool(REDIS_URL)

async def save_task_state(task_state: TaskState):
    """保存任务状态到 Redis 和文件"""
    # 保存到文件
    task_dir = task_state.task_paths.task_dir
    task_dir.mkdir(parents=True, exist_ok=True)
    state_file = task_dir / f"{task_state.task_id}_state.pkl"
    
    with open(state_file, 'wb') as f:
        pickle.dump(task_state, f)
    
    # 保存到 Redis
    redis = await get_redis_connection()
    try:
        key = f"{REDIS_TASK_STATE_PREFIX}{task_state.task_id}"
        task_dict = task_state.to_dict()
        await redis.set(key, json.dumps(task_dict), expire=REDIS_TASK_STATE_EXPIRY)
        logger.debug(f"任务状态已保存到 Redis: {task_state.task_id}")
    except Exception as e:
        logger.error(f"保存任务状态到 Redis 失败: {e}")
    finally:
        redis.close()
        await redis.wait_closed()
    
    return state_file

async def load_task_state(task_id: str, task_dir: Path = None) -> TaskState:
    """
    从 Redis 或文件加载任务状态
    优先从 Redis 加载，如果失败则从文件加载
    """
    # 尝试从 Redis 加载
    redis = await get_redis_connection()
    try:
        key = f"{REDIS_TASK_STATE_PREFIX}{task_id}"
        data = await redis.get(key, encoding='utf-8')
        if data:
            task_dict = json.loads(data)
            task_state = TaskState.from_dict(task_dict)
            logger.debug(f"从 Redis 加载任务状态: {task_id}")
            return task_state
    except Exception as e:
        logger.error(f"从 Redis 加载任务状态失败: {e}")
    finally:
        redis.close()
        await redis.wait_closed()
    
    # 如果 Redis 加载失败且提供了 task_dir，则从文件加载
    if task_dir:
        try:
            state_file = task_dir / f"{task_id}_state.pkl"
            with open(state_file, 'rb') as f:
                task_state = pickle.load(f)
            logger.debug(f"从文件加载任务状态: {task_id}")
            return task_state
        except Exception as e:
            logger.error(f"从文件加载任务状态失败: {e}")
    
    raise ValueError(f"无法加载任务状态: {task_id}")

async def update_task_state(task_state: TaskState):
    """更新 Redis 中的任务状态"""
    await save_task_state(task_state)

async def delete_task_state(task_id: str):
    """删除 Redis 中的任务状态"""
    redis = await get_redis_connection()
    try:
        key = f"{REDIS_TASK_STATE_PREFIX}{task_id}"
        await redis.delete(key)
        logger.debug(f"已删除 Redis 中的任务状态: {task_id}")
    except Exception as e:
        logger.error(f"删除 Redis 中的任务状态失败: {e}")
    finally:
        redis.close()
        await redis.wait_closed()

async def push_to_queue(queue_name: str, item: dict):
    """将任务推送到指定的 Redis 队列"""
    redis = await get_redis_connection()
    try:
        await redis.rpush(queue_name, json.dumps(item))
        logger.debug(f"已推送任务到队列: {queue_name}")
    except Exception as e:
        logger.error(f"推送任务到队列失败: {e}")
    finally:
        redis.close()
        await redis.wait_closed()

async def get_queue_length(queue_name: str) -> int:
    """获取队列长度"""
    redis = await get_redis_connection()
    try:
        length = await redis.llen(queue_name)
        return length
    except Exception as e:
        logger.error(f"获取队列长度失败: {e}")
        return 0
    finally:
        redis.close()
        await redis.wait_closed()
