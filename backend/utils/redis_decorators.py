import logging
import functools
import json
import asyncio
import time
import aioredis
from pathlib import Path
from typing import Callable, Any, Optional, AsyncGenerator, TypeVar, Union, Literal
from utils.redis_utils import load_task_state, save_task_state, get_redis_connection

logger = logging.getLogger(__name__)
T = TypeVar('T')
WorkerResult = Union[T, AsyncGenerator[T, None]]
WorkerMode = Literal['base', 'stream']

def redis_worker_decorator(
    input_queue: str,
    next_queue: Optional[str] = None,
    worker_name: Optional[str] = None,
    mode: WorkerMode = 'base'
) -> Callable:
    """Redis 队列 Worker 装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            worker_display_name = worker_name or func.__name__
            wlogger = getattr(self, 'logger', logger)

            redis = await get_redis_connection()
            wlogger.info(f"[{worker_display_name}] 启动. 输入队列: {input_queue}, 下游队列: {next_queue if next_queue else '无'}")

            processed_count = 0
            try:
                while True:
                    try:
                        # 从 Redis 队列获取任务
                        item_json = await redis.blpop(input_queue, timeout=0)
                        if not item_json:
                            continue
                            
                        item_data = json.loads(item_json[1].decode('utf-8'))
                        if item_data.get('stop_signal'):
                            if next_queue:
                                await redis.rpush(next_queue, json.dumps({'stop_signal': True}))
                            wlogger.info(f"[{worker_display_name}] 收到停止信号。已处理 {processed_count} 个item。")
                            break

                        task_id = item_data.get('task_id')
                        if not task_id:
                            wlogger.warning(f"[{worker_display_name}] 收到无效任务，缺少 task_id")
                            continue

                        # 获取任务路径
                        task_dir = Path(item_data.get('task_dir', f"/tmp/tasks/{task_id}"))
                        
                        # 加载任务状态
                        try:
                            task_state = await load_task_state(task_id, task_dir)
                            if not task_state:
                                wlogger.warning(f"[{worker_display_name}] 无法加载任务状态，跳过任务 {task_id}")
                                continue
                        except Exception as e:
                            wlogger.error(f"[{worker_display_name}] 加载任务状态失败: {e}, 跳过任务 {task_id}")
                            continue

                        wlogger.debug(f"[{worker_display_name}] 从 {input_queue} 取出一个item. TaskID={task_id}")

                        start_time = time.time()
                        if mode == 'stream':
                            async for result in func(self, item_data, task_state, *args, **kwargs):
                                if result is not None and next_queue:
                                    await redis.rpush(next_queue, json.dumps({
                                        'task_id': task_id,
                                        'task_dir': str(task_dir),
                                        'data': result
                                    }))
                        else:
                            result = await func(self, item_data, task_state, *args, **kwargs)
                            if result is not None and next_queue:
                                await redis.rpush(next_queue, json.dumps({
                                    'task_id': task_id,
                                    'task_dir': str(task_dir),
                                    'data': result
                                }))

                        # 保存更新后的任务状态
                        await save_task_state(task_state)

                        processed_count += 1
                        elapsed = time.time() - start_time
                        wlogger.debug(f"[{worker_display_name}] item处理完成，耗时 {elapsed:.2f}s. "
                                    f"TaskID={task_id}, 已处理计数: {processed_count}")

                    except asyncio.CancelledError:
                        wlogger.warning(f"[{worker_display_name}] 被取消. 已处理 {processed_count} 个item")
                        if next_queue:
                            await redis.rpush(next_queue, json.dumps({'stop_signal': True}))
                        break
                    except Exception as e:
                        wlogger.error(f"[{worker_display_name}] 发生异常: {e}. 已处理 {processed_count} 个item", exc_info=True)
                        if next_queue:
                            await redis.rpush(next_queue, json.dumps({'stop_signal': True}))
                        break
            finally:
                wlogger.info(f"[{worker_display_name}] 结束. 共处理 {processed_count} 个item.")
                redis.close()
                await redis.wait_closed()

        return wrapper
    return decorator
