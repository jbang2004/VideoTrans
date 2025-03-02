"""
工作者装饰器模块 - 包含所有与工作者相关的装饰器
"""
import functools
import json
import asyncio
import time
import aioredis
from pathlib import Path
from typing import Callable, Any, Optional, AsyncGenerator, TypeVar, Union, Literal, Dict
from utils.redis_utils import load_task_state, save_task_state, get_redis_connection
from utils.serialization import SentenceSerializer
from utils.log_config import get_logger

logger = get_logger(__name__)
T = TypeVar('T')
WorkerResult = Union[T, AsyncGenerator[T, None]]
WorkerMode = Literal['base', 'stream']
SerializationMode = Literal['json', 'msgpack']

def handle_errors(custom_logger = None) -> Callable:
    """错误处理装饰器。可应用于需要统一捕获日志的异步函数。"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # 如果当前对象有 logger 属性则使用，否则用传入的或全局 logger
            actual_logger = custom_logger if custom_logger else (getattr(args[0], 'logger', logger) if args else logger)
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                actual_logger.info(f"{func.__name__} 正常结束，耗时 {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                actual_logger.error(f"{func.__name__} 执行出错，耗时 {elapsed:.2f}s, 错误: {e}", exc_info=True)
                raise
        return wrapper
    return decorator

def worker_decorator(
    input_queue_attr: str,
    next_queue_attr: Optional[str] = None,
    worker_name: Optional[str] = None,
    mode: WorkerMode = 'base'
) -> Callable:
    """通用 Worker 装饰器 (内存队列版本)"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, task_state, *args, **kwargs):
            worker_display_name = worker_name or func.__name__
            wlogger = getattr(self, 'logger', logger)

            input_queue = getattr(task_state, input_queue_attr)
            next_queue = getattr(task_state, next_queue_attr) if next_queue_attr else None

            wlogger.info(f"[{worker_display_name}] 启动 (TaskID={task_state.task_id}). "
                         f"输入队列: {input_queue_attr}, 下游队列: {next_queue_attr if next_queue_attr else '无'}")

            processed_count = 0
            try:
                while True:
                    try:
                        queue_size_before = input_queue.qsize()
                        item = await input_queue.get()
                        if item is None:
                            if next_queue:
                                await next_queue.put(None)
                            wlogger.info(f"[{worker_display_name}] 收到停止信号。已处理 {processed_count} 个item。")
                            break

                        wlogger.debug(f"[{worker_display_name}] 从 {input_queue_attr} 取出一个item. 队列剩余: {queue_size_before}")

                        start_time = time.time()
                        if mode == 'stream':
                            async for result in func(self, item, task_state, *args, **kwargs):
                                if result is not None and next_queue:
                                    await next_queue.put(result)
                        else:
                            result = await func(self, item, task_state, *args, **kwargs)
                            if result is not None and next_queue:
                                await next_queue.put(result)

                        processed_count += 1
                        elapsed = time.time() - start_time
                        wlogger.debug(f"[{worker_display_name}] item处理完成，耗时 {elapsed:.2f}s. "
                                      f"TaskID={task_state.task_id}, 已处理计数: {processed_count}")

                    except asyncio.CancelledError:
                        wlogger.warning(f"[{worker_display_name}] 被取消 (TaskID={task_state.task_id}). "
                                        f"已处理 {processed_count} 个item")
                        if next_queue:
                            await next_queue.put(None)
                        break
                    except Exception as e:
                        wlogger.error(f"[{worker_display_name}] 发生异常: {e} (TaskID={task_state.task_id}). "
                                      f"已处理 {processed_count} 个item", exc_info=True)
                        if next_queue:
                            await next_queue.put(None)
                        break
            finally:
                wlogger.info(f"[{worker_display_name}] 结束 (TaskID={task_state.task_id}). 共处理 {processed_count} 个item.")

        return wrapper
    return decorator

def redis_worker_decorator(
    input_queue: str,
    next_queue: Optional[str] = None,
    worker_name: Optional[str] = None,
    mode: WorkerMode = 'base',
    serialization_mode: SerializationMode = 'msgpack'
) -> Callable:
    """Redis 队列 Worker 装饰器
    
    Args:
        input_queue: 输入队列名称
        next_queue: 下游队列名称（可选）
        worker_name: Worker 显示名称（可选）
        mode: Worker 模式，'base' 或 'stream'
        serialization_mode: 序列化模式，'json' 或 'msgpack'
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            worker_display_name = worker_name or func.__name__
            wlogger = getattr(self, 'logger', logger)

            redis = await get_redis_connection()
            wlogger.info(f"[{worker_display_name}] 启动. 输入队列: {input_queue}, 下游队列: {next_queue if next_queue else '无'}, "
                        f"序列化模式: {serialization_mode}")

            processed_count = 0
            try:
                while True:
                    try:
                        # 从 Redis 队列获取任务
                        item_raw = await redis.blpop(input_queue, timeout=0)
                        if not item_raw:
                            continue
                        
                        # 根据序列化模式反序列化数据
                        if serialization_mode == 'json':
                            item_data = json.loads(item_raw[1].decode('utf-8'))
                        else:  # msgpack
                            # 检查是否为 JSON 格式的控制消息
                            try:
                                item_str = item_raw[1].decode('utf-8')
                                if item_str.startswith('{') and item_str.endswith('}'):
                                    item_data = json.loads(item_str)
                                    if 'stop_signal' in item_data:
                                        # 控制消息使用 JSON 格式
                                        pass
                                    else:
                                        # 尝试从 JSON 中提取 msgpack 数据
                                        if 'data' in item_data and isinstance(item_data['data'], str):
                                            item_data['data'] = SentenceSerializer.deserialize_from_redis(item_data['data'])
                                else:
                                    # 直接使用 msgpack 反序列化
                                    item_data = SentenceSerializer.deserialize(item_raw[1])
                            except UnicodeDecodeError:
                                # 二进制数据，直接使用 msgpack 反序列化
                                item_data = SentenceSerializer.deserialize(item_raw[1])
                            except Exception as e:
                                wlogger.error(f"[{worker_display_name}] 反序列化失败: {e}")
                                continue
                        
                        if item_data.get('stop_signal'):
                            if next_queue:
                                # 停止信号使用 JSON 格式
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

                        wlogger.info(f"[{worker_display_name}] 从 {input_queue} 取出一个item. TaskID={task_id}")
                        
                        # 打印反序列化后的第一个句子对象内容（如果存在）
                        try:
                            data_to_log = item_data.get('data', item_data) if isinstance(item_data, dict) else item_data
                            wlogger.debug(f"[{worker_display_name}] 解析后的输入数据类型: {type(data_to_log)}")
                            
                            if isinstance(data_to_log, list) and len(data_to_log) > 0:
                                first_item = data_to_log[0]
                                wlogger.debug(f"[{worker_display_name}] 第一个元素类型: {type(first_item)}")
                                
                                if hasattr(first_item, '__dict__'):
                                    # 如果是对象，打印其属性
                                    item_attrs = {k: str(v) for k, v in first_item.__dict__.items() 
                                                if k not in ['audio_data', 'waveform'] and not k.startswith('_')}
                                    wlogger.debug(f"[{worker_display_name}] 第一个句子对象属性: {item_attrs}")
                                elif isinstance(first_item, dict):
                                    # 如果是字典，直接打印（排除大型二进制数据）
                                    filtered_dict = {k: str(v) for k, v in first_item.items() 
                                                  if k not in ['audio_data', 'waveform'] and not isinstance(v, bytes)}
                                    wlogger.debug(f"[{worker_display_name}] 第一个句子对象: {filtered_dict}")
                                else:
                                    wlogger.debug(f"[{worker_display_name}] 第一个句子对象类型: {type(first_item)}, 值: {str(first_item)}")
                            elif isinstance(data_to_log, dict):
                                # 直接是单个字典对象
                                filtered_dict = {k: str(v) for k, v in data_to_log.items() 
                                              if k not in ['audio_data', 'waveform'] and not isinstance(v, bytes)}
                                wlogger.debug(f"[{worker_display_name}] 句子对象: {filtered_dict}")
                        except Exception as e:
                            wlogger.debug(f"[{worker_display_name}] 打印句子对象失败: {str(e)}")
                            import traceback
                            wlogger.debug(f"[{worker_display_name}] 错误详情: {traceback.format_exc()}")

                        start_time = time.time()
                        if mode == 'stream':
                            async for result in func(self, item_data, task_state, *args, **kwargs):
                                if result is not None and next_queue:
                                    # 根据序列化模式序列化数据
                                    if serialization_mode == 'json':
                                        await redis.rpush(next_queue, json.dumps({
                                            'task_id': task_id,
                                            'task_dir': str(task_dir),
                                            'data': result
                                        }))
                                    else:  # msgpack
                                        # 将结果序列化为 msgpack 格式
                                        serialized_result = SentenceSerializer.serialize_to_redis(result)
                                        await redis.rpush(next_queue, json.dumps({
                                            'task_id': task_id,
                                            'task_dir': str(task_dir),
                                            'data': serialized_result
                                        }))
                        else:
                            result = await func(self, item_data, task_state, *args, **kwargs)
                            if result is not None and next_queue:
                                # 根据序列化模式序列化数据
                                if serialization_mode == 'json':
                                    await redis.rpush(next_queue, json.dumps({
                                        'task_id': task_id,
                                        'task_dir': str(task_dir),
                                        'data': result
                                    }))
                                else:  # msgpack
                                    # 将结果序列化为 msgpack 格式
                                    serialized_result = SentenceSerializer.serialize_to_redis(result)
                                    await redis.rpush(next_queue, json.dumps({
                                        'task_id': task_id,
                                        'task_dir': str(task_dir),
                                        'data': serialized_result
                                    }))

                        # 保存更新后的任务状态
                        await save_task_state(task_state)

                        processed_count += 1
                        elapsed = time.time() - start_time
                        wlogger.info(f"[{worker_display_name}] item处理完成，耗时 {elapsed:.2f}s. "
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
