import logging
import functools
from typing import Callable, Any, Optional, Coroutine, AsyncGenerator, TypeVar, Union, Literal
import asyncio
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')
WorkerResult = Union[T, AsyncGenerator[T, None]]
WorkerMode = Literal['base', 'stream']

def handle_errors(custom_logger: Optional[logging.Logger] = None) -> Callable:
    """
    错误处理装饰器。可应用于需要统一捕获日志的异步函数。
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # 选用 custom_logger 或从 args[0].logger 获取
            actual_logger = custom_logger
            if not actual_logger and args and hasattr(args[0], 'logger'):
                actual_logger = args[0].logger

            if not actual_logger:
                actual_logger = logger

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                actual_logger.debug(f"{func.__name__} 正常结束，耗时 {elapsed:.2f}s")
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
    """
    通用 Worker 装饰器，用于将一个异步函数变成“从指定队列读取数据 -> 处理 -> 放入下游队列”的流式流程。
    新增更多日志输出，如队列长度、处理耗时等，以便在后台看到详细信息。
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, task_state, *args, **kwargs):
            """
            self: 通常是 pipeline_scheduler.PipelineScheduler 实例
            task_state: 每个任务独立的状态
            *args, **kwargs: 可能的额外参数
            """
            worker_display_name = worker_name or func.__name__
            wlogger = getattr(self, 'logger', logger)

            # 队列引用
            input_queue = getattr(task_state, input_queue_attr)
            next_queue = getattr(task_state, next_queue_attr) if next_queue_attr else None

            wlogger.info(f"[{worker_display_name}] 启动 (TaskID={task_state.task_id}). "
                         f"输入队列: {input_queue_attr}, 下游队列: {next_queue_attr if next_queue_attr else '无'}")

            processed_count = 0  # 记录本Worker已处理多少item
            try:
                while True:
                    try:
                        queue_size_before = input_queue.qsize()
                        item = await input_queue.get()
                        if item is None:
                            # 结束信号
                            if next_queue:
                                await next_queue.put(None)
                            wlogger.info(f"[{worker_display_name}] 收到停止信号。已处理 {processed_count} 个item。")
                            break

                        # 记录处理前队列长度
                        wlogger.debug(f"[{worker_display_name}] 从 {input_queue_attr} 队列取出一个item. "
                                      f"队列剩余: {queue_size_before} -> (处理中)")

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
                        # 出错后可选择是否结束下游
                        if next_queue:
                            await next_queue.put(None)
                        break
            finally:
                wlogger.info(f"[{worker_display_name}] 结束 (TaskID={task_state.task_id}). 共处理 {processed_count} 个item.")
        return wrapper
    return decorator
