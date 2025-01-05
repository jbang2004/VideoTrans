import logging
import functools
from typing import Callable, Any, Optional, Coroutine, AsyncGenerator, TypeVar, Union, Literal
import asyncio

T = TypeVar('T')
WorkerResult = Union[T, AsyncGenerator[T, None]]
WorkerMode = Literal['base', 'stream']

def handle_errors(logger: Optional[logging.Logger] = None) -> Callable:
    """错误处理装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # 如果第一个参数是实例且有 logger 属性，使用实例的 logger
            actual_logger = logger
            if args and hasattr(args[0], 'logger'):
                actual_logger = args[0].logger
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if actual_logger:
                    actual_logger.error(f"{func.__name__} 失败: {str(e)}")
                raise
        return wrapper
    return decorator

def worker_decorator(
    input_queue_attr: str,
    next_queue_attr: Optional[str] = None,
    worker_name: Optional[str] = None,
    mode: WorkerMode = 'base'
) -> Callable:
    """统一的worker装饰器
    
    Args:
        input_queue_attr: 输入队列的属性名
        next_queue_attr: 输出队列的属性名
        worker_name: worker显示名称
        mode: 工作模式 ('base'|'stream')
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs) -> None:
            worker_display_name = worker_name or func.__name__.replace('_', ' ')
            logger = self.logger
            input_queue = getattr(self.task_state, input_queue_attr)
            next_queue = getattr(self.task_state, next_queue_attr) if next_queue_attr else None
            logger.info(f"{worker_display_name}启动")
            
            try:
                while True:
                    try:
                        item = await input_queue.get()
                        self.logger.debug(f"{worker_name} 收到数据")
                        
                        if item is None:
                            if next_queue:
                                await next_queue.put(None)
                            logger.info(f"{worker_display_name}收到停止信号")
                            break
                            
                        if mode == 'stream':
                            # 流式处理模式
                            async for result in func(self, item, *args, **kwargs):
                                if result is not None and next_queue:
                                    self.logger.debug(f"{worker_name} 输出数据")
                                    await next_queue.put(result)
                        else:
                            # 基础处理模式
                            result = await func(self, item, *args, **kwargs)
                            if result is not None and next_queue:
                                self.logger.debug(f"{worker_name} 输出数据")
                                await next_queue.put(result)
                                
                    except asyncio.CancelledError:
                        logger.info(f"{worker_display_name}被取消")
                        if next_queue:
                            await next_queue.put(None)
                        break
                    except Exception as e:
                        logger.error(f"{worker_display_name}异常: {str(e)}")
                        if next_queue:
                            await next_queue.put(None)
                        break
                        
            finally:
                logger.info(f"{worker_display_name}结束")
                
        return wrapper
    return decorator