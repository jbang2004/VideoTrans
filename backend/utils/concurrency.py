# utils/concurrency.py
import functools
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

CPU_COUNT = os.cpu_count() or 1
GLOBAL_EXECUTOR = ThreadPoolExecutor(max_workers=8)

async def run_sync(func, *args, **kwargs):
    """在线程池中运行同步函数"""
    loop = asyncio.get_running_loop()
    partial_func = functools.partial(func, *args, **kwargs)
    return await loop.run_in_executor(GLOBAL_EXECUTOR, partial_func)