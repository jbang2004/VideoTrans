#!/usr/bin/env python3
"""
Worker 启动器：用于启动所有 worker 进程
"""
import os
import sys
import asyncio
import logging
import signal
import argparse
from pathlib import Path
from config import Config
from workers.segment_worker.worker import SegmentWorker
from workers.asr_worker.worker import ASRWorker
from workers.translation_worker.worker import TranslationWorker
from workers.modelin_worker.worker import ModelInWorker
from workers.tts_worker.worker import TTSTokenWorker
from workers.duration_worker.worker import DurationWorker
from workers.audio_gen_worker.worker import AudioGenWorker
from workers.mixer_worker.worker import MixerWorker

# 导入日志配置模块
from utils.log_config import configure_logging

# 初始化日志系统
log_dir = configure_logging()
logger = logging.getLogger(__name__)

# 定义所有工作者类及其显示名称
WORKERS = {
    'segment': (SegmentWorker, "分段 Worker"),
    'asr': (ASRWorker, "ASR Worker"),
    'translation': (TranslationWorker, "翻译 Worker"),
    'modelin': (ModelInWorker, "模型输入 Worker"),
    'tts': (TTSTokenWorker, "TTS Token生成 Worker"),
    'duration': (DurationWorker, "时长对齐 Worker"),
    'audio': (AudioGenWorker, "音频生成 Worker"),
    'mixer': (MixerWorker, "混音 Worker"),
}

async def start_worker(worker_class, config, worker_name):
    """启动单个 worker"""
    try:
        logger.info(f"启动 {worker_name}...")
        worker = worker_class(config)
        
        # 对于需要特殊初始化的 worker
        if hasattr(worker, 'initialize') and callable(worker.initialize):
            worker = await worker.initialize()
        
        # 针对不同类型的worker调用不同的方法
        if worker_class == SegmentWorker:
            logger.info(f"{worker_name} 开始处理队列...")
            await asyncio.gather(
                worker.run_init(),
                worker.run_extract()
            )
        elif hasattr(worker, 'run') and callable(worker.run):
            logger.info(f"{worker_name} 开始处理队列...")
            await worker.run()
        else:
            logger.error(f"{worker_name} 没有可调用的run方法")
            
        # 创建一个永不结束的任务，让 worker 保持运行
        await asyncio.Future()
    except Exception as e:
        logger.error(f"{worker_name} 启动失败: {e}", exc_info=True)

async def start_all_workers(config):
    """启动所有 worker"""
    # 创建所有 worker 任务
    tasks = []
    for worker_id, (worker_class, worker_name) in WORKERS.items():
        task = asyncio.create_task(start_worker(worker_class, config, worker_name))
        tasks.append(task)
    
    # 等待所有 worker 完成
    await asyncio.gather(*tasks)

def handle_signal(sig, frame):
    """处理信号"""
    logger.info(f"收到信号 {sig}，准备关闭所有 worker...")
    for task in asyncio.all_tasks():
        task.cancel()
    
    # 强制退出
    sys.exit(0)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动 Redis 队列 Worker")
    parser.add_argument('--worker', type=str, 
                        help=f'指定要启动的 worker: {", ".join(WORKERS.keys())}, all')
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    config = Config()
    
    if args.worker in WORKERS:
        worker_class, worker_name = WORKERS[args.worker]
        asyncio.run(start_worker(worker_class, config, worker_name))
    elif args.worker == 'all' or args.worker is None:
        # 启动所有 worker
        asyncio.run(start_all_workers(config))
    else:
        logger.error("无效的 worker 名称")

if __name__ == "__main__":
    main()
