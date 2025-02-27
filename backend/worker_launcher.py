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
from config import Config
from workers.segment_worker.worker import SegmentWorker
from workers.asr_worker.worker import ASRWorker
from workers.translation_worker.worker import TranslationWorker
from workers.modelin_worker.worker import ModelInWorker
from workers.tts_worker.worker import TTSTokenWorker
from workers.duration_worker.worker import DurationWorker
from workers.audio_gen_worker.worker import AudioGenWorker
from workers.mixer_worker.worker import MixerWorker

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def start_worker(worker_class, config, worker_name):
    """启动单个 worker"""
    try:
        logger.info(f"启动 {worker_name}...")
        worker = worker_class(config)
        
        # 对于需要特殊初始化的 worker
        if hasattr(worker, 'initialize') and callable(worker.initialize):
            worker = await worker.initialize()
        
        # 创建一个永不结束的任务，让 worker 保持运行
        # 不再调用 worker.run()，而是让装饰器处理队列监听
        await asyncio.Future()
    except Exception as e:
        logger.error(f"{worker_name} 启动失败: {e}", exc_info=True)

async def start_all_workers(config):
    """启动所有 worker"""
    workers = [
        (SegmentWorker, "分段初始化 Worker"),
        (ASRWorker, "ASR Worker"),
        (TranslationWorker, "翻译 Worker"),
        (ModelInWorker, "模型输入 Worker"),
        (TTSTokenWorker, "TTS Token生成 Worker"),
        (DurationWorker, "时长对齐 Worker"),
        (AudioGenWorker, "音频生成 Worker"),
        (MixerWorker, "混音 Worker"),
    ]
    
    # 创建所有 worker 任务
    tasks = []
    for worker_class, worker_name in workers:
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
    parser.add_argument('--worker', type=str, help='指定要启动的 worker: segment, asr, translation, modelin, tts, duration, audio, mixer, all')
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    config = Config()
    
    if args.worker == 'segment':
        asyncio.run(start_worker(SegmentWorker, config, "分段 Worker"))
    elif args.worker == 'asr':
        asyncio.run(start_worker(ASRWorker, config, "ASR Worker"))
    elif args.worker == 'translation':
        asyncio.run(start_worker(TranslationWorker, config, "翻译 Worker"))
    elif args.worker == 'modelin':
        asyncio.run(start_worker(ModelInWorker, config, "模型输入 Worker"))
    elif args.worker == 'tts':
        asyncio.run(start_worker(TTSTokenWorker, config, "TTS Token生成 Worker"))
    elif args.worker == 'duration':
        asyncio.run(start_worker(DurationWorker, config, "时长对齐 Worker"))
    elif args.worker == 'audio':
        asyncio.run(start_worker(AudioGenWorker, config, "音频生成 Worker"))
    elif args.worker == 'mixer':
        asyncio.run(start_worker(MixerWorker, config, "混音 Worker"))
    else:
        # 启动所有 worker
        asyncio.run(start_all_workers(config))

if __name__ == "__main__":
    main()
