#!/usr/bin/env python3
"""
日志配置模块 - 提供统一的日志配置功能
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Union, List

# 默认日志目录
DEFAULT_LOG_DIR = Path("/tmp/cosysense_logs")

# 关键模块日志路径
CORE_MODULES = [
    'workers',
    'workers.segment_worker',
    'workers.asr_worker', 
    'workers.mixer_worker',
    'workers.translation_worker',
    'workers.modelin_worker',
    'workers.tts_worker',
    'workers.duration_worker',
    'workers.audio_gen_worker',
    'utils',
    'utils.worker_decorators',
    'utils.redis_utils',
    'utils.task_state'
]

def configure_logging(
    log_dir: Optional[Union[str, Path]] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    modules: Optional[List[str]] = None
) -> Path:
    """
    配置日志系统
    
    Args:
        log_dir: 日志目录路径，默认为 /tmp/cosysense_logs
        console_level: 控制台日志级别，默认为 INFO
        file_level: 文件日志级别，默认为 DEBUG
        modules: 需要特别配置的模块，默认为预定义的核心模块列表
        
    Returns:
        日志目录路径
    """
    # 设置日志目录
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR
    else:
        log_dir = Path(log_dir)
    
    # 确保日志目录存在
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 总是设置为最低级别，让处理器决定显示级别
    
    # 清除已有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器 - 所有日志
    main_log_file = log_dir / "worker.log"
    file_handler = logging.FileHandler(main_log_file)
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # 为特定模块配置日志器
    if modules is None:
        modules = CORE_MODULES
    
    for module_name in modules:
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.DEBUG)  # 每个模块的日志器设置为最低级别
        
        # 为worker模块添加单独的日志文件
        if module_name.startswith('workers.') and '.' in module_name:
            worker_name = module_name.split('.')[-1]
            worker_log_file = log_dir / f"{worker_name}.log"
            
            # 检查是否已有该处理器
            has_handler = any(
                isinstance(h, logging.FileHandler) and 
                getattr(h, 'baseFilename', '') == str(worker_log_file)
                for h in logger.handlers
            )
            
            if not has_handler:
                worker_handler = logging.FileHandler(worker_log_file)
                worker_handler.setLevel(file_level)
                worker_handler.setFormatter(file_formatter)
                logger.addHandler(worker_handler)
    
    # 记录配置完成信息
    logging.info(f"日志系统配置完成。主日志文件: {main_log_file}")
    return log_dir

def get_logger(name: str, log_dir: Optional[Union[str, Path]] = None) -> logging.Logger:
    """
    获取已配置的日志器，如果日志系统未配置则先配置
    
    Args:
        name: 日志器名称
        log_dir: 日志目录，如果日志系统未配置时使用
        
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    
    # 检查是否需要配置日志系统
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        configure_logging(log_dir=log_dir)
    
    return logger
