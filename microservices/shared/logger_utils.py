import logging
import sys
from pathlib import Path
from pythonjsonlogger import jsonlogger
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None, json_format: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    if json_format:
        formatter = jsonlogger.JsonFormatter(fmt='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

class ServiceLogger:
    """
    微服务专用日志记录器，提供统一的日志记录接口
    """
    def __init__(self, service_name: str, task_id: Optional[str] = None, log_file: Optional[str] = None):
        self.service_name = service_name
        self.task_id = task_id
        self.logger = setup_logger(name=service_name, log_file=log_file, json_format=True)

    def _format_message(self, message: str) -> str:
        return f"[TaskID={self.task_id}] {message}" if self.task_id else message

    def info(self, message: str, **kwargs):
        extra = {"service": self.service_name, **kwargs}
        if self.task_id:
            extra["task_id"] = self.task_id
        self.logger.info(self._format_message(message), extra=extra)

    def error(self, message: str, exc_info=False, **kwargs):
        extra = {"service": self.service_name, **kwargs}
        if self.task_id:
            extra["task_id"] = self.task_id
        self.logger.error(self._format_message(message), exc_info=exc_info, extra=extra)

    def warning(self, message: str, **kwargs):
        extra = {"service": self.service_name, **kwargs}
        if self.task_id:
            extra["task_id"] = self.task_id
        self.logger.warning(self._format_message(message), extra=extra)

    def debug(self, message: str, **kwargs):
        extra = {"service": self.service_name, **kwargs}
        if self.task_id:
            extra["task_id"] = self.task_id
        self.logger.debug(self._format_message(message), extra=extra)

    def set_task_id(self, task_id: str):
        self.task_id = task_id
