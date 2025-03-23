from ray import serve
import asyncio
from pathlib import Path
import logging
from typing import Dict, Any, Optional

from config import Config
from utils.task_storage import TaskPaths
from utils.task_state import TaskState

logger = logging.getLogger(__name__)

@serve.deployment(
    name="StateManager"
)
class StateManager:
    """
    集中式任务状态管理器
    
    负责:
    1. 创建和管理任务状态
    2. 提供任务状态查询
    3. 提供任务状态更新
    4. 提供HLS管理器引用
    """
    def __init__(self, config=None):
        self.config = config or Config()
        self.tasks = {}  # 存储所有任务状态
        self.locks = {}  # 用于任务级别的并发控制
        logger.info("StateManager初始化完成")
    
    async def create_task(self, task_id, video_path, target_language, generate_subtitle):
        """
        创建新任务并返回任务状态
        
        Args:
            task_id: 任务ID
            video_path: 视频路径
            target_language: 目标语言
            generate_subtitle: 是否生成字幕
            
        Returns:
            dict: 包含task_state的字典
        """
        self.locks[task_id] = asyncio.Lock()
        
        async with self.locks[task_id]:
            logger.info(f"创建任务: {task_id}, 视频: {video_path}, 语言: {target_language}")
            
            # 创建任务路径
            task_paths = TaskPaths(self.config, task_id)
            task_paths.create_directories()
            
            # 创建任务状态
            task_state = TaskState(
                task_id=task_id,
                video_path=video_path,
                task_paths=task_paths,
                target_language=target_language,
                generate_subtitle=generate_subtitle
            )
            
            # 存储任务信息
            self.tasks[task_id] = {
                "status": "processing",
                "message": "任务初始化完成",
                "progress": 0,
                "hls_ready": False,
                "task_state": task_state,
                "create_time": asyncio.get_event_loop().time()
            }
            
            return {
                "task_state": task_state
            }
    
    async def get_task_state(self, task_id):
        """获取任务状态对象"""
        if task_id not in self.tasks:
            logger.warning(f"任务不存在: {task_id}")
            return None
        return self.tasks[task_id]["task_state"]
    

    
    async def get_task_status(self, task_id):
        """
        获取任务状态信息（面向API）
        
        Returns:
            dict: 包含status, message, progress, hls_ready等信息
        """
        if task_id not in self.tasks:
            logger.warning(f"获取任务状态失败, 任务不存在: {task_id}")
            return None
        
        # 返回不包含task_state和hls_manager的任务信息副本
        status_info = {
            k: v for k, v in self.tasks[task_id].items() 
            if k not in ["task_state", "hls_manager", "create_time"]
        }
        
        # 添加完成百分比
        status_info["progress"] = self._calculate_progress(task_id)
        
        return status_info
    
    def _calculate_progress(self, task_id):
        """根据batch_counter计算进度百分比"""
        if task_id not in self.tasks:
            return 0
        
        task_state = self.tasks[task_id]["task_state"]
        
        # 根据批次计数器计算大致进度，假设最多25个批次
        if task_state.batch_counter > 0:
            # 最多到95%，留5%给最终合并步骤
            progress = min(95, int(task_state.batch_counter * 4))
        else:
            # 初始进度5%
            progress = 5
            
        # 如果状态是成功，则设为100%
        if self.tasks[task_id]["status"] == "success":
            progress = 100
            
        return progress
    
    async def update_task_status(self, task_id, updates):
        """
        更新任务状态信息
        
        Args:
            task_id: 任务ID
            updates: 要更新的状态字段字典
        """
        if task_id not in self.tasks:
            logger.warning(f"更新任务状态失败, 任务不存在: {task_id}")
            return False
        
        async with self.locks[task_id]:
            logger.debug(f"更新任务状态: {task_id}, 更新: {updates}")
            self.tasks[task_id].update(updates)
            return True
    
    async def update_task_progress(self, task_id, batch_counter=None, hls_ready=None):
        """
        更新任务进度信息
        
        Args:
            task_id: 任务ID
            batch_counter: 批次计数器
            hls_ready: HLS是否就绪
        """
        if task_id not in self.tasks:
            logger.warning(f"更新任务进度失败, 任务不存在: {task_id}")
            return False
        
        async with self.locks[task_id]:
            # 更新任务状态对象
            task_state = self.tasks[task_id]["task_state"]
            
            if batch_counter is not None:
                task_state.batch_counter = batch_counter
                logger.debug(f"更新任务进度: {task_id}, 批次: {batch_counter}")
            
            if hls_ready is not None:
                task_state.hls_ready = hls_ready
                self.tasks[task_id]["hls_ready"] = hls_ready
                logger.info(f"更新HLS状态: {task_id}, hls_ready: {hls_ready}")
            
            return True
    
    async def complete_task(self, task_id, success=True, message=None, final_video_path=None):
        """
        标记任务完成
        
        Args:
            task_id: 任务ID
            success: 是否成功
            message: 完成消息
            final_video_path: 最终视频路径
        """
        if task_id not in self.tasks:
            logger.warning(f"标记任务完成失败, 任务不存在: {task_id}")
            return False
        
        async with self.locks[task_id]:
            status = "success" if success else "error"
            msg = message or ("处理完成" if success else "处理失败")
            
            logger.info(f"标记任务完成: {task_id}, 状态: {status}, 消息: {msg}")
            
            self.tasks[task_id].update({
                "status": status,
                "message": msg,
                "progress": 100 if success else self.tasks[task_id].get("progress", 0),
                "final_video_path": str(final_video_path) if final_video_path else None
            })
            
            # 如果任务成功完成，可以清理一些内存资源
            if success:
                # 保留任务状态但清除不再需要的大对象引用
                if "hls_manager" in self.tasks[task_id]:
                    del self.tasks[task_id]["hls_manager"]
            
            return True
    
    async def cleanup_old_tasks(self, max_age_hours=24):
        """
        清理超过指定时间的旧任务
        
        Args:
            max_age_hours: 最大保留时间(小时)
        """
        current_time = asyncio.get_event_loop().time()
        
        tasks_to_delete = []
        for task_id, task_info in self.tasks.items():
            if "create_time" not in task_info:
                continue
                
            age = (current_time - task_info["create_time"]) / 3600  # 转换为小时
            if age > max_age_hours:
                tasks_to_delete.append(task_id)
        
        for task_id in tasks_to_delete:
            async with self.locks.get(task_id, asyncio.Lock()):
                logger.info(f"清理过期任务: {task_id}")
                if task_id in self.tasks:
                    del self.tasks[task_id]
                if task_id in self.locks:
                    del self.locks[task_id] 