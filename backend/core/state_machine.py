from enum import Enum
from typing import Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

class TaskState(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PREPROCESSING = "preprocessing"
    PREPROCESSED = "preprocessed"
    TRANSLATING = "translating"
    TRANSLATED = "translated"
    TTS_PROCESSING = "tts_processing"
    TTS_COMPLETED = "tts_completed"
    ALIGNING = "aligning"
    ALIGNED = "aligned"
    MIXING = "mixing"
    MIXED = "mixed"
    HLS_PROCESSING = "hls_processing"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"

class StateTransition:
    """状态转换规则"""
    
    # 定义允许的状态转换
    ALLOWED_TRANSITIONS: Dict[TaskState, Set[TaskState]] = {
        TaskState.PENDING: {TaskState.UPLOADING, TaskState.ERROR},
        TaskState.UPLOADING: {TaskState.UPLOADED, TaskState.ERROR},
        TaskState.UPLOADED: {TaskState.PREPROCESSING, TaskState.ERROR},
        TaskState.PREPROCESSING: {TaskState.PREPROCESSED, TaskState.ERROR},
        TaskState.PREPROCESSED: {TaskState.TRANSLATING, TaskState.TTS_PROCESSING, TaskState.ERROR},
        TaskState.TRANSLATING: {TaskState.TRANSLATED, TaskState.ERROR, TaskState.PAUSED},
        TaskState.TRANSLATED: {TaskState.TTS_PROCESSING, TaskState.ERROR},
        TaskState.TTS_PROCESSING: {TaskState.TTS_COMPLETED, TaskState.ERROR, TaskState.PAUSED},
        TaskState.TTS_COMPLETED: {TaskState.ALIGNING, TaskState.ERROR},
        TaskState.ALIGNING: {TaskState.ALIGNED, TaskState.ERROR},
        TaskState.ALIGNED: {TaskState.MIXING, TaskState.ERROR},
        TaskState.MIXING: {TaskState.MIXED, TaskState.ERROR},
        TaskState.MIXED: {TaskState.HLS_PROCESSING, TaskState.ERROR},
        TaskState.HLS_PROCESSING: {TaskState.COMPLETED, TaskState.ERROR},
        TaskState.ERROR: {TaskState.PREPROCESSING, TaskState.TRANSLATING, TaskState.TTS_PROCESSING},
        TaskState.PAUSED: {TaskState.TRANSLATING, TaskState.TTS_PROCESSING, TaskState.ERROR}
    }
    
    # 定义每个状态对应的处理actor
    STATE_ACTORS: Dict[TaskState, str] = {
        TaskState.UPLOADED: "video_separator",
        TaskState.PREPROCESSING: "asr_model", 
        TaskState.TRANSLATING: "translator",
        TaskState.TTS_PROCESSING: "my_index_tts",
        TaskState.ALIGNING: "duration_aligner",
        TaskState.MIXING: "media_mixer",
        TaskState.HLS_PROCESSING: "hls_manager"
    }
    
    @classmethod
    def can_transition(cls, from_state: TaskState, to_state: TaskState) -> bool:
        """检查是否允许状态转换"""
        return to_state in cls.ALLOWED_TRANSITIONS.get(from_state, set())
    
    @classmethod
    def get_next_states(cls, current_state: TaskState) -> Set[TaskState]:
        """获取当前状态的所有可能下一状态"""
        return cls.ALLOWED_TRANSITIONS.get(current_state, set())
    
    @classmethod
    def get_responsible_actor(cls, state: TaskState) -> Optional[str]:
        """获取负责处理该状态的actor"""
        return cls.STATE_ACTORS.get(state)

class TaskStateMachine:
    """任务状态机管理器"""
    
    def __init__(self, supabase_client):
        self.supabase_client = supabase_client
        self.logger = logger
    
    async def transition_state(self, task_id: str, new_state: TaskState, 
                             error_message: str = None) -> bool:
        """执行状态转换"""
        try:
            # 获取当前状态
            task = await self.supabase_client.get_task(task_id)
            if not task:
                self.logger.error(f"任务 {task_id} 不存在")
                return False
            
            current_state = TaskState(task.get('status', 'pending'))
            
            # 检查转换是否合法
            if not StateTransition.can_transition(current_state, new_state):
                self.logger.error(f"任务 {task_id} 不允许从 {current_state.value} 转换到 {new_state.value}")
                return False
            
            # 更新状态
            update_data = {'status': new_state.value}
            if error_message:
                update_data['error_message'] = error_message
            elif new_state != TaskState.ERROR:
                update_data['error_message'] = None
            
            await self.supabase_client.update_task(task_id, update_data)
            self.logger.info(f"任务 {task_id} 状态从 {current_state.value} 转换到 {new_state.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"任务 {task_id} 状态转换失败: {e}")
            return False
    
    async def get_tasks_by_state(self, state: TaskState) -> List[Dict]:
        """获取指定状态的所有任务"""
        try:
            from supabase import create_client
            client = await self.supabase_client._ensure_client()
            response = await client.table('tasks').select('*').eq('status', state.value).execute()
            return response.data
        except Exception as e:
            self.logger.error(f"获取状态 {state.value} 的任务失败: {e}")
            return []
    
    async def can_process_task(self, task_id: str, actor_name: str) -> bool:
        """检查actor是否可以处理该任务"""
        try:
            task = await self.supabase_client.get_task(task_id)
            if not task:
                return False
            
            current_state = TaskState(task.get('status'))
            responsible_actor = StateTransition.get_responsible_actor(current_state)
            
            return responsible_actor == actor_name
        except Exception as e:
            self.logger.error(f"检查任务 {task_id} 处理权限失败: {e}")
            return False 