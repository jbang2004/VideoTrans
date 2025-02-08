import logging
from typing import List, Dict, Any

from core.auto_sense import SenseAutoModel
from utils.task_state import TaskState
from utils.decorators import worker_decorator
from core.sentence_tools import Sentence

logger = logging.getLogger(__name__)

class ASRWorker:
    """语音识别Worker"""
    
    def __init__(self, sense_model: SenseAutoModel):
        self.sense_model = sense_model
        self.logger = logger
        
    @worker_decorator(
        input_queue_attr='asr_queue',
        next_queue_attr='translation_queue',
        worker_name='ASR Worker'
    )
    async def asr_maker(self, segment_info: Dict[str, Any], task_state: TaskState) -> Dict[str, Any]:
        """处理音频识别"""
        try:
            self.logger.debug(
                f"[语音识别Worker] 开始处理分段 {segment_info['segment_index']} -> "
                f"TaskID={task_state.task_id}"
            )
            
            # 执行ASR
            sentences = await self.sense_model.generate_async(
                input=segment_info['vocals_path'],
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False
            )
            
            if not sentences:
                self.logger.warning(
                    f"[语音识别Worker] 分段 {segment_info['segment_index']} 未识别到语音 -> "
                    f"TaskID={task_state.task_id}"
                )
                return None
                
            self.logger.info(
                f"[语音识别Worker] 识别到 {len(sentences)} 条句子, "
                f"seg={segment_info['segment_index']}, TaskID={task_state.task_id}"
            )
            
            # 设置句子属性
            for s in sentences:
                s.segment_index = segment_info['segment_index']
                s.segment_start = segment_info['start']
                s.task_id = task_state.task_id
                s.sentence_id = task_state.sentence_counter
                task_state.sentence_counter += 1
                
            return sentences
            
        except Exception as e:
            self.logger.error(
                f"[语音识别Worker] 处理分段 {segment_info['segment_index']} 失败: {e} -> "
                f"TaskID={task_state.task_id}"
            )
            task_state.errors.append({
                'segment_index': segment_info['segment_index'],
                'stage': 'asr',
                'error': str(e)
            })
            return None 