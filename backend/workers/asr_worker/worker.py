import logging
from typing import Dict, Any
from utils.task_state import TaskState
from utils.decorators import worker_decorator
from .auto_sense import SenseAutoModel

logger = logging.getLogger(__name__)

class ASRWorker:
    """语音识别 Worker"""

    def __init__(self, config):
        """初始化 ASRWorker"""
        self.config = config
        self.logger = logger
        
        # 直接初始化 ASR 模型
        self.sense_model = SenseAutoModel(
            config=config,
            **config.ASR_MODEL_KWARGS
        )

    @worker_decorator(
        input_queue_attr='asr_queue',
        next_queue_attr='translation_queue',
        worker_name='ASR Worker'
    )
    async def run(self, segment_info: Dict[str, Any], task_state: TaskState) -> Dict[str, Any]:
        """处理音频识别"""
        try:
            self.logger.debug(
                f"[ASR Worker] 开始处理分段 {segment_info['segment_index']} -> TaskID={task_state.task_id}"
            )
            # 调用 ASR 模型（异步接口）
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
                    f"[ASR Worker] 分段 {segment_info['segment_index']} 未识别到语音 -> TaskID={task_state.task_id}"
                )
                return None

            self.logger.info(
                f"[ASR Worker] 识别到 {len(sentences)} 条句子, seg={segment_info['segment_index']}, TaskID={task_state.task_id}"
            )
            for s in sentences:
                s.segment_index = segment_info['segment_index']
                s.segment_start = segment_info['start']
                s.task_id = task_state.task_id
                s.sentence_id = task_state.sentence_counter
                task_state.sentence_counter += 1

            return sentences

        except Exception as e:
            self.logger.error(
                f"[ASR Worker] 分段 {segment_info['segment_index']} 处理失败: {e} -> TaskID={task_state.task_id}",
                exc_info=True
            )
            task_state.errors.append({
                'segment_index': segment_info['segment_index'],
                'stage': 'asr',
                'error': str(e)
            })
            return None

if __name__ == '__main__':
    print("ASR Worker 模块加载成功")
