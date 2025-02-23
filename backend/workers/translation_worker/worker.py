import logging
from typing import List, Any
from utils.decorators import worker_decorator
from utils.task_state import TaskState
from .translation.translator import Translator
from .translation.deepseek_client import DeepSeekClient
from .translation.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

class TranslationWorker:
    """
    翻译 Worker：调用 Translator 对句子进行批量翻译。
    """

    def __init__(self, config):
        """初始化 TranslationWorker"""
        self.config = config
        self.logger = logger
        
        # 根据配置选择翻译客户端
        translation_model = config.TRANSLATION_MODEL.lower()
        if translation_model == "deepseek":
            client = DeepSeekClient(api_key=config.DEEPSEEK_API_KEY)
        elif translation_model == "gemini":
            client = GeminiClient(api_key=config.GEMINI_API_KEY)
        else:
            raise ValueError(f"不支持的翻译模型: {translation_model}")
        
        # 直接初始化 Translator
        self.translator = Translator(translation_client=client)

    @worker_decorator(
        input_queue_attr='translation_queue',
        next_queue_attr='modelin_queue',
        worker_name='翻译 Worker',
        mode='stream'
    )
    async def run(self, sentences_list: List[Any], task_state: TaskState):
        if not sentences_list:
            return
        self.logger.debug(f"[翻译 Worker] 收到 {len(sentences_list)} 句子, TaskID={task_state.task_id}")

        async for translated_batch in self.translator.translate_sentences(
            sentences_list,
            batch_size=self.config.TRANSLATION_BATCH_SIZE,
            target_language=task_state.target_language
        ):
            yield translated_batch

if __name__ == '__main__':
    print("Translation Worker 模块加载成功")