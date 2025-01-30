import asyncio
import logging
from typing import Dict, List, AsyncGenerator, Protocol, TypeVar, Generic, Optional
from dataclasses import dataclass
from .prompt import (
    TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_USER_PROMPT,
    SIMPLIFICATION_SYSTEM_PROMPT,
    SIMPLIFICATION_USER_PROMPT,
    LANGUAGE_MAP
)

logger = logging.getLogger(__name__)

class TranslationClient(Protocol):
    async def translate(
        self,
        texts: Dict[str, str],
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, str]:
        ...

@dataclass
class BatchConfig:
    """批处理配置"""
    initial_size: int = 100
    min_size: int = 1
    required_successes: int = 2
    retry_delay: float = 0.1

T = TypeVar('T')

class Translator:
    def __init__(self, translation_client: TranslationClient):
        self.translation_client = translation_client
        self.logger = logging.getLogger(__name__)

    async def translate(self, texts: Dict[str, str], target_language: str = "zh") -> Dict[str, str]:
        """执行翻译并处理结果（字典形式）"""
        try:
            system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
                target_language=LANGUAGE_MAP.get(target_language, target_language)
            )
            user_prompt = TRANSLATION_USER_PROMPT.format(
                target_language=LANGUAGE_MAP.get(target_language, target_language),
                json_content=texts
            )
            return await self.translation_client.translate(
                texts=texts,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            self.logger.error(f"翻译失败: {str(e)}")
            raise

    async def simplify(self, texts: Dict[str, str]) -> Dict[str, str]:
        """执行文本简化（字典形式）"""
        try:
            system_prompt = SIMPLIFICATION_SYSTEM_PROMPT
            user_prompt = SIMPLIFICATION_USER_PROMPT.format(json_content=texts)
            return await self.translation_client.translate(
                texts=texts,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            self.logger.error(f"简化失败: {str(e)}")
            raise

    async def _process_batch(
        self,
        items: List[T],
        process_func: callable,
        config: BatchConfig,
        error_handler: Optional[callable] = None,
        reduce_batch_on_error: bool = True  # <-- 新增参数
    ) -> AsyncGenerator[List[T], None]:
        """通用批处理逻辑"""
        if not items:
            return

        i = 0
        batch_size = config.initial_size
        success_count = 0

        while i < len(items):
            try:
                batch = items[i:i+batch_size]
                if not batch:
                    break

                results = await process_func(batch)
                
                if results:
                    success_count += 1
                    yield results
                    i += len(batch)

                    # 如果连续成功且当前批次小于初始大小，尝试恢复
                    if reduce_batch_on_error and batch_size < config.initial_size and success_count >= config.required_successes:
                        self.logger.debug(f"连续成功{success_count}次，恢复到初始批次大小: {config.initial_size}")
                        batch_size = config.initial_size
                        success_count = 0

                if i < len(items):
                    await asyncio.sleep(config.retry_delay)

            except Exception as e:
                self.logger.error(f"批处理失败: {str(e)}")
                if reduce_batch_on_error and batch_size > config.min_size:
                    # 只有在 reduce_batch_on_error=True 时才减小批次大小
                    batch_size = max(batch_size // 2, config.min_size)
                    success_count = 0
                    self.logger.debug(f"出错后减小批次大小到: {batch_size}")
                    continue
                else:
                    # 出错但我们不打算减小批次大小，或者已经到达最小批次
                    if error_handler:
                        yield error_handler(batch)
                    i += len(batch)

    async def translate_sentences(
        self,
        sentences: List,
        batch_size: int = 100,
        target_language: str = "zh"
    ) -> AsyncGenerator[List, None]:
        """批量翻译处理，需要在出错时减小批次大小 -> reduce_batch_on_error=True"""
        if not sentences:
            self.logger.warning("收到空的句子列表")
            return

        config = BatchConfig(initial_size=batch_size)

        async def process_batch(batch: List) -> Optional[List]:
            texts = {str(j): s.raw_text for j, s in enumerate(batch)}
            self.logger.debug(f"翻译批次: {len(texts)}条文本")
            
            translated = await self.translate(texts, target_language)
            
            if len(translated) == len(texts):
                for j, sentence in enumerate(batch):
                    sentence.trans_text = translated[str(j)]
                return batch
            return None

        def handle_error(batch: List) -> List:
            for sentence in batch:
                sentence.trans_text = sentence.raw_text
            return batch

        # 这里保持默认或显式写出 reduce_batch_on_error=True
        async for batch_result in self._process_batch(
            sentences,
            process_batch,
            config,
            error_handler=handle_error,
            reduce_batch_on_error=True
        ):
            yield batch_result

    async def simplify_sentences(
        self,
        sentences: List,
        batch_size: int = 4
    ) -> AsyncGenerator[List, None]:
        """
        执行文本简化并处理结果，与 translate_sentences 类似，但我们
        不需要在出错时自动缩小批次大小 -> reduce_batch_on_error=False
        """
        if not sentences:
            self.logger.warning("收到空的句子列表")
            return

        config = BatchConfig(initial_size=batch_size, min_size=1, required_successes=2)

        async def process_batch(batch: List) -> Optional[List]:
            texts = {str(i): s.trans_text for i, s in enumerate(batch)}
            self.logger.debug(f"简化批次: {len(texts)}条文本")

            batch_result = await self.simplify(texts)

            if len(batch_result) == len(texts):
                for i, s in enumerate(batch):
                    old_text = s.trans_text
                    new_text = batch_result[str(i)]
                    s.trans_text = new_text
                    self.logger.info(f"精简: {old_text} -> {new_text}")
                return batch
            return None

        def handle_error(batch: List) -> List:
            # 出错时，原样返回，不做任何更改
            return batch

        # 这里显式指定 reduce_batch_on_error=False
        async for batch_result in self._process_batch(
            sentences,
            process_batch,
            config,
            error_handler=handle_error,
            reduce_batch_on_error=False
        ):
            yield batch_result
