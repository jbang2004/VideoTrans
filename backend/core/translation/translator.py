import asyncio
import logging
from typing import Dict, List, AsyncGenerator, Protocol, Optional, TypeVar
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
        """执行翻译并返回包含 "thinking" 与 "output" 字段的 JSON"""
        try:
            system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
                target_language=LANGUAGE_MAP.get(target_language, target_language)
            )
            user_prompt = TRANSLATION_USER_PROMPT.format(
                target_language=LANGUAGE_MAP.get(target_language, target_language),
                json_content=texts
            )
            return await self.translation_client.translate(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            self.logger.error(f"翻译失败: {str(e)}")
            raise

    async def simplify(self, texts: Dict[str, str]) -> Dict[str, str]:
        """执行简化并返回包含 "thinking"、"slight"、"moderate"、"extreme" 字段的 JSON"""
        try:
            system_prompt = SIMPLIFICATION_SYSTEM_PROMPT
            user_prompt = SIMPLIFICATION_USER_PROMPT.format(json_content=texts)
            return await self.translation_client.translate(
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
        reduce_batch_on_error: bool = True
    ) -> AsyncGenerator[List[T], None]:
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
                    if reduce_batch_on_error and batch_size < config.initial_size and success_count >= config.required_successes:
                        self.logger.debug(f"连续成功{success_count}次，恢复到初始批次大小: {config.initial_size}")
                        batch_size = config.initial_size
                        success_count = 0

                    if i < len(items):
                        await asyncio.sleep(config.retry_delay)

            except Exception as e:
                self.logger.error(f"批处理失败: {str(e)}")
                if reduce_batch_on_error and batch_size > config.min_size:
                    batch_size = max(batch_size // 2, config.min_size)
                    success_count = 0
                    self.logger.debug(f"出错后减小批次大小到: {batch_size}")
                    continue
                else:
                    if error_handler:
                        yield error_handler(batch)
                    i += len(batch)

    async def translate_sentences(
        self,
        sentences: List,
        batch_size: int = 100,
        target_language: str = "zh"
    ) -> AsyncGenerator[List, None]:
        """
        批量翻译处理，将每个句子的原始文本翻译后赋值给 sentence.trans_text。
        """
        if not sentences:
            self.logger.warning("收到空的句子列表")
            return

        config = BatchConfig(initial_size=batch_size)

        async def process_batch(batch: List) -> Optional[List]:
            texts = {str(j): s.raw_text for j, s in enumerate(batch)}
            self.logger.debug(f"翻译批次: {len(texts)}条文本")
            translated = await self.translate(texts, target_language)
            if "output" not in translated:
                self.logger.error("翻译结果中缺少 output 字段")
                return None
            translated_texts = translated["output"]
            if len(translated_texts) == len(texts):
                for j, sentence in enumerate(batch):
                    sentence.trans_text = translated_texts[str(j)]
                return batch
            return None

        def handle_error(batch: List) -> List:
            for sentence in batch:
                sentence.trans_text = sentence.raw_text
            return batch

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
        batch_size: int = 4,
        target_speed: float = 1.1  # 目标语速设定为 max_speed，此处默认值 1.1
    ) -> AsyncGenerator[List, None]:
        """
        批量精简处理，对于语速过快的句子（由 DurationAligner 筛选），
        根据原文本与各精简版本的长度比较，计算理想文本长度后选择最佳候选版本。

        理想文本长度计算公式：
            ideal_length = len(old_text) * (target_speed / s.speed)
        当 target_speed = max_speed 时，可确保精简后的文本达到预期的语速要求。
        """
        if not sentences:
            self.logger.warning("收到空的句子列表")
            return

        config = BatchConfig(initial_size=batch_size, min_size=1, required_successes=2)

        async def process_batch(batch: List) -> Optional[List]:
            texts = {str(i): s.trans_text for i, s in enumerate(batch)}
            self.logger.debug(f"简化批次: {len(texts)}条文本")
            batch_result = await self.simplify(texts)
            
            if "thinking" not in batch_result or not any(key in batch_result for key in ["slight", "moderate", "extreme"]):
                self.logger.error("简化结果格式不正确，缺少必要字段")
                return None
                
            for i, s in enumerate(batch):
                old_text = s.trans_text
                str_i = str(i)
                
                # 确保每个句子的简化结果都存在
                if not any(str_i in batch_result.get(key, {}) for key in ["slight", "moderate", "extreme"]):
                    self.logger.error(f"句子 {i} 的简化结果不完整")
                    continue

                # 根据原文本长度和当前语速计算理想文本长度
                ideal_length = len(old_text) * (target_speed / s.speed) if s.speed > 0 else len(old_text)
                
                acceptable_candidates = {}
                non_acceptable_candidates = {}
                
                for key in ["slight", "moderate", "extreme"]:
                    if key in batch_result and str_i in batch_result[key]:
                        candidate_text = batch_result[key][str_i]
                        if candidate_text:
                            candidate_length = len(candidate_text)
                            if candidate_length <= ideal_length:
                                acceptable_candidates[key] = candidate_text
                            else:
                                non_acceptable_candidates[key] = candidate_text
                
                if acceptable_candidates:
                    # 在满足候选长度不超过理想长度的版本中，选择文本最长的版本
                    chosen_key, chosen_text = max(acceptable_candidates.items(), key=lambda item: len(item[1]))
                elif non_acceptable_candidates:
                    # 若所有候选均超过理想长度，则选择与理想长度差值最小的版本
                    chosen_key, chosen_text = min(non_acceptable_candidates.items(), key=lambda item: abs(len(item[1]) - ideal_length))
                else:
                    chosen_text = old_text

                s.trans_text = chosen_text
                self.logger.info(
                    f"精简: {old_text} -> {chosen_text} (理想长度: {ideal_length}, s.speed: {s.speed})"
                )
            return batch

        def handle_error(batch: List) -> List:
            # 出错时原样返回
            return batch

        async for batch_result in self._process_batch(
            sentences,
            process_batch,
            config,
            error_handler=handle_error,
            reduce_batch_on_error=False
        ):
            yield batch_result
