import asyncio
import logging
from typing import Dict, List, AsyncGenerator, Protocol
from json_repair import loads
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

class Translator:
    def __init__(self, translation_client: TranslationClient):
        self.translation_client = translation_client
        self.logger = logging.getLogger(__name__)

    async def translate(self, texts: Dict[str, str], target_language: str = "zh") -> Dict[str, str]:
        """执行翻译并处理结果"""
        try:
            system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
                target_language=LANGUAGE_MAP.get(target_language, target_language)
            )
            user_prompt = TRANSLATION_USER_PROMPT.format(
                target_language=LANGUAGE_MAP.get(target_language, target_language),
                json_content=texts
            )
            result = await self.translation_client.translate(
                texts=texts,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            return result
        except Exception as e:
            self.logger.error(f"翻译失败: {str(e)}")
            raise

    async def simplify(self, texts: Dict[str, str], batch_size: int = 4) -> Dict[str, str]:
        """执行文本简化并处理结果，支持批量处理和错误恢复"""
        if not texts:
            return {}

        result = {}
        keys = list(texts.keys())
        i = 0
        success_count = 0
        current_batch_size = batch_size

        while i < len(keys):
            try:
                batch_keys = keys[i:i+current_batch_size]
                batch_texts = {k: texts[k] for k in batch_keys}
                
                self.logger.debug(f"简化批次: {len(batch_texts)}条文本, 大小: {current_batch_size}, 位置: {i}")

                system_prompt = SIMPLIFICATION_SYSTEM_PROMPT
                user_prompt = SIMPLIFICATION_USER_PROMPT.format(json_content=batch_texts)
                
                batch_result = await self.translation_client.translate(
                    texts=batch_texts,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                
                if len(batch_result) == len(batch_texts):
                    success_count += 1
                    result.update(batch_result)
                    i += len(batch_texts)
                    self.logger.debug(f"简化成功: {len(batch_texts)}条文本, 连续成功: {success_count}次")
                    
                    if current_batch_size < batch_size and success_count >= 2:
                        self.logger.debug(f"连续成功{success_count}次，恢复到初始批次大小: {batch_size}")
                        current_batch_size = batch_size
                        success_count = 0
                else:
                    raise ValueError(f"简化结果不完整 (输入: {len(batch_texts)}, 内容: {batch_texts}, 输出: {len(batch_result)}, 内容: {batch_result})")

                if i < len(keys):
                    await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"简化失败: {str(e)}")
                if current_batch_size > 1:
                    current_batch_size = max(current_batch_size // 2, 1)
                    success_count = 0
                    self.logger.debug(f"出错后减小批次大小到: {current_batch_size}")
                    continue
                else:
                    for k in batch_keys:
                        result[k] = texts[k]
                    i += len(batch_keys)

        return result

    async def translate_sentences(
        self,
        sentences: List,
        batch_size: int = 100,
        target_language: str = "zh"
    ) -> AsyncGenerator[List, None]:
        """批量翻译处理"""
        if not sentences:
            self.logger.warning("收到空的句子列表")
            return

        i = 0
        initial_size = batch_size
        success_count = 0
        required_successes = 2

        while i < len(sentences):
            if batch_size < initial_size and success_count >= required_successes:
                self.logger.debug(f"连续成功{success_count}次，恢复到初始批次大小: {initial_size}")
                batch_size = initial_size
                success_count = 0

            success = False
            pos = i

            while not success and batch_size >= 1:
                try:
                    batch = sentences[pos:pos+batch_size]
                    if not batch:
                        break

                    texts = {str(j): s.raw_text for j, s in enumerate(batch)}
                    self.logger.debug(f"翻译批次: {len(texts)}条文本, 大小: {batch_size}, 位置: {pos}")

                    translated = await self.translate(texts, target_language)
                    
                    if len(translated) == len(texts):
                        success = True
                        success_count += 1
                        self.logger.info(f"翻译成功: {len(batch)}条文本, 连续成功: {success_count}次")

                        self.logger.debug(f"原文: {texts}，翻译结果: {translated}")
                        results = []
                        for j, sentence in enumerate(batch):
                            sentence.trans_text = translated[str(j)]
                            results.append(sentence)

                        yield results
                        i += len(batch)
                    else:
                        batch_size = max(batch_size // 2, 1)
                        success_count = 0
                        self.logger.warning(f"翻译不完整 (输入: {len(texts)}, 输出: {len(translated)}), 减小到: {batch_size}")
                        self.logger.debug(f"原文: {texts}，翻译结果: {translated}")
                        continue

                    if i < len(sentences):
                        await asyncio.sleep(0.1)

                except Exception as e:
                    self.logger.error(f"翻译失败: {str(e)}")
                    if batch_size > 1:
                        batch_size = max(batch_size // 2, 1)
                        success_count = 0
                        self.logger.debug(f"出错后减小批次大小到: {batch_size}")
                        continue
                    else:
                        results = []
                        for sentence in batch:
                            sentence.trans_text = sentence.raw_text
                            results.append(sentence)
                        yield results
                        i += 1
