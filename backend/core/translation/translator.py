import asyncio
import logging
from typing import Dict, List, AsyncGenerator, Protocol
from json_repair import loads

logger = logging.getLogger(__name__)

class TranslationClient(Protocol):
    async def translate(self, texts: Dict[str, str], target_language: str = "zh") -> str:
        ...

class Translator:
    def __init__(self, translation_client: TranslationClient):
        """初始化翻译器
        Args:
            translation_client: 翻译客户端实例，例如 GLM4Client 或 GeminiClient
        """
        self.translation_client = translation_client

    async def translate(self, texts: Dict[str, str], target_language: str = "zh") -> Dict[str, str]:
        """执行单批次翻译并处理 JSON 结果"""
        try:
            json_string = await self.translation_client.translate(texts, target_language)
            return loads(json_string)
        except Exception as e:
            logger.error(f"翻译过程中发生错误: {str(e)}", exc_info=True)
            raise

    async def translate_sentences(
        self, 
        sentences: List, 
        batch_size: int = 100,
        target_language: str = "zh"
    ) -> AsyncGenerator[List, None]:
        """批量翻译处理
        Args:
            sentences: 待翻译的句子列表
            batch_size: 批处理大小，默认100
            target_language: 目标语言代码 (zh/en/ja/ko)
        Yields:
            每个批次翻译后的句子列表
        """
        if not sentences:
            logger.warning("收到空的句子列表")
            return

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]

            texts_to_translate = {str(j): s.raw_text for j, s in enumerate(batch_sentences)}
            logger.info(f"翻译批次: {texts_to_translate}")

            try:
                # 使用 translate 方法获取解析后的字典
                translated_texts = await self.translate(texts_to_translate, target_language)
                logger.info(f"批次 {i//batch_size + 1}/{(len(sentences)-1)//batch_size + 1} 翻译完成")

                # 批量更新翻译结果
                batch_results = []
                for j, sentence in enumerate(batch_sentences):
                    key = str(j)
                    if key in translated_texts:
                        sentence.trans_text = translated_texts[key]
                    else:
                        logger.warning(f"批次 {i//batch_size + 1} 中索引 {j} 的翻译结果缺失")
                        sentence.trans_text = sentence.raw_text
                        logger.info(f"结果缺失后的翻译文本: {sentence.trans_text}")
                    batch_results.append(sentence)

                # 立即产出这个批次的结果
                yield batch_results

                # 避免API限流
                if i + batch_size < len(sentences):
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"translate_sentences: 批次 {i//batch_size + 1} 翻译失败: {str(e)}")
                # 发生错误时返回原文
                error_batch = []
                for sentence in batch_sentences:
                    sentence.trans_text = sentence.raw_text
                    error_batch.append(sentence)
                yield error_batch 