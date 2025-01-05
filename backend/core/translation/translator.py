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
            # 打印json_string的内容和格式
            logger.info(f"翻译结果: {json_string}，类型: {type(json_string)}")
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

        i = 0
        initial_size = batch_size  # 保存初始批次大小
        success_count = 0  # 连续成功计数
        required_successes = 2  # 需要连续成功几次才恢复大批次

        while i < len(sentences):
            # 在连续成功足够次数后才尝试恢复到初始批次大小
            if batch_size < initial_size and success_count >= required_successes:
                logger.info(f"连续成功{success_count}次，恢复到初始批次大小: {initial_size}")
                batch_size = initial_size
                success_count = 0

            success = False
            pos = i  # 保存当前位置

            while not success and batch_size >= 1:
                try:
                    # 获取当前批次的句子
                    batch = sentences[pos:pos+batch_size]
                    if not batch:
                        break

                    texts = {str(j): s.raw_text for j, s in enumerate(batch)}
                    logger.info(f"翻译批次: {len(texts)}条文本, 大小: {batch_size}, 位置: {pos}")

                    # 翻译并检查结果完整性
                    translated = await self.translate(texts, target_language)
                    
                    if len(translated) == len(texts):
                        success = True
                        success_count += 1
                        logger.info(f"翻译成功: {len(batch)}条文本, 连续成功: {success_count}次")
                        
                        # 处理翻译结果
                        results = []
                        for j, sentence in enumerate(batch):
                            sentence.trans_text = translated[str(j)]
                            results.append(sentence)

                        yield results
                        i += len(batch)
                    else:
                        # 结果不完整，减小批次大小重试
                        batch_size = max(batch_size // 2, 1)
                        success_count = 0
                        logger.warning(f"翻译不完整 (输入: {len(texts)}, 输出: {len(translated)}), 减小到: {batch_size}")
                        continue

                    # 避免API限流
                    if i < len(sentences):
                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"翻译失败: {str(e)}")
                    if batch_size > 1:
                        batch_size = max(batch_size // 2, 1)
                        success_count = 0
                        logger.info(f"出错后减小批次大小到: {batch_size}")
                        continue
                    else:
                        # 单句翻译失败，使用原文
                        results = []
                        for sentence in batch:
                            sentence.trans_text = sentence.raw_text
                            results.append(sentence)
                        yield results
                        i += 1 