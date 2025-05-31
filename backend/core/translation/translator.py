import asyncio
import logging
from typing import Dict, List, AsyncGenerator, Optional, TypeVar, Any
import ray
from ray import serve
from .prompt import (
    TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_USER_PROMPT,
    LANGUAGE_MAP
)
from .deepseek_client import DeepSeekClient
from .gemini_client import GeminiClient
from .grok_client import GrokClient as XaiGrokClient
from .groq_client import GroqClient as GroqSDKClient
from config import Config
logger = logging.getLogger(__name__)

T = TypeVar('T')

@serve.deployment(
    name="translator",
    ray_actor_options={"num_cpus": 1},
    num_replicas=2
)
class Translator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        translation_model = (self.config.TRANSLATION_MODEL or "deepseek").strip().lower()
        
        if translation_model == "deepseek":
            self.client = DeepSeekClient(api_key=self.config.DEEPSEEK_API_KEY)
        elif translation_model == "gemini":
            self.client = GeminiClient(api_key=self.config.GEMINI_API_KEY)
        elif translation_model == "grok":
            self.client = XaiGrokClient(api_key=self.config.XAI_API_KEY)
        elif translation_model == "groq":
            self.client = GroqSDKClient(api_key=self.config.GROQ_API_KEY)
        else:
            raise ValueError(f"不支持的翻译模型：{translation_model}")
        self.logger.info(f"初始化翻译Actor，使用模型: {translation_model}")

    async def _invoke_client(self, system_prompt: str, user_prompt: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """统一调用模型接口"""
        try:
            result = await self.client.translate(system_prompt=system_prompt, user_prompt=user_prompt)
            return result if result else default
        except Exception as e:
            self.logger.error(f"模型调用失败: {e}")
            raise

    async def translate(self, texts: Dict[str, str], target_language: str = "zh") -> Dict[str, str]:
        """翻译文本"""
        system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
            target_language=LANGUAGE_MAP.get(target_language, target_language)
        )
        user_prompt = TRANSLATION_USER_PROMPT.format(
            target_language=LANGUAGE_MAP.get(target_language, target_language),
            json_content=texts
        )
        return await self._invoke_client(system_prompt, user_prompt, {"output": {}})

    async def translate_sentences(
        self,
        sentences: List,
        batch_size: int = 50,
        target_language: str = "zh"
    ) -> AsyncGenerator[List, None]:
        """翻译句子，返回异步生成器"""
        if not sentences:
            self.logger.warning("收到空的句子列表")
            return

        # 确保 batch_size 是整数
        try:
            batch_size = int(batch_size)
        except (ValueError, TypeError):
            self.logger.warning(f"无效的 batch_size 类型 {type(batch_size)}: {batch_size}，将使用默认值 50")
            batch_size = 50

        self.logger.debug(f"翻译处理 {len(sentences)} 个句子，初始批次大小: {batch_size}")

        # 动态批次处理参数
        current_batch_size = batch_size
        min_batch_size = 1
        success_count = 0
        required_successes = 2

        i = 0
        while i < len(sentences):
            batch = sentences[i:i + current_batch_size]
            if not batch:
                break

            try:
                # 构建待翻译文本
                texts = {str(j): s.raw_text for j, s in enumerate(batch)}
                self.logger.debug(f"翻译批次: {len(texts)}条文本")
                
                # 调用翻译API
                translated = await self.translate(texts, target_language)
                
                if "output" not in translated:
                    self.logger.error("翻译结果中缺少 output 字段")
                    raise Exception("翻译结果格式错误")
                    
                translated_texts = translated["output"]
                if len(translated_texts) != len(texts):
                    self.logger.error("翻译结果数量不匹配")
                    raise Exception("翻译结果数量不匹配")

                # 设置翻译结果
                for j, sentence in enumerate(batch):
                    sentence.trans_text = translated_texts.get(str(j), sentence.raw_text)

                # 翻译成功
                success_count += 1
                yield batch
                i += len(batch)
                
                # 连续成功后恢复初始批次大小
                if current_batch_size < batch_size and success_count >= required_successes:
                    self.logger.debug(f"连续成功{success_count}次，恢复到初始批次大小: {batch_size}")
                    current_batch_size = batch_size
                    success_count = 0

                # 批次间延迟
                if i < len(sentences):
                    await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"翻译批次失败: {e}")
                
                # 出错后减小批次大小
                if current_batch_size > min_batch_size:
                    current_batch_size = max(current_batch_size // 2, min_batch_size)
                    success_count = 0
                    self.logger.debug(f"出错后减小批次大小到: {current_batch_size}")
                    continue
                else:
                    # 批次已经是最小，使用原文作为翻译结果
                    self.logger.warning(f"最小批次仍失败，使用原文: {len(batch)}个句子")
                    for sentence in batch:
                        sentence.trans_text = sentence.raw_text
                    yield batch
                    i += len(batch)