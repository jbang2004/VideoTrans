import asyncio
import logging
from typing import Dict, List, AsyncGenerator, Optional, TypeVar, Generic
from dataclasses import dataclass
import ray
from ray import serve
from .prompt import (
    TRANSLATION_SYSTEM_PROMPT,
    TRANSLATION_USER_PROMPT,
    SIMPLIFICATION_SYSTEM_PROMPT,
    SIMPLIFICATION_USER_PROMPT,
    LANGUAGE_MAP
)
from .deepseek_client import DeepSeekClient
from .gemini_client import GeminiClient
from config import Config
logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    """批处理配置"""
    initial_size: int = 100
    min_size: int = 1
    required_successes: int = 2
    retry_delay: float = 0.1

T = TypeVar('T')

@serve.deployment(
    name="translator",
    num_replicas=1,
    ray_actor_options={"num_cpus": Config().TRANSLATOR_ACTOR_NUM_CPUS},
    # autoscaling_config={
    #     "min_replicas": 1,
    #     "max_replicas": 5,
    #     "target_num_ongoing_requests_per_replica": 10
    # }
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
        else:
            raise ValueError(f"不支持的翻译模型：{translation_model}")
        self.logger.info(f"初始化翻译Actor，使用模型: {translation_model}")

    async def translate(self, texts: Dict[str, str], target_language: str = "zh") -> Dict[str, str]:
        """翻译文本"""
        try:
            system_prompt = TRANSLATION_SYSTEM_PROMPT.format(
                target_language=LANGUAGE_MAP.get(target_language, target_language)
            )
            user_prompt = TRANSLATION_USER_PROMPT.format(
                target_language=LANGUAGE_MAP.get(target_language, target_language),
                json_content=texts
            )
            return await self.client.translate(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as e:
            self.logger.error(f"翻译失败: {str(e)}")
            raise

    async def simplify(self, texts: Dict[str, str]) -> Dict[str, str]:
        """简化文本"""
        try:
            system_prompt = SIMPLIFICATION_SYSTEM_PROMPT
            user_prompt = SIMPLIFICATION_USER_PROMPT.format(json_content=texts)
            return await self.client.translate(system_prompt=system_prompt, user_prompt=user_prompt)
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
        """动态批处理核心逻辑"""
        if not items:
            return

        i = 0
        batch_size = config.initial_size
        success_count = 0

        while i < len(items):
            try:
                batch = items[i:i + batch_size]
                if not batch:
                    break

                results = await process_func(batch)
                if results:
                    success_count += 1
                    yield results
                    i += len(batch)
                    # 如果出错后批次变小，连续成功后恢复初始批次大小
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
        """翻译句子，返回异步生成器"""
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
        target_speed: float = 1.1
    ) -> AsyncGenerator[List, None]:
        """简化句子，返回异步生成器"""
        if not sentences:
            self.logger.warning("收到空的句子列表")
            return

        self.logger.debug(f"处理 {len(sentences)} 个句子")

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
                
                if not any(str_i in batch_result.get(key, {}) for key in ["slight", "moderate", "extreme"]):
                    self.logger.error(f"句子 {i} 的简化结果不完整")
                    continue

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
                    chosen_key, chosen_text = max(acceptable_candidates.items(), key=lambda item: len(item[1]))
                elif non_acceptable_candidates:
                    chosen_key, chosen_text = min(non_acceptable_candidates.items(), key=lambda item: abs(len(item[1]) - ideal_length))
                else:
                    chosen_text = old_text

                s.trans_text = chosen_text
                self.logger.info(
                    f"精简: {old_text} -> {chosen_text} (理想长度: {ideal_length}, s.speed: {s.speed})"
                )
            return batch

        def handle_error(batch: List) -> List:
            return batch

        async for batch_result in self._process_batch(
            sentences,
            process_batch,
            config,
            error_handler=handle_error,
            reduce_batch_on_error=False
        ):
            yield batch_result

    async def translate_sentences_simple(self, sentences, target_language, batch_size=5):
        """
        一次性翻译多个句子，返回翻译结果的列表
        避免使用流式处理导致的内存管理问题
        
        Args:
            sentences: 句子列表
            target_language: 目标语言代码
            batch_size: 批处理大小
            
        Returns:
            翻译完成的句子列表
        """
        if not sentences:
            self.logger.warning("收到空的句子列表，跳过翻译")
            return []
            
        self.logger.info(f"一次性翻译 {len(sentences)} 个句子，目标语言: {target_language}")
        
        # 处理小批次，避免一次处理太多句子
        translated_sentences = []
        batch_count = (len(sentences) + batch_size - 1) // batch_size
        
        try:
            for i in range(batch_count):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(sentences))
                batch = sentences[start_idx:end_idx]
                
                # 翻译一批句子
                batch_texts = [s.raw_text for s in batch]
                
                # 使用翻译方法翻译
                translations = await self.translate(batch_texts, target_language)
                
                # 更新翻译结果
                for j, translation in enumerate(translations):
                    batch[j].trans_text = translation
                    
                # 添加到结果
                translated_sentences.extend(batch)
                
                # 清理内存
                import gc
                gc.collect()
                
                # 输出进度日志
                self.logger.debug(f"已翻译 {end_idx}/{len(sentences)} 个句子")
                
            self.logger.info(f"翻译完成，共 {len(translated_sentences)} 个句子")
            return translated_sentences
            
        except Exception as e:
            self.logger.error(f"翻译句子失败: {str(e)}", exc_info=True)
            raise