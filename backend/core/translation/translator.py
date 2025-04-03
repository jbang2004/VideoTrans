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
    ray_actor_options={"num_cpus": 1},
    num_replicas=2  # 增加副本以提高并发处理能力
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
            result = await self.client.translate(system_prompt=system_prompt, user_prompt=user_prompt)
            # 确保模型返回的是字典
            return result if result else {"output": {}}
        except Exception as e:
            self.logger.error(f"翻译失败: {str(e)}")
            raise
        finally:
            # 显式删除大型提示文本，帮助内存回收
            if 'system_prompt' in locals(): del system_prompt
            if 'user_prompt' in locals(): del user_prompt

    async def simplify(self, texts: Dict[str, str]) -> Dict[str, str]:
        """简化文本"""
        try:
            system_prompt = SIMPLIFICATION_SYSTEM_PROMPT
            user_prompt = SIMPLIFICATION_USER_PROMPT.format(json_content=texts)
            result = await self.client.translate(system_prompt=system_prompt, user_prompt=user_prompt)
            return result if result else {}
        except Exception as e:
            self.logger.error(f"简化失败: {str(e)}")
            raise
        finally:
            # 显式删除大型提示文本，帮助内存回收
            if 'system_prompt' in locals(): del system_prompt
            if 'user_prompt' in locals(): del user_prompt

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
            batch = None
            results = None
            
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
                    if error_handler and batch:
                        yield error_handler(batch)
                    if batch:
                        i += len(batch)
            finally:
                # 显式删除批处理结果以帮助内存回收
                if batch is not None: del batch
                if results is not None: del results

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
            texts = None
            translated_texts = None
            
            try:
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
            except Exception as e:
                self.logger.error(f"处理翻译批次失败: {e}")
                raise
            finally:
                # 显式删除临时变量以帮助内存回收
                if texts is not None: del texts
                if translated_texts is not None: del translated_texts

        def handle_error(batch: List) -> List:
            # 错误处理：使用原始文本作为翻译
            for sentence in batch:
                sentence.trans_text = sentence.raw_text
            return batch

        try:
            async for batch_result in self._process_batch(
                sentences,
                process_batch,
                config,
                error_handler=handle_error,
                reduce_batch_on_error=True
            ):
                yield batch_result
        except Exception as e:
            self.logger.error(f"翻译句子生成器发生错误: {e}")
            raise

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
            texts = None
            batch_result = None
            
            try:
                texts = {str(i): s.trans_text for i, s in enumerate(batch)}
                self.logger.debug(f"简化批次: {len(texts)}条文本")
                batch_result = await self.simplify(texts)
                
                if "thinking" not in batch_result or not any(key in batch_result for key in ["minimal", "slight", "moderate", "significant", "extreme"]):
                    self.logger.error("简化结果格式不正确，缺少必要字段")
                    return None
                    
                for i, s in enumerate(batch):
                    old_text = s.trans_text
                    str_i = str(i)
                    
                    if not any(str_i in batch_result.get(key, {}) for key in ["minimal", "slight", "moderate", "significant", "extreme"]):
                        self.logger.error(f"句子 {i} 的简化结果不完整")
                        continue

                    ideal_length = len(old_text) * (target_speed / s.speed) if s.speed > 0 else len(old_text)
                    
                    # 存储所有可接受和不可接受的候选文本
                    acceptable_candidates = {}
                    non_acceptable_candidates = {}
                    
                    # 按精简程度检查候选文本
                    simplification_levels = ["minimal", "slight", "moderate", "significant", "extreme"]
                    for key in simplification_levels:
                        if key in batch_result and str_i in batch_result[key]:
                            candidate_text = batch_result[key][str_i]
                            if candidate_text:
                                candidate_length = len(candidate_text)
                                if candidate_length <= ideal_length:
                                    acceptable_candidates[key] = candidate_text
                                else:
                                    non_acceptable_candidates[key] = candidate_text
                    
                    # 如果有可接受的候选文本（长度小于等于理想长度），选择最接近理想长度的（最长的可接受文本）
                    if acceptable_candidates:
                        best_candidate = None
                        min_diff = float('inf')
                        
                        for key, text in acceptable_candidates.items():
                            diff = abs(len(text) - ideal_length)
                            if diff < min_diff:
                                min_diff = diff
                                best_candidate = (key, text)
                        
                        chosen_key, chosen_text = best_candidate
                        
                    # 如果没有可接受的候选文本，选择最接近理想长度的不可接受文本
                    elif non_acceptable_candidates:
                        best_candidate = None
                        min_diff = float('inf')
                        
                        for key, text in non_acceptable_candidates.items():
                            diff = abs(len(text) - ideal_length)
                            if diff < min_diff:
                                min_diff = diff
                                best_candidate = (key, text)
                        
                        chosen_key, chosen_text = best_candidate
                        
                    # 如果没有可用的候选文本，保持原文
                    else:
                        chosen_key = "原文"
                        chosen_text = old_text

                    s.trans_text = chosen_text
                    self.logger.info(
                        f"精简[{chosen_key}]: {old_text} -> {chosen_text} (理想长度: {ideal_length}, 实际长度: {len(chosen_text)}, s.speed: {s.speed})"
                    )
                    
                    # 释放不再需要的候选文本字典
                    del acceptable_candidates
                    del non_acceptable_candidates
                        
                return batch
            except Exception as e:
                self.logger.error(f"处理简化批次失败: {e}")
                raise
            finally:
                # 显式删除临时变量以帮助内存回收
                if texts is not None: del texts
                if batch_result is not None: del batch_result

        def handle_error(batch: List) -> List:
            return batch

        try:
            async for batch_result in self._process_batch(
                sentences,
                process_batch,
                config,
                error_handler=handle_error,
                reduce_batch_on_error=False
            ):
                yield batch_result
        except Exception as e:
            self.logger.error(f"简化句子生成器发生错误: {e}")
            raise

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
                
                texts = None
                translations = None
                
                try:
                    # 翻译一批句子
                    texts = {str(j): s.raw_text for j, s in enumerate(batch)}
                    
                    # 使用翻译方法翻译
                    translated = await self.translate(texts, target_language)
                    
                    # 提取翻译结果
                    if "output" in translated:
                        translations = translated["output"]
                        
                        # 更新翻译结果
                        for j, s in enumerate(batch):
                            s.trans_text = translations.get(str(j), s.raw_text)
                    else:
                        self.logger.error(f"翻译结果缺少输出字段，批次 {i+1}/{batch_count}")
                        # 回退使用原始文本
                        for s in batch:
                            s.trans_text = s.raw_text
                        
                    # 添加到结果
                    translated_sentences.extend(batch)
                    
                    # 输出进度日志
                    self.logger.debug(f"已翻译 {end_idx}/{len(sentences)} 个句子")
                    
                except Exception as e:
                    self.logger.error(f"批次翻译失败: {str(e)}, 批次 {i+1}/{batch_count}")
                    # 对于失败的批次，使用原始文本作为翻译
                    for s in batch:
                        s.trans_text = s.raw_text
                    translated_sentences.extend(batch)
                finally:
                    # 释放大型临时变量
                    if texts is not None: del texts
                    if translations is not None: del translations
                    # 让Python有更多机会进行垃圾回收
                    if i % 5 == 0:  # 每5个批次
                        await asyncio.sleep(0.1)  # 允许事件循环运行并有机会进行垃圾回收
            
            self.logger.info(f"翻译完成，共 {len(translated_sentences)} 个句子")
            return translated_sentences
            
        except Exception as e:
            self.logger.error(f"翻译句子失败: {str(e)}", exc_info=True)
            # 确保即使出错也返回原始句子（使用原文作为翻译）
            for s in sentences:
                if not hasattr(s, 'trans_text') or s.trans_text is None:
                    s.trans_text = s.raw_text
            return sentences