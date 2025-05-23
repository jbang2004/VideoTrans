import asyncio
import logging
from typing import Dict, List, AsyncGenerator, Optional, TypeVar, Any
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
from .grok_client import GrokClient as XaiGrokClient
from .groq_client import GroqClient as GroqSDKClient
from config import Config
from core.supabase_client import SupabaseClient
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
    # 简化等级常量
    SIMPLIFICATION_LEVELS = ["minimal", "slight", "moderate", "significant", "extreme"]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.supabase_client = SupabaseClient(config=self.config)
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

    # 私有方法：统一调用模型接口
    async def _invoke_client(self, system_prompt: str, user_prompt: str, default: Dict[str, Any]) -> Dict[str, Any]:
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

    async def simplify(self, texts: Dict[str, str]) -> Dict[str, str]:
        """简化文本"""
        system_prompt = SIMPLIFICATION_SYSTEM_PROMPT
        user_prompt = SIMPLIFICATION_USER_PROMPT.format(json_content=texts)
        return await self._invoke_client(system_prompt, user_prompt, {})

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

    async def translate_sentences(
        self,
        task_id: str,
        target_language: str,
        batch_size: int = 50
    ) -> AsyncGenerator[List, None]:
        """翻译句子，返回异步生成器"""
        sentences = await self.supabase_client.get_sentences(task_id, as_objects=True)
        if not sentences:
            self.logger.warning(f"[{task_id}] 翻译：数据库中没有检索到句子")
            return
        self.logger.info(f"[{task_id}] 翻译：获取到 {len(sentences)} 个句子")

        # 使用传入的 target_language 并更新任务状态
        try:
            asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'translating', 'target_language': target_language}))
            self.logger.info(f"[{task_id}] 翻译：异步更新任务状态为 translating，目标语言: {target_language}")
        except Exception as e:
            self.logger.error(f"[{task_id}] 翻译：更新状态或存储 target_language 时出错: {e}", exc_info=True)
            return

        # 确保 batch_size 是整数
        try:
            batch_size = int(batch_size)
        except (ValueError, TypeError):
            self.logger.warning(f"无效的 batch_size 类型 {type(batch_size)}: {batch_size}，将使用默认值 50")
            batch_size = 50 # 使用默认值

        config = BatchConfig(initial_size=batch_size)

        async def process_batch(batch: List) -> Optional[List]:
            texts = None
            translated_texts = None
            
            try:
                texts = {str(j): s.raw_text for j, s in enumerate(batch)}
                self.logger.debug(f"翻译批次: {len(texts)}条文本")
                translated = await self.translate(texts, target_language)

                # 继续处理翻译结果
                if "output" not in translated:
                    self.logger.error(f"[{task_id}] 翻译结果中缺少 output 字段. 原始响应: {translated}")
                    # 抛出异常以触发 _process_batch 中的重试逻辑
                    raise ValueError(f"[{task_id}] 翻译结果中缺少 output 字段")
                    
                translated_texts = translated["output"]
                if len(translated_texts) == len(texts):
                    # 继续更新数据库和句子对象
                    for j, sentence in enumerate(batch):
                        sentence.trans_text = translated_texts[str(j)]
                        await self.supabase_client.update_sentence_translation(
                            task_id,
                            sentence.sentence_id,
                            sentence.trans_text
                        )
                    return batch
                self.logger.error(f"[{task_id}] 翻译返回数量与输入不匹配。输入: {len(texts)}, 输出: {len(translated_texts)}. 原始响应: {translated}")
                # 抛出异常以触发 _process_batch 中的重试逻辑
                raise ValueError(f"[{task_id}] 翻译返回数量与输入不匹配。输入: {len(texts)}, 输出: {len(translated_texts)}")
            except Exception as e:
                self.logger.error(f"[{task_id}] 处理翻译批次失败: {e}")
                raise

        def handle_error(batch: List) -> List:
            for sentence in batch:
                sentence.trans_text = sentence.raw_text
                asyncio.create_task(self.supabase_client.update_sentence_translation(
                    task_id, 
                    sentence.sentence_id, 
                    sentence.raw_text
                ))
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
            self.logger.error(f"[{task_id}] 翻译句子生成器发生错误: {e}")
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

        # 确保 batch_size 是整数
        try:
            batch_size = int(batch_size)
        except (ValueError, TypeError):
            self.logger.warning(f"无效的 batch_size 类型 {type(batch_size)}: {batch_size}，将使用默认值 4")
            batch_size = 4 # 使用默认值

        config = BatchConfig(initial_size=batch_size, min_size=1, required_successes=2)

        async def process_batch(batch: List) -> Optional[List]:
            texts = {str(i): s.trans_text for i, s in enumerate(batch)}
            self.logger.debug(f"简化批次: {len(texts)}条文本")
            batch_result = await self.simplify(texts)
            
            if not any(key in batch_result for key in self.SIMPLIFICATION_LEVELS):
                self.logger.error("简化结果格式不正确，缺少必要字段")
                return None
                
            for i, s in enumerate(batch):
                old_text = s.trans_text
                str_i = str(i)
                
                if not any(str_i in batch_result.get(key, {}) for key in self.SIMPLIFICATION_LEVELS):
                    self.logger.error(f"句子 {i} 的简化结果不完整")
                    continue

                ideal_length = len(old_text) * (target_speed / s.speed) if s.speed > 0 else len(old_text)
                
                # 存储所有可接受和不可接受的候选文本
                acceptable_candidates = {}
                non_acceptable_candidates = {}
                
                # 按精简程度检查候选文本
                for key in self.SIMPLIFICATION_LEVELS:
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
                    # 在可接受的候选中选择最长的那个（因此最接近 ideal_length）
                    chosen_key, chosen_text = max(acceptable_candidates.items(), key=lambda item: len(item[1]))
                elif non_acceptable_candidates:
                    # 在不可接受的候选中选择最短的那个（因此最接近 ideal_length）
                    chosen_key, chosen_text = min(non_acceptable_candidates.items(), key=lambda item: len(item[1]))
                else:
                    chosen_key = "原文"
                    chosen_text = old_text

                s.trans_text = chosen_text
                self.logger.info(
                    f"精简[{chosen_key}]: {old_text} -> {chosen_text} (理想长度: {ideal_length}, 实际长度: {len(chosen_text)}, s.speed: {s.speed})"
                )
                
            return batch

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