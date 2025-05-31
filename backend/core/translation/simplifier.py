import asyncio
import logging
from typing import Dict, List, Optional, TypeVar, Any
import ray
from ray import serve
from .prompt import (
    SIMPLIFICATION_SYSTEM_PROMPT,
    SIMPLIFICATION_USER_PROMPT
)
from .deepseek_client import DeepSeekClient
from .gemini_client import GeminiClient
from .grok_client import GrokClient as XaiGrokClient
from .groq_client import GroqClient as GroqSDKClient
from config import Config

logger = logging.getLogger(__name__)

T = TypeVar('T')

@serve.deployment(
    name="simplifier",
    ray_actor_options={"num_cpus": 1},
    num_replicas=1,
    logging_config={"log_level": "INFO"}
)
class Simplifier:
    # 简化等级常量
    SIMPLIFICATION_LEVELS = ["minimal", "slight", "moderate", "significant", "extreme"]

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        simplification_model = (self.config.TRANSLATION_MODEL or "deepseek").strip().lower()
        
        if simplification_model == "deepseek":
            self.client = DeepSeekClient(api_key=self.config.DEEPSEEK_API_KEY)
        elif simplification_model == "gemini":
            self.client = GeminiClient(api_key=self.config.GEMINI_API_KEY)
        elif simplification_model == "grok":
            self.client = XaiGrokClient(api_key=self.config.XAI_API_KEY)
        elif simplification_model == "groq":
            self.client = GroqSDKClient(api_key=self.config.GROQ_API_KEY)
        else:
            raise ValueError(f"不支持的简化模型：{simplification_model}")
        self.logger.info(f"初始化简化Actor，使用模型: {simplification_model}")

    async def _invoke_client(self, system_prompt: str, user_prompt: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """统一调用模型接口"""
        try:
            result = await self.client.translate(system_prompt=system_prompt, user_prompt=user_prompt)
            return result if result else default
        except Exception as e:
            self.logger.error(f"模型调用失败: {e}")
            raise

    async def simplify(self, texts: Dict[str, str]) -> Dict[str, str]:
        """简化文本"""
        system_prompt = SIMPLIFICATION_SYSTEM_PROMPT
        user_prompt = SIMPLIFICATION_USER_PROMPT.format(json_content=texts)
        return await self._invoke_client(system_prompt, user_prompt, {})

    async def simplify_sentences(
        self,
        sentences: List,
        target_speed: float = 1.1
    ) -> List:
        """简化所有句子"""
        if not sentences:
            self.logger.warning("收到空的句子列表")
            return []

        self.logger.debug(f"简化处理 {len(sentences)} 个句子")
        
        try:
            # 构建待简化的文本字典
            texts = {str(i): s.trans_text for i, s in enumerate(sentences)}
            self.logger.debug(f"简化文本: {len(texts)}条")
            
            # 调用模型进行简化
            batch_result = await self.simplify(texts)
            
            if not any(key in batch_result for key in self.SIMPLIFICATION_LEVELS):
                self.logger.error("简化结果格式不正确，缺少必要字段")
                return sentences
                
            # 处理每个句子的简化结果
            for i, s in enumerate(sentences):
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
                
                # 如果有可接受的候选文本（长度小于等于理想长度），选择最长的那个
                if acceptable_candidates:
                    chosen_key, chosen_text = max(acceptable_candidates.items(), key=lambda item: len(item[1]))
                elif non_acceptable_candidates:
                    # 在不可接受的候选中选择最短的那个
                    chosen_key, chosen_text = min(non_acceptable_candidates.items(), key=lambda item: len(item[1]))
                else:
                    chosen_key = "原文"
                    chosen_text = old_text

                s.trans_text = chosen_text
                self.logger.info(
                    f"精简[{chosen_key}]: {old_text} -> {chosen_text} (理想长度: {ideal_length}, 实际长度: {len(chosen_text)}, s.speed: {s.speed})"
                )
                
            return sentences
            
        except Exception as e:
            self.logger.error(f"简化失败: {e}")
            # 出错时使用原句子
            return sentences