# =========================== gemini_client.py ===========================
import logging
import re
from typing import Dict

# google.generativeai is synchronous, so we wrap it in concurrency.run_sync
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from json_repair import loads

# [MODIFIED] 引入统一线程管理，与 deepseek_client 的用法一致
from utils import concurrency

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key: str):
        """初始化 Gemini 客户端"""
        if not api_key:
            raise ValueError("Gemini API key must be provided")

        # 配置 Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash') # gemini-1.5-flash、gemini-2.0-flash-exp
        logger.info("Gemini 客户端初始化成功")

    def _extract_output_content(self, text: str) -> str:
        """
        从响应中提取 <OUTPUT>...</OUTPUT> 标签中的内容。如果没有找到，
        记录警告并直接返回原文本，让后续的 JSON 解析尝试处理。
        """
        pattern = r"<OUTPUT>(.*?)</OUTPUT>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        logger.warning("未找到 <OUTPUT> 标签，返回原始内容供 JSON 解析")
        return text

    async def translate(
        self,
        texts: Dict[str, str],
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, str]:
        """
        以与 DeepSeekClient 相同的方式调用，返回 {key: translation} 格式的字典。
        """
        try:
            # 使用 concurrency.run_sync 调用同步的 Gemini API
            response = await concurrency.run_sync(
                self.model.generate_content,
                [system_prompt, user_prompt],
                generation_config=GenerationConfig(temperature=0.3)
            )
            logger.info(f"Gemini 原文请求内容:\n{user_prompt}")
            # Gemini 的返回对象通常含有一个 .text 属性
            result_text = response.text

            logger.info(f"Gemini 原始返回内容:\n{result_text}")

            # 提取 <OUTPUT>...</OUTPUT> 内容
            output_content = self._extract_output_content(result_text)
            logger.debug(f"提取的 <OUTPUT> 内容:\n{output_content}")

            # 使用 json_repair.loads 解析 JSON，自动修复轻微格式问题
            parsed_result = loads(output_content)

            # 应返回一个 dict，与传入 texts 的 keys 相对应
            logger.debug("Gemini 请求成功，JSON 解析完成")
            return parsed_result

        except Exception as e:
            logger.error(f"Gemini 请求失败: {str(e)}")
            raise
