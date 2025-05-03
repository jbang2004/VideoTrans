import asyncio
import logging
from groq import AsyncGroq
from json_repair import loads
from typing import Dict

logger = logging.getLogger(__name__)

class GroqClient:
    """
    基于 Groq Chat Completions API 的翻译客户端。
    使用 Groq SDK 向 Groq 服务发送 chat completion 请求，
    并将返回的 JSON 解析为字典返回。
    """
    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-4-maverick-17b-128e-instruct"):
        if not api_key:
            raise ValueError("必须提供 Groq 的 API Key")
        # 初始化 Groq 客户端
        self.client = AsyncGroq(api_key=api_key)
        self.model_name = model_name

    async def translate(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        """
        执行一次 chat completion，请求返回 JSON 字符串，
        解析为字典返回。期望返回结果包含 "output" 字段供上层使用。
        """
        try:
            # 将同步接口放到线程池中执行
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            # 获取模型返回文本
            text = response.choices[0].message.content
            logger.debug(f"[GroqClient] 原始返回: {text!r}")
            # 修复并解析 JSON
            result = loads(text)
            if isinstance(result, dict):
                return result
            # 若非 dict，则返回空 output
            return {"output": {}}
        except Exception as e:
            logger.error(f"[GroqClient] 请求/解析失败: {e}")
            raise 