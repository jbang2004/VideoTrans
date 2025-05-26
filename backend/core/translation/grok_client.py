import asyncio
import logging
from openai import OpenAI
from json_repair import loads
from typing import Dict

logger = logging.getLogger(__name__)

class GrokClient:
    """
    基于 x.ai（Grok）API 的 "翻译" 客户端。
    使用 OpenAI SDK，向 https://api.x.ai/v1 发送 chat completion，
    并将回复解析为 JSON 返回。
    """
    def __init__(self, api_key: str, model_name: str = "grok-3-mini-fast"):
        if not api_key:
            raise ValueError("必须提供 x.ai (Grok) 的 API Key")
        # 构造 OpenAI client，base_url 指向 x.ai
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )
        self.model_name = model_name

    async def translate(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        """
        执行一次 chat completion，请求返回 JSON 字符串，
        再解析为字典并返回。格式需包含 "output" 字段，
        以供上层 Simplifier 使用。
        """
        try:
            # 调用 OpenAI SDK（同步接口）放到线程池
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            # 拿到模型返回的文本
            text = response.choices[0].message.content
            logger.debug(f"[GrokClient] 原始返回: {text!r}")
            # 解析成 dict
            result = loads(text)
            return result if isinstance(result, dict) else {"output": {}}
        except Exception as e:
            logger.error(f"[GrokClient] 请求/解析失败: {e}")
            raise 