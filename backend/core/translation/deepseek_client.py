# =========================== deepseek_client.py ===========================
import json
import logging
from openai import OpenAI
from typing import Dict
from json_repair import loads

# [MODIFIED] 引入统一线程管理
from utils import concurrency

logger = logging.getLogger(__name__)

class DeepSeekClient:
    def __init__(self, api_key: str):
        """初始化 DeepSeek 客户端"""
        if not api_key:
            raise ValueError("DeepSeek API key must be provided")
            
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        logger.info("DeepSeek 客户端初始化成功")

    async def translate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, str]:
        """
        直接调用 DeepSeek API，要求返回 JSON 格式的内容。
        """
        try:
            response = await concurrency.run_sync(
                self.client.chat.completions.create,
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.3
            )
            result = response.choices[0].message.content
            logger.info(f"DeepSeek 原文请求内容:\n{user_prompt}")
            logger.info(f"DeepSeek 原始返回内容:\n{result}")
            # 直接解析返回的 JSON 格式文本
            parsed_result = loads(result)
            logger.debug("DeepSeek 请求成功，JSON 解析完成")
            return parsed_result
            
        except Exception as e:
            logger.error(f"DeepSeek 请求失败: {str(e)}")
            if "503" in str(e):
                logger.error("连接错误：无法连接到 DeepSeek API，可能是代理或网络问题")
            raise
