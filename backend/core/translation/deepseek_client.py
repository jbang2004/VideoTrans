import json
import logging
from openai import OpenAI
from typing import Dict
from json_repair import loads

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
        texts: Dict[str, str],
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, str]:
        """调用 DeepSeek 模型进行处理
        
        Args:
            texts: 要处理的文本字典
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            
        Returns:
            处理后的文本字典
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={
                    'type': 'json_object'
                }
            )
            
            result = response.choices[0].message.content
            logger.info(f"DeepSeek 请求结果: {result}")
            logger.info("DeepSeek 请求成功")
            return loads(result)
            
        except Exception as e:
            logger.error(f"DeepSeek 请求失败: {str(e)}")
            if "503" in str(e):
                logger.error("连接错误：无法连接到 DeepSeek API，可能是代理或网络问题")
            raise 