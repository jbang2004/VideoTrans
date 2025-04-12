import logging
from typing import Dict
import asyncio

from google import genai
from google.genai import types

from json_repair import loads

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key: str):
        """初始化 Gemini 客户端"""
        if not api_key:
            raise ValueError("Gemini API key must be provided")
        # 配置 Gemini
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-2.0-flash'
        logger.info("Gemini 客户端初始化成功 (使用 google-genai SDK)")
    
    async def translate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, str]:
        """
        直接调用 Gemini API，要求返回 JSON 格式的内容。
        """
        try:

            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.8)
            )
            logger.info(f"Gemini 原文请求内容:\n{user_prompt}")
            result_text = response.text
            logger.info(f"Gemini 原始返回内容 (长度: {len(result_text)}):\n{result_text!r}")
            
            if not result_text or not result_text.strip():
                logger.error("Gemini 返回了空响应")
                raise ValueError("Empty response from Gemini")
                
            # 尝试修复和解析 JSON
            try:
                parsed_result = loads(result_text)
                logger.debug("Gemini 请求成功，JSON 解析完成")
                return parsed_result
            except Exception as json_error:
                logger.error(f"JSON 解析失败，原始内容: {result_text!r}")
                logger.error(f"JSON 解析错误详情: {str(json_error)}")
                raise
            
        except Exception as e:
            logger.error(f"Gemini 请求失败: {str(e)}")
            raise