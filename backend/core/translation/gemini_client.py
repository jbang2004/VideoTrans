import json
import logging
import google.generativeai as genai
from typing import Dict
import os

from .prompt import TRANSLATION_PROMPT, SYSTEM_PROMPT, LANGUAGE_MAP, EXAMPLE_OUTPUTS

logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self, api_key: str):
        """初始化 Gemini 客户端"""
        if not api_key:
            raise ValueError("Gemini API key must be provided")
            
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini 客户端初始化成功")

    async def translate(self, texts: Dict[str, str], target_language: str = "zh") -> str:
        """调用 Gemini 模型进行翻译，返回 JSON 字符串
        
        Args:
            texts: 要翻译的文本字典
            target_language: 目标语言代码 (zh/en/ja/ko)
        """
        try:
            if target_language not in LANGUAGE_MAP:
                raise ValueError(f"不支持的目标语言: {target_language}")
                
            prompt = TRANSLATION_PROMPT.format(
                json_content=json.dumps(texts, ensure_ascii=False, indent=2),
                target_language=LANGUAGE_MAP[target_language],
                example_output=EXAMPLE_OUTPUTS[target_language]
            )
            
            response = self.model.generate_content(
                [SYSTEM_PROMPT.format(target_language=LANGUAGE_MAP[target_language]), prompt],
                generation_config=genai.types.GenerationConfig(temperature=0.3)
            )
            logger.info(f"Gemini 翻译请求成功，目标语言: {LANGUAGE_MAP[target_language]}")
            return response.text
        except Exception as e:
            logger.error(f"Gemini 翻译请求失败: {str(e)}")
            if "503" in str(e):
                logger.error("连接错误：无法连接到 Gemini API，可能是代理或网络问题")
            raise 