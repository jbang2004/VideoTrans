import logging
import json
from json_repair import loads
import openai
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from typing import Dict
from .concurrency import run_sync

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文"
}

TRANSLATION_SYSTEM_PROMPT = "你将扮演久经历练的翻译官，致力于将提供的JSON格式原文翻译成地道的{target_language}。"
TRANSLATION_USER_PROMPT = """
**翻译目标**：{target_language}
请将以下 JSON 中各字段内容翻译成地道的 {target_language}，并保持 JSON 结构：
{json_content}
只返回 JSON, 包含 "thinking" 和 "output" 字段
"""

SIMPLIFICATION_SYSTEM_PROMPT = "你是一个语言简化专家，专注于对文本进行不同程度的精简。"
SIMPLIFICATION_USER_PROMPT = """
请对以下 JSON 结构中 value 部分进行"slight"、"moderate"、"extreme"程度的精简，并输出 "thinking"、"slight"、"moderate"、"extreme" 四个字段：
{json_content}
"""

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        if not api_key:
            raise ValueError("DeepSeek API key must be provided")
        self.api_key = api_key
        openai.api_key = api_key
        openai.api_base = base_url

    async def translate(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        try:
            response = await run_sync(
                openai.ChatCompletion.create,
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.3
            )
            result = response.choices[0].message.content
            if not result.strip():
                raise ValueError("Empty response from DeepSeek")
            return loads(result)
        except Exception as e:
            logger.error(f"DeepSeek translation failed: {e}")
            raise

class GeminiClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key must be provided")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("GeminiClient Initialized")

    async def translate(self, system_prompt: str, user_prompt: str) -> Dict[str, str]:
        try:
            response = await run_sync(
                self.model.generate_content,
                [system_prompt, user_prompt],
                generation_config=GenerationConfig(temperature=0.3)
            )
            result_text = response.text
            if not result_text.strip():
                raise ValueError("Empty response from Gemini")
            return loads(result_text)
        except Exception as e:
            logger.error(f"Gemini translation failed: {e}")
            raise

class Translator:
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger(__name__)

    async def translate(self, texts: Dict[str, str], target_language: str = "zh") -> Dict[str, str]:
        try:
            tgt_lang = LANGUAGE_MAP.get(target_language, target_language)
            system_prompt = TRANSLATION_SYSTEM_PROMPT.format(target_language=tgt_lang)
            user_prompt = TRANSLATION_USER_PROMPT.format(
                target_language=tgt_lang,
                json_content=json.dumps(texts, ensure_ascii=False)
            )
            return await self.client.translate(system_prompt, user_prompt)
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            raise

    async def simplify(self, texts: Dict[str, str]) -> Dict[str, str]:
        try:
            system_prompt = SIMPLIFICATION_SYSTEM_PROMPT
            user_prompt = SIMPLIFICATION_USER_PROMPT.format(
                json_content=json.dumps(texts, ensure_ascii=False)
            )
            return await self.client.translate(system_prompt, user_prompt)
        except Exception as e:
            self.logger.error(f"Simplification failed: {e}")
            raise
