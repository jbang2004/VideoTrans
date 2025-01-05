import json
import logging
import aiohttp
from typing import Dict
from .prompt import TRANSLATION_PROMPT, SYSTEM_PROMPT, LANGUAGE_MAP, EXAMPLE_OUTPUTS

logger = logging.getLogger(__name__)

class DeepSeekClient:
    def __init__(self, api_key: str):
        """初始化 DeepSeek 客户端"""
        if not api_key:
            raise ValueError("DeepSeek API key must be provided")
            
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self._session = None
        logger.info("DeepSeek 客户端初始化成功")
    
    async def _ensure_session(self):
        """确保 aiohttp session 已创建"""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._session

    async def translate(self, texts: Dict[str, str], target_language: str = "zh") -> str:
        """调用 DeepSeek 模型进行翻译，返回 JSON 字符串
        
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
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(target_language=LANGUAGE_MAP[target_language])},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            session = await self._ensure_session()
            async with session.post(self.base_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"DeepSeek API 请求失败: {response.status} - {error_text}")
                    
                result = await response.json()
                logger.info(f"DeepSeek 翻译请求成功，目标语言: {LANGUAGE_MAP[target_language]}")
                return result["choices"][0]["message"]["content"]
                
        except aiohttp.ClientError as e:
            logger.error(f"DeepSeek API 连接错误: {str(e)}")
            if "503" in str(e):
                logger.error("连接错误：无法连接到 DeepSeek API，可能是代理或网络问题")
            raise
        except Exception as e:
            logger.error(f"DeepSeek 翻译请求失败: {str(e)}")
            raise
            
    async def close(self):
        """关闭 aiohttp session"""
        if self._session is not None:
            await self._session.close()
            self._session = None 