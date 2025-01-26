# =========================== deepseek_client.py ===========================
import json
import logging
import re
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

    def _extract_output_content(self, text: str) -> str:
        """从响应中提取 <OUTPUT> 标签中的内容"""
        pattern = r"<OUTPUT>(.*?)</OUTPUT>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        logger.warning("未找到 <OUTPUT> 标签，返回原始内容")
        return text

    async def translate(
        self,
        texts: Dict[str, str],
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, str]:
        """
        将 sync 调用 self.client.chat.completions.create(...) 放到统一的线程池执行。
        """
        try:
            response = await concurrency.run_sync(
                self.client.chat.completions.create,
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            result = response.choices[0].message.content
            logger.debug(f"DeepSeek 请求结果: {result}")
            
            # 提取 <OUTPUT> 标签中的内容
            output_content = self._extract_output_content(result)
            logger.debug(f"提取的 OUTPUT 内容: {output_content}")
            
            parsed_result = loads(output_content)
            logger.debug("DeepSeek 请求成功")
            return parsed_result
            
        except Exception as e:
            logger.error(f"DeepSeek 请求失败: {str(e)}")
            if "503" in str(e):
                logger.error("连接错误：无法连接到 DeepSeek API，可能是代理或网络问题")
            raise
