# =========================== deepseek_client.py ===========================
import json
import logging
import asyncio
import httpx
from typing import Dict
from json_repair import loads

logger = logging.getLogger(__name__)

class DeepSeekClient:
    BASE_URL = "https://api.deepseek.com/chat/completions"

    def __init__(self, api_key: str):
        """初始化 DeepSeek 客户端"""
        if not api_key:
            raise ValueError("DeepSeek API key must be provided")

        self.api_key = api_key
        # 移除 OpenAI 客户端初始化
        # self.client = OpenAI(
        #     api_key=api_key,
        #     base_url="https://api.deepseek.com",
        # )
        # 改为初始化 httpx 异步客户端, 并增加超时时间
        self.http_client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0  # 设置超时时间为 30 秒
        )
        logger.info("DeepSeek 客户端初始化成功 (使用 httpx, 超时 30s)")

    async def translate(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Dict[str, str]:
        """
        直接调用 DeepSeek API (使用 httpx)，要求返回 JSON 格式的内容。
        """
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 1.3, # 注意：文档建议 temperature 范围通常在 0 到 1 之间，1.3 较高
            "stream": False # 确保不是流式响应
        }

        try:
            # 之前: 使用 openai client
            # response = await asyncio.to_thread(
            #     self.client.chat.completions.create,
            #     model="deepseek-chat",
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": user_prompt}
            #     ],
            #     temperature=1.3
            # )
            # result = response.choices[0].message.content

            # 之后: 使用 httpx post 请求
            response = await self.http_client.post(self.BASE_URL, json=payload)
            response.raise_for_status() # 如果状态码不是 2xx，则抛出异常
            response_data = response.json()

            # 检查响应结构是否符合预期
            if not response_data or 'choices' not in response_data or not response_data['choices']:
                logger.error(f"DeepSeek 返回了无效的响应结构: {response_data}")
                raise ValueError("Invalid response structure from DeepSeek")

            result = response_data['choices'][0]['message']['content']

            logger.info(f"DeepSeek 原文请求内容:\n{user_prompt}")
            logger.info(f"DeepSeek 原始返回内容 (长度: {len(result)}):\n{result!r}")

            if not result or not result.strip():
                logger.error("DeepSeek 返回了空响应")
                raise ValueError("Empty response from DeepSeek")

            # 尝试修复和解析 JSON
            try:
                parsed_result = loads(result)
                logger.debug("DeepSeek 请求成功，JSON 解析完成")
                return parsed_result
            except Exception as json_error:
                logger.error(f"JSON 解析失败，原始内容: {result!r}")
                logger.error(f"JSON 解析错误详情: {str(json_error)}")
                raise

        except httpx.HTTPStatusError as e:
            logger.error(f"DeepSeek API 请求失败，状态码: {e.response.status_code}, 响应: {e.response.text}")
            if e.response.status_code == 503:
                 logger.error("连接错误：无法连接到 DeepSeek API，可能是代理或网络问题")
            raise
        except Exception as e:
            # 之前: logger.error(f"DeepSeek 处理失败: {str(e)}")
            # 之后: 使用 exc_info=True 记录更详细的错误信息和 traceback
            logger.error(f"DeepSeek 处理失败: {type(e).__name__}", exc_info=True)
            raise

    # 添加一个关闭 http 客户端的方法，以便在应用结束时调用
    async def close(self):
        await self.http_client.aclose()
        logger.info("DeepSeek httpx 客户端已关闭")
