import json
import logging
from zhipuai import ZhipuAI
from .prompt import GLM4_TRANSLATION_PROMPT, GLM4_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class GLM4Client:
    def __init__(self, api_key: str):
        """初始化 GLM-4 客户端"""
        if not api_key:
            raise ValueError("API key must be provided")
        self.client = ZhipuAI(api_key=api_key)
        logger.info("GLM-4 客户端初始化成功")

    async def translate(self, texts: dict) -> str:
        """调用 GLM-4 模型进行翻译，返回 JSON 字符串"""
        prompt = GLM4_TRANSLATION_PROMPT.format(json_content=json.dumps(texts, ensure_ascii=False, indent=2))
        try:
            logger.debug(f"需要翻译的JSON: {json.dumps(texts, ensure_ascii=False, indent=2)}")
            response = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": GLM4_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.8
            )
            content = response.choices[0].message.content
            logger.debug(f"翻译结果: {content}")
            return content
        except Exception as e:
            logger.error(f"GLM-4 翻译请求失败: {str(e)}")
            raise
