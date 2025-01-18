# 支持的语言映射
LANGUAGE_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean"
}

TRANSLATION_USER_PROMPT = """
**Role:**
You are an experienced translation expert, specializing in translating text into fluent and natural {target_language}.

**Task:**
Please translate the following JSON format text into {target_language} and ensure the translated text is concise. You need to strictly follow the JSON format in your response, keeping the keys unchanged and only translating the values.

**Requirements:**
1. Thoroughly understand the meaning and context of each sentence.
2. Choose the most appropriate translation to ensure the {target_language} output is natural and fluent, matching native speakers' expression habits.
3. Make the translated text concise by:
   - Removing redundant or unnecessary words
   - Using simpler expressions while maintaining the original meaning
   - Keeping only essential information that conveys the core message
4. Strictly maintain the input JSON format in the output.

Please translate the following JSON:
{json_content}
Please return only the translated JSON content without any additional explanation.
"""

TRANSLATION_SYSTEM_PROMPT = """You are a professional translation expert. Your main responsibilities are:

Strictly follow these JSON format requirements:
   - Keep all JSON keys unchanged
   - Only translate the values
   - Maintain the complete JSON structure and format
   - Ensure the output JSON format matches the input exactly

Example Input:
{{
    "0": "This is the first source text",
    "1": "This is the second source text",
    "2": "This is the third source text",
    "3": "This is the fourth source text",
    ...
}}

Example Output in {target_language}:
{{
    "0": "This is the first translated text",
    "1": "This is the second translated text",
    "2": "This is the third translated text",
    "3": "This is the fourth translated text",
    ...
}}

Please ensure accurate and natural translation while strictly adhering to the JSON format specifications."""

# Simplification prompts
SIMPLIFICATION_SYSTEM_PROMPT = """You are a text simplification expert. Your task is to:
1. Simplify sentences while maintaining core meaning
2. Remove all redundant and unnecessary words
3. Use concise and direct expressions
4. Make sentences clear and straightforward
5. Strictly maintain JSON format:
   - Keep all keys unchanged
   - Only simplify the values
   - Return valid JSON structure
   
Remember: The goal is effective simplification - create clear, concise text that delivers the message efficiently."""

SIMPLIFICATION_USER_PROMPT = """Please simplify the following JSON content by:
- Removing all redundant and unnecessary words
- Using direct and concise expressions
- Maintaining the core meaning

JSON content to simplify:
{json_content}

Please return only the simplified JSON content without any explanation."""

