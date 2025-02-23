# core/tts_token_gener.py

import logging
import asyncio
from typing import List
from services.cosyvoice.client import CosyVoiceClient
from utils import concurrency

class TTSTokenGenerator:
    def __init__(self, cosyvoice_client: CosyVoiceClient):
        self.cosyvoice_client = cosyvoice_client
        self.logger = logging.getLogger(__name__)

    async def tts_token_maker(self, sentences: List):
        """
        并发生成TTS tokens，并存储预估时长
        """
        if not sentences:
            return []

        tasks = []
        for s in sentences:
            text_uuid = s.model_input.get('text_uuid')
            speaker_uuid = s.model_input.get('speaker_uuid')
            if not text_uuid or not speaker_uuid:
                self.logger.warning("缺少text_uuid或speaker_uuid，无法生成TTS tokens")
                continue

            tasks.append(asyncio.create_task(self._generate_tts_single_async(s, text_uuid, speaker_uuid)))

        processed = await asyncio.gather(*tasks)
        return processed

    async def _generate_tts_single_async(self, sentence, text_uuid: str, speaker_uuid: str):
        return await concurrency.run_sync(
            self._generate_tts_single, sentence, text_uuid, speaker_uuid
        )

    def _generate_tts_single(self, sentence, text_uuid: str, speaker_uuid: str):
        duration_ms, success = self.cosyvoice_client.generate_tts_tokens(text_uuid, speaker_uuid)
        if not success:
            self.logger.error(f"生成TTS tokens失败 (text_uuid={text_uuid}, speaker_uuid={speaker_uuid})")
            return sentence

        sentence.duration = duration_ms
        self.logger.debug(f"[TTS Token] (text_uuid={text_uuid}, speaker_uuid={speaker_uuid}) 生成完毕 => 估计时长 {duration_ms}ms")
        return sentence