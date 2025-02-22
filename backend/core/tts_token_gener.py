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

    async def tts_token_maker(self, sentences: List, reuse_uuid: bool = False):
        """
        并发生成TTS tokens，并存储预估时长
        """
        if not sentences:
            return []

        tasks = []
        for s in sentences:
            if not reuse_uuid and 'uuid' not in s.model_input:
                self.logger.warning("句子缺少UUID，无法生成TTS tokens")
                continue
            
            uuid = s.model_input.get('uuid')
            if not uuid:
                self.logger.warning("句子的UUID为空，无法生成TTS tokens")
                continue

            tasks.append(asyncio.create_task(self._generate_tts_single_async(s, uuid)))

        processed = await asyncio.gather(*tasks)
        return processed

    async def _generate_tts_single_async(self, sentence, uuid: str):
        return await concurrency.run_sync(
            self._generate_tts_single, sentence, uuid
        )

    def _generate_tts_single(self, sentence, uuid: str):
        # 生成TTS tokens并获取时长
        duration_ms, success = self.cosyvoice_client.generate_tts_tokens(uuid)
        if not success:
            self.logger.error(f"生成TTS tokens失败 (UUID={uuid})")
            return sentence

        sentence.duration = duration_ms
        self.logger.debug(f"[TTS Token] (UUID={uuid}) 生成完毕 => 估计时长 {duration_ms}ms")
        return sentence

    def _merge_features(self, text_f, speaker_f):
        merged = type(text_f)()
        # 拷贝 text_f
        merged.normalized_text_segments.extend(text_f.normalized_text_segments)
        for seg in text_f.text_segments:
            seg_msg = merged.text_segments.add()
            seg_msg.tokens.extend(seg.tokens)

        # 拷贝 speaker_f
        if speaker_f:
            merged.embedding.extend(speaker_f.embedding)
            merged.prompt_speech_feat.extend(speaker_f.prompt_speech_feat)
            merged.prompt_speech_feat_len = speaker_f.prompt_speech_feat_len
            merged.prompt_speech_token.extend(speaker_f.prompt_speech_token)
            merged.prompt_speech_token_len = speaker_f.prompt_speech_token_len

        return merged
