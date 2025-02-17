# core/tts_token_gener.py

import logging
import asyncio
import uuid
from services.cosyvoice.client import CosyVoiceClient
from utils import concurrency

class TTSTokenGenerator:
    def __init__(self, cosyvoice_client: CosyVoiceClient):
        self.cosyvoice_client = cosyvoice_client
        self.logger = logging.getLogger(__name__)

    async def tts_token_maker(self, sentences, reuse_uuid=False):
        """
        并发生成TTS tokens, 直接存到 sentence.model_input['tts_tokens_features']
        """
        if not sentences:
            return []

        tasks = []
        for s in sentences:
            if reuse_uuid and 'uuid' in s.model_input and s.model_input['uuid']:
                u = s.model_input['uuid']
            else:
                u = str(uuid.uuid1())
                s.model_input['uuid'] = u

            tasks.append(asyncio.create_task(self._generate_tts_single_async(s, u)))

        processed = await asyncio.gather(*tasks)
        return processed

    async def _generate_tts_single_async(self, sentence, main_uuid):
        return await concurrency.run_sync(
            self._generate_tts_single, sentence, main_uuid
        )

    def _generate_tts_single(self, sentence, main_uuid):
        text_f = sentence.model_input.get('text_features')
        speaker_f = sentence.model_input.get('speaker_features')

        merged_f = self._merge_features(text_f, speaker_f)
        out_features, duration_ms = self.cosyvoice_client.generate_tts_tokens(merged_f, main_uuid)
        sentence.duration = duration_ms

        # 直接存储到 'tts_tokens_features'
        sentence.model_input['tts_tokens_features'] = out_features

        self.logger.debug(f"[TTS Token] (UUID={main_uuid}) 生成完毕 => 估计时长 {duration_ms}ms")
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
