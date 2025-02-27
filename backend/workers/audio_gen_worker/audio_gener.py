# core/audio_gener.py

import logging
import asyncio
import numpy as np
from typing import List
from services.cosyvoice.client import CosyVoiceClient
from utils import concurrency

class AudioGenerator:
    def __init__(self, cosyvoice_client: CosyVoiceClient, sample_rate: int = 24000):
        self.cosyvoice_client = cosyvoice_client
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

    async def vocal_audio_maker(self, sentences: List):
        """
        并发生成音频
        """
        tasks = []
        for s in sentences:
            text_uuid = s.model_input.get('text_uuid')
            speaker_uuid = s.model_input.get('speaker_uuid')
            if not text_uuid or not speaker_uuid:
                self.logger.warning("缺少text_uuid或speaker_uuid，无法生成音频")
                continue

            tasks.append(self._generate_single_async(s))

        try:
            results = await asyncio.gather(*tasks)
            return results
        except Exception as e:
            self.logger.error(f"音频生成失败: {str(e)}")
            raise

    async def _generate_single_async(self, sentence):
        try:
            final_audio = await concurrency.run_sync(self._generate_audio_single, sentence)
            sentence.generated_audio = final_audio
            return sentence
        except Exception as e:
            self.logger.error(
                f"音频生成失败 (text_uuid: {sentence.model_input.get('text_uuid', 'unknown')}, "
                f"speaker_uuid: {sentence.model_input.get('speaker_uuid', 'unknown')}): {str(e)}"
            )
            sentence.generated_audio = None
            return sentence

    def _generate_audio_single(self, sentence):
        """
        使用text_uuid和speaker_uuid从服务端获取音频
        """
        text_uuid = sentence.model_input.get('text_uuid')
        speaker_uuid = sentence.model_input.get('speaker_uuid')
        if not text_uuid or not speaker_uuid:
            self.logger.warning("缺少text_uuid或speaker_uuid，无法生成音频")
            return np.zeros(0, dtype=np.float32)

        speed = getattr(sentence, 'speed', 1.0) or 1.0
        audio_np, dur_sec = self.cosyvoice_client.token2wav(text_uuid, speaker_uuid, speed=speed)

        if getattr(sentence, 'is_first', False) and getattr(sentence, 'start', 0) > 0:
            silence_samples = int(sentence.start * self.sample_rate / 1000)
            audio_np = np.concatenate([
                np.zeros(silence_samples, dtype=np.float32),
                audio_np
            ])

        if hasattr(sentence, 'silence_duration') and sentence.silence_duration > 0:
            silence_samples = int(sentence.silence_duration * self.sample_rate / 1000)
            audio_np = np.concatenate([
                audio_np,
                np.zeros(silence_samples, dtype=np.float32)
            ])

        self.logger.debug(
            f"音频生成完成 (text_uuid={text_uuid}, speaker_uuid={speaker_uuid}), "
            f"长度={len(audio_np)/self.sample_rate:.2f}s"
        )
        return audio_np