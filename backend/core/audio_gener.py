# core/audio_gener.py

import logging
import asyncio
import numpy as np
from services.cosyvoice.client import CosyVoiceClient
from utils import concurrency

class AudioGenerator:
    def __init__(self, cosyvoice_client: CosyVoiceClient, sample_rate: int = 24000):
        self.cosyvoice_client = cosyvoice_client
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

    async def vocal_audio_maker(self, sentences):
        tasks = []
        for s in sentences:
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
                f"音频生成失败 (UUID: {sentence.model_input.get('uuid', 'unknown')}): {str(e)}"
            )
            sentence.generated_audio = None
            return sentence

    def _generate_audio_single(self, sentence):
        """
        直接使用 'tts_tokens_features' 来 token2wav
        """
        if 'tts_tokens_features' not in sentence.model_input:
            self.logger.warning("没有 tts_tokens_features，无法合成音频")
            return np.zeros(0, dtype=np.float32)

        model_input = sentence.model_input['tts_tokens_features']
        speed = getattr(sentence, 'speed', 1.0) or 1.0

        audio_np, dur_sec = self.cosyvoice_client.token2wav(model_input, speed=speed)

        # 首段静音
        if getattr(sentence, 'is_first', False) and getattr(sentence, 'start', 0) > 0:
            silence_samples = int(sentence.start * self.sample_rate / 1000)
            audio_np = np.concatenate([
                np.zeros(silence_samples, dtype=np.float32),
                audio_np
            ])

        # 尾部留白
        if hasattr(sentence, 'silence_duration') and sentence.silence_duration > 0:
            silence_samples = int(sentence.silence_duration * self.sample_rate / 1000)
            audio_np = np.concatenate([
                audio_np,
                np.zeros(silence_samples, dtype=np.float32)
            ])

        self.logger.debug(
            f"音频生成完成 (UUID: {sentence.model_input.get('uuid', 'unknown')}), "
            f"长度={len(audio_np)/self.sample_rate:.2f}s"
        )
        return audio_np
