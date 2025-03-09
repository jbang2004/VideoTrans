import logging
import asyncio
import numpy as np
import ray

from utils import concurrency

class AudioGenerator:
    def __init__(self, cosyvoice_model_actor, sample_rate: int = None):
        """
        Args:
            cosyvoice_model_actor: CosyVoice模型Actor引用
            sample_rate: 采样率，如果为None则使用Actor的采样率
        """
        self.cosyvoice_actor = cosyvoice_model_actor
        # 获取采样率（如果未指定）
        self.sample_rate = sample_rate or ray.get(cosyvoice_model_actor.get_sample_rate.remote())
        self.logger = logging.getLogger(__name__)

    async def vocal_audio_maker(self, batch_sentences):
        """异步批量生成音频"""
        tasks = []
        for s in batch_sentences:
            tasks.append(self._generate_single_async(s))

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"音频生成失败: {str(e)}")
            raise

    async def _generate_single_async(self, sentence):
        """异步生成单个音频（调用Actor）"""
        try:
            audio_np = await concurrency.run_sync(
                self._generate_audio_single, sentence
            )
            sentence.generated_audio = audio_np
        except Exception as e:
            self.logger.error(f"音频生成失败 (ID: {sentence.model_input.get('tts_token_id', 'unknown')}): {str(e)}")
            sentence.generated_audio = None

    def _generate_audio_single(self, sentence):
        """调用Actor生成音频"""
        model_input = sentence.model_input
        self.logger.debug(f"开始生成音频 (TTS Token ID: {model_input.get('tts_token_id', 'unknown')})")

        try:
            tts_token_id = model_input.get('tts_token_id')
            speaker_feature_id = model_input.get('speaker_feature_id')

            if not tts_token_id or not speaker_feature_id:
                self.logger.debug(f"缺少必要的参数，仅生成空波形 (TTS Token ID: {tts_token_id})")
                return np.zeros(0, dtype=np.float32)
            
            # 获取语速
            speed = sentence.speed if sentence.speed else 1.0
            
            # 调用Actor生成音频，使用缓存的特征
            audio = ray.get(self.cosyvoice_actor.generate_audio.remote(
                tts_token_id, speaker_feature_id, speed
            ))
            
            # 添加首句静音
            if sentence.is_first and sentence.start > 0:
                silence_samples = int(sentence.start * self.sample_rate / 1000)
                audio = np.concatenate([np.zeros(silence_samples, dtype=np.float32), audio])

            # 添加尾部静音
            if sentence.silence_duration > 0:
                silence_samples = int(sentence.silence_duration * self.sample_rate / 1000)
                audio = np.concatenate([audio, np.zeros(silence_samples, dtype=np.float32)])

            return audio

        except Exception as e:
            self.logger.error(f"音频生成失败 (TTS Token ID: {model_input.get('tts_token_id', 'unknown')}): {str(e)}")
            raise
