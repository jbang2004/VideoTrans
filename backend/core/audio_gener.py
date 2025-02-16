import logging
import asyncio
import numpy as np
import torch
from utils import concurrency

class AudioGenerator:
    def __init__(self, cosyvoice_client, sample_rate: int = 24000):
        """
        Args:
            cosyvoice_client: gRPC 封装 (CosyVoiceClient)
            sample_rate: 最终合成采样率
        """
        self.cosyvoice_client = cosyvoice_client
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

    async def vocal_audio_maker(self, batch_sentences):
        """
        异步批量生成音频
        """
        tasks = []
        for s in batch_sentences:
            tasks.append(self._generate_single_async(s))

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"音频生成失败: {str(e)}")
            raise

    async def _generate_single_async(self, sentence):
        """
        异步：核心逻辑 -> concurrency.run_sync
        """
        try:
            final_audio = await concurrency.run_sync(
                self._generate_audio_single, sentence
            )
            sentence.generated_audio = final_audio
        except Exception as e:
            self.logger.error(
                f"音频生成失败 (UUID: {sentence.model_input.get('uuid', 'unknown')}): {str(e)}"
            )
            sentence.generated_audio = None

    def _generate_audio_single(self, sentence):
        """生成单个句子的音频"""
        model_input = sentence.model_input
        self.logger.debug(f"开始生成音频 (UUID: {model_input.get('uuid', 'unknown')})")

        try:
            # 获取tokens
            tts_token_list = model_input.get('segment_speech_tokens', [])
            uuids_list = model_input.get('segment_uuids', [])

            if not tts_token_list:
                self.logger.debug(f"空的语音标记, 仅生成空波形 (UUID: {model_input.get('uuid', 'unknown')})")
                final_audio = np.zeros(0, dtype=np.float32)
            else:
                # 准备说话人特征
                speaker_info = {
                    'prompt_token': model_input.get('prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                    'prompt_feat': model_input.get('prompt_speech_feat', torch.zeros(1, 0, 80)),
                    'embedding': model_input.get('embedding', torch.zeros(0, 192))  # 使用统一的embedding字段
                }
                
                # 生成音频
                speed = sentence.speed if sentence.speed else 1.0
                
                # 一次性生成所有段落的音频
                res = self.cosyvoice_client.token2wav(
                    tokens_list=tts_token_list,
                    uuids_list=uuids_list,
                    speaker_info=speaker_info,
                    speed=speed
                )
                final_audio = res['audio']

            # 处理首句静音
            if sentence.is_first and sentence.start > 0:
                silence_samples = int(sentence.start * self.sample_rate / 1000)
                final_audio = np.concatenate([
                    np.zeros(silence_samples, dtype=np.float32),
                    final_audio
                ])

            # 尾部留白
            if hasattr(sentence, 'silence_duration') and sentence.silence_duration > 0:
                silence_samples = int(sentence.silence_duration * self.sample_rate / 1000)
                final_audio = np.concatenate([
                    final_audio,
                    np.zeros(silence_samples, dtype=np.float32)
                ])

            self.logger.debug(
                f"音频生成完成 (UUID: {model_input.get('uuid', 'unknown')}, "
                f"最终长度: {len(final_audio)/self.sample_rate:.2f}秒)"
            )

            return final_audio

        except Exception as e:
            self.logger.error(
                f"音频生成失败 (UUID: {model_input.get('uuid', 'unknown')}): {str(e)}"
            )
            raise
