import logging
import asyncio
import torch
import numpy as np
import os
from utils import concurrency

class AudioGenerator:
    def __init__(self, cosyvoice_model, sample_rate: int = None, max_workers=None):
        """
        Args:
            cosyvoice_model: CosyVoice模型
            sample_rate: 采样率，如果为None则使用cosyvoice_model的采样率
        """
        self.cosyvoice_model = cosyvoice_model.model
        self.sample_rate = sample_rate or cosyvoice_model.sample_rate
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
        """异步生成单个音频 (使用 run_sync 统一线程池)"""
        try:
            audio_np = await concurrency.run_sync(self._generate_audio_single, sentence)
            sentence.generated_audio = audio_np
        except Exception as e:
            self.logger.error(f"音频生成失败 (UUID: {sentence.model_input.get('uuid', 'unknown')}): {str(e)}")
            sentence.generated_audio = None

    def _generate_audio_single(self, sentence):
        """生成单个音频或静音，并拼接必要的前后静音。"""

        model_input = sentence.model_input
        self.logger.debug(f"开始生成音频 (主UUID: {model_input.get('uuid', 'unknown')})")

        try:
            segment_audio_list = []

            # 如果 tokens/uuids 为空，我们不直接 return，而是给出一个空数组让后续逻辑继续执行
            tokens_list = model_input.get('segment_speech_tokens', [])
            uuids_list = model_input.get('segment_uuids', [])

            if not tokens_list or not uuids_list:
                # 在这里仅记录日志，并在 segment_audio_list 放一个零长度数组
                self.logger.debug(f"空的语音标记, 仅生成空波形，后续仍可添加静音 (UUID: {model_input.get('uuid', 'unknown')})")
                segment_audio_list.append(np.zeros(0, dtype=np.float32))
            else:
                # 否则逐段生成音频
                for i, (tokens, segment_uuid) in enumerate(zip(tokens_list, uuids_list)):
                    if not tokens:
                        # 如果某个段 token 为空，也放一个零长度数组占位
                        segment_audio_list.append(np.zeros(0, dtype=np.float32))
                        continue

                    token2wav_kwargs = {
                        'token': torch.tensor(tokens).unsqueeze(dim=0),
                        'token_offset': 0,
                        'finalize': True,
                        'prompt_token': model_input.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                        'prompt_feat': model_input.get('prompt_speech_feat', torch.zeros(1, 0, 80)),
                        'embedding': model_input.get('flow_embedding', torch.zeros(0)),
                        'uuid': segment_uuid,
                        'speed': sentence.speed if sentence.speed else 1.0
                    }

                    segment_output = self.cosyvoice_model.token2wav(**token2wav_kwargs)
                    segment_audio = segment_output.cpu().numpy()

                    # 如果是多通道，转单通道
                    if segment_audio.ndim > 1:
                        segment_audio = segment_audio.mean(axis=0)
                    
                    segment_audio_list.append(segment_audio)
                    self.logger.debug(
                        f"段落 {i+1}/{len(uuids_list)} 生成完成，"
                        f"时长: {len(segment_audio)/self.sample_rate:.2f}秒"
                    )

            # 拼接所有段落
            if segment_audio_list:
                final_audio = np.concatenate(segment_audio_list)
            else:
                # 理论上不会出现，因为最少有一个空数组
                final_audio = np.zeros(0, dtype=np.float32)

            # ----- 添加首句静音（若是本分段第一句且 start>0） -----
            if sentence.is_first and sentence.start > 0:
                silence_samples = int(sentence.start * self.sample_rate / 1000)
                final_audio = np.concatenate([np.zeros(silence_samples, dtype=np.float32), final_audio])

            # ----- 添加尾部静音 -----
            if sentence.silence_duration > 0:
                silence_samples = int(sentence.silence_duration * self.sample_rate / 1000)
                final_audio = np.concatenate([final_audio, np.zeros(silence_samples, dtype=np.float32)])

            self.logger.debug(
                f"音频生成完成 (UUID: {model_input.get('uuid', 'unknown')}, "
                f"段落数: {len(segment_audio_list)}, 最终长度: {len(final_audio)/self.sample_rate:.2f}秒)"
            )

            return final_audio

        except Exception as e:
            self.logger.error(f"音频生成失败 (UUID: {model_input.get('uuid', 'unknown')}): {str(e)}")
            raise
