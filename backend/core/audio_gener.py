import logging
import asyncio
import torch
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
            self.logger.error(f"音频生成失败 (UUID: {sentence.model_input.get('uuid', 'unknown')}): {str(e)}")
            sentence.generated_audio = None

    def _generate_audio_single(self, sentence):
        """调用Actor生成音频"""
        model_input = sentence.model_input
        self.logger.debug(f"开始生成音频 (主UUID: {model_input.get('uuid', 'unknown')})")

        try:
            segment_audio_list = []
            
            tokens_list = model_input.get('segment_speech_tokens', [])
            uuids_list = model_input.get('segment_uuids', [])

            if not tokens_list or not uuids_list:
                self.logger.debug(f"空的语音标记, 仅生成空波形 (UUID: {model_input.get('uuid', 'unknown')})")
                segment_audio_list.append(np.zeros(0, dtype=np.float32))
            else:
                for i, (tokens, segment_uuid) in enumerate(zip(tokens_list, uuids_list)):
                    if not tokens:
                        segment_audio_list.append(np.zeros(0, dtype=np.float32))
                        continue

                    # 准备参数
                    token_tensor = torch.tensor(tokens).unsqueeze(dim=0)
                    prompt_token = model_input.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32))
                    prompt_feat = model_input.get('prompt_speech_feat', torch.zeros(1, 0, 80))
                    embedding = model_input.get('flow_embedding', torch.zeros(0))
                    speed = sentence.speed if sentence.speed else 1.0
                    
                    # 调用Actor生成音频
                    segment_audio = ray.get(self.cosyvoice_actor.generate_audio.remote(
                        token_tensor, 0, segment_uuid, prompt_token, prompt_feat, embedding, speed
                    ))
                    
                    # 如果是多通道，转单通道
                    if segment_audio.ndim > 1:
                        segment_audio = segment_audio.mean(axis=0)
                    
                    segment_audio_list.append(segment_audio)
                    self.logger.debug(
                        f"段落 {i+1}/{len(uuids_list)} 生成完成，"
                        f"时长: {len(segment_audio)/self.sample_rate:.2f}秒"
                    )

            # 拼接音频和静音（与原代码相同）
            if segment_audio_list:
                final_audio = np.concatenate(segment_audio_list)
            else:
                final_audio = np.zeros(0, dtype=np.float32)

            # 添加首句静音
            if sentence.is_first and sentence.start > 0:
                silence_samples = int(sentence.start * self.sample_rate / 1000)
                final_audio = np.concatenate([np.zeros(silence_samples, dtype=np.float32), final_audio])

            # 添加尾部静音
            if sentence.silence_duration > 0:
                silence_samples = int(sentence.silence_duration * self.sample_rate / 1000)
                final_audio = np.concatenate([final_audio, np.zeros(silence_samples, dtype=np.float32)])

            return final_audio

        except Exception as e:
            self.logger.error(f"音频生成失败 (UUID: {model_input.get('uuid', 'unknown')}): {str(e)}")
            raise
