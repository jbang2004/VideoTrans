import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import os
import inspect

class AudioGenerator:
    def __init__(self, cosyvoice_model, sample_rate: int = None, max_workers=None):
        """
        Args:
            cosyvoice_model: CosyVoice模型
            sample_rate: 采样率，如果为None则使用cosyvoice_model的采样率
            max_workers: 并行处理的最大工作线程数，默认为None（将根据CPU核心数自动设置）
        """
        self.cosyvoice_model = cosyvoice_model.model
        self.sample_rate = sample_rate or cosyvoice_model.sample_rate
        cpu_count = os.cpu_count()
        self.max_workers = min(max_workers or cpu_count, cpu_count)
        self.executor = None
        self.logger = logging.getLogger(__name__)

    async def vocal_audio_maker(self, batch_sentences):
        """异步批量生成音频"""
        try:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                self.logger.debug(f"创建音频生成线程池，max_workers={self.max_workers}")
            
            tasks = [
                self._generate_single_async(sentence)
                for sentence in batch_sentences
            ]
            await asyncio.gather(*tasks)
            
        except Exception as e:
            self.logger.error(f"音频生成失败: {str(e)}")
            raise

    async def _generate_single_async(self, sentence):
        """异步生成单个音频"""
        loop = asyncio.get_event_loop()
        try:
            audio_np = await loop.run_in_executor(
                self.executor, 
                self._generate_audio_single, 
                sentence
            )
            sentence.generated_audio = audio_np
            
        except Exception as e:
            self.logger.error(f"音频生成失败 (UUID: {sentence.model_input.get('uuid', 'unknown')}): {str(e)}")
            sentence.generated_audio = None

    def _generate_audio_single(self, sentence):
        """生成单个音频"""
        model_input = sentence.model_input
        self.logger.debug(f"开始生成音频 (主UUID: {model_input.get('uuid', 'unknown')})")
        
        try:
            if not model_input.get('segment_speech_tokens') or not model_input.get('segment_uuids'):
                self.logger.debug(f"空的语音标记，创建空音频 (UUID: {model_input.get('uuid', 'unknown')})")
                return np.zeros(0)

            segment_audio_list = []
            
            for i, (tokens, segment_uuid) in enumerate(zip(model_input['segment_speech_tokens'], 
                                                         model_input['segment_uuids'])):
                if not tokens:
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
                if segment_audio.ndim > 1:
                    segment_audio = segment_audio.mean(axis=0)
                segment_audio_list.append(segment_audio)
                
                self.logger.debug(f"段落 {i+1}/{len(model_input['segment_uuids'])} 生成完成，"
                                  f"时长: {len(segment_audio)/self.sample_rate:.2f}秒")

            if not segment_audio_list:
                return np.zeros(0)
                
            final_audio = np.concatenate(segment_audio_list)

            if sentence.is_first and sentence.start > 0:
                silence_samples = int(sentence.start * self.sample_rate / 1000)
                final_audio = np.concatenate([np.zeros(silence_samples), final_audio])

            if sentence.silence_duration > 0:
                silence_samples = int(sentence.silence_duration * self.sample_rate / 1000)
                final_audio = np.concatenate([final_audio, np.zeros(silence_samples)])

            self.logger.debug(f"音频生成完成 (主UUID: {model_input.get('uuid', 'unknown')}, "
                              f"段落数: {len(segment_audio_list)}, "
                              f"总时长: {len(final_audio)/self.sample_rate:.2f}秒)")
            
            return final_audio

        except Exception as e:
            self.logger.error(f"音频生成失败 (UUID: {model_input.get('uuid', 'unknown')}): {str(e)}")
            raise