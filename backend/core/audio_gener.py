import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import os
import inspect  # 导入 inspect 模块

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
        # 获取CPU核心数
        cpu_count = os.cpu_count()
        # 如果未指定max_workers，则使用CPU核心数
        self.max_workers = min(max_workers or cpu_count, cpu_count)
        self.executor = None  # 延迟初始化executor
        self.logger = logging.getLogger(__name__)

    async def vocal_audio_maker(self, batch_sentences):
        """异步批量生成音频"""
        try:
            # 延迟初始化executor
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
            self.logger.error(f"音频生成失败 (UUID: {sentence.model_input['uuid']}): {str(e)}")
            sentence.generated_audio = None  # 确保失败时设置为 None

    def _generate_audio_single(self, sentence):
        """生成单个音频"""
        model_input = sentence.model_input
        self.logger.info(f"开始生成音频 (UUID: {model_input['uuid']})")
        
        try:
            # 检查 tts_speech_token 是否为空
            if model_input['tts_speech_token'] is None or len(model_input['tts_speech_token']) == 0:
                self.logger.info(f"空的语音标记，创建空音频 (UUID: {model_input['uuid']})")
                speech_output = torch.zeros(1, 0)  # 创建空的2维张量
            else:
                token2wav_kwargs = {
                    'token': torch.tensor(model_input['tts_speech_token']).unsqueeze(dim=0),
                    'token_offset': 0,  # 显式设置为0
                    'finalize': True,  # 确保最终生成
                    'prompt_token': model_input.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                    'prompt_feat': model_input.get('prompt_speech_feat', torch.zeros(1, 0, 80)),
                    'embedding': model_input.get('flow_embedding', torch.zeros(0)),
                    'uuid': model_input['uuid'],
                    'speed': sentence.speed if sentence.speed else 1.0
                }

                speech_output = self.cosyvoice_model.token2wav(**token2wav_kwargs)
            # 处理静音
            if sentence.is_first and sentence.start > 0:
                speech_output = self._add_silence(speech_output, sentence.start, 'before')  # 单位：毫秒

            if sentence.silence_duration > 0:
                speech_output = self._add_silence(speech_output, sentence.silence_duration, 'after') # 单位：毫秒

            # 确保音频数据为1维并在CPU上
            audio_np = speech_output.cpu().numpy()
            if audio_np.ndim > 1:
                # 平均声道以转换为单声道
                audio_np = audio_np.mean(axis=0)

            
            return audio_np

        except Exception as e:
            self.logger.error(f"音频生成失败 (UUID: {model_input['uuid']}): {str(e)}")
            raise

    def _add_silence(self, audio, silence_duration_ms, position='before'):
        """添加静音"""
        silence_samples = int(silence_duration_ms / 1000 * self.sample_rate) # 毫秒转换为秒再计算样本数
        silence = torch.zeros(1, silence_samples, device=audio.device)
        
        # 处理空音频的情况
        if audio is None or (isinstance(audio, torch.Tensor) and audio.numel() == 0):
            audio = torch.zeros(1, 0, device=silence.device)
        elif isinstance(audio, torch.Tensor) and audio.dim() == 1:
            # 如果是1维张量，转换为2维 (1, length)
            audio = audio.unsqueeze(0)
        
        if position == 'before':
            return torch.concat((silence, audio), dim=1)
        elif position == 'after':
            return torch.concat((audio, silence), dim=1)
        else:
            raise ValueError("position 必须是 'before' 或 'after'")