import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
import torch
import os

class TTSTokenGenerator:
    def __init__(self, cosyvoice_model, Hz=25, max_workers=None):
        """
        Args:
            cosyvoice_model: CosyVoice模型
            Hz: 采样率
            max_workers: 并行处理的最大工作线程数，默认为None（将根据CPU核心数自动设置）
        """
        self.cosyvoice_model = cosyvoice_model
        self.Hz = Hz
        # 获取CPU核心数
        cpu_count = os.cpu_count()
        # 如果未指定max_workers，则使用CPU核心数
        self.max_workers = min(max_workers or cpu_count, cpu_count)
        self.executor = None  # 初始化为 None，等待第一次使用时创建
        self.logger = logging.getLogger(__name__)

    async def tts_token_maker(self, batch_sentences):
        """异步批量生成 tts_speech_token"""
        try:
            # 如果 executor 不存在，创建一个新的
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                self.logger.debug(f"创建新的线程池，max_workers={self.max_workers}")
            
            tasks = [
                self._generate_single_async(sentence)
                for sentence in batch_sentences
            ]
            await asyncio.gather(*tasks)
            
            # 验证生成结果
            for sentence in batch_sentences:
                if not sentence.model_input.get('tts_speech_token'):
                    self.logger.error(f"TTS token 生成失败: {sentence.trans_text}")
                    
        except Exception as e:
            self.logger.error(f"TTS token 生成失败: {str(e)}")
            raise

    async def _generate_single_async(self, sentence):
        """异步生成单个句子的 tts_speech_token"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor, 
            self._generate_tts_single, 
            sentence
        )

    def _generate_tts_single(self, sentence):
        """生成单个句子的 tts_speech_token"""
        model_input = sentence.model_input
        this_uuid = str(uuid.uuid1())
        
        try:
            with self.cosyvoice_model.lock:
                self.cosyvoice_model.tts_speech_token_dict[this_uuid] = []
                self.cosyvoice_model.llm_end_dict[this_uuid] = False
                if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                    self.cosyvoice_model.mel_overlap_dict[this_uuid] = None
                self.cosyvoice_model.hift_cache_dict[this_uuid] = None

            self.cosyvoice_model.llm_job(
                model_input['text'],
                model_input.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                model_input.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                model_input.get('llm_embedding', torch.zeros(0, 192)),
                this_uuid
            )

            token_count = len(self.cosyvoice_model.tts_speech_token_dict[this_uuid])
            duration = token_count / self.Hz
            target_duration = sentence.target_duration / 1000 if sentence.target_duration else duration  # 毫秒转秒
            diff = duration - target_duration

            # 更新 model_input
            model_input['tts_speech_token'] = self.cosyvoice_model.tts_speech_token_dict[this_uuid]
            model_input['uuid'] = this_uuid
            
            # 更新 Sentence 对象
            sentence.duration = duration * 1000  # 单位：毫秒
            sentence.diff = diff * 1000  # 单位：毫秒

            self.logger.info(f"TTS token 生成完成 (UUID: {this_uuid}, 时长: {duration:.2f}s)")

        except Exception as e:
            self.logger.error(f"生成失败 (UUID: {this_uuid}): {str(e)}")
            with self.cosyvoice_model.lock:
                common_dicts = ['tts_speech_token_dict', 'llm_end_dict', 'hift_cache_dict']
                for dict_name in common_dicts:
                    getattr(self.cosyvoice_model, dict_name).pop(this_uuid, None)
                if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                    self.cosyvoice_model.mel_overlap_dict.pop(this_uuid, None)
            raise