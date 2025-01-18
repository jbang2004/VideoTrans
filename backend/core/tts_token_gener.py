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
        self.cosyvoice_model = cosyvoice_model.model
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
                if not sentence.model_input.get('segment_speech_tokens'):
                    self.logger.error(f"TTS token 生成失败: {sentence.trans_text}")
                    
        except Exception as e:
            self.logger.error(f"TTS token 生成失败: {str(e)}")
            raise

    async def _generate_single_async(self, sentence):
        """异步生成单个句子的 tts_speech_token"""
        loop = asyncio.get_event_loop()
        # 生成新的 UUID
        this_uuid = str(uuid.uuid1())
        await loop.run_in_executor(
            self.executor, 
            self._generate_tts_single, 
            sentence,
            this_uuid
        )

    def _generate_tts_single(self, sentence, this_uuid=None):
        """生成单个句子的 tts_speech_token
        
        Args:
            sentence: 句子对象
            this_uuid: 指定的 UUID，如果为 None 则使用句子现有的 UUID
        """
        model_input = sentence.model_input
        # 如果没有指定 UUID，则使用现有的
        main_uuid = this_uuid or model_input.get('uuid') or str(uuid.uuid1())
        
        try:
            # 初始化存储多段结果
            segment_tokens_list = []
            segment_uuids = []
            total_token_count = 0
            
            # 处理每段文本
            for i, (text, text_len) in enumerate(zip(model_input['text'], model_input['text_len'])):
                # 为每段生成独立的UUID
                segment_uuid = f"{main_uuid}_seg_{i}"
                
                with self.cosyvoice_model.lock:
                    self.cosyvoice_model.tts_speech_token_dict[segment_uuid] = []
                    self.cosyvoice_model.llm_end_dict[segment_uuid] = False
                    if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                        self.cosyvoice_model.mel_overlap_dict[segment_uuid] = None
                    self.cosyvoice_model.hift_cache_dict[segment_uuid] = None

                # 生成当前段的token
                self.cosyvoice_model.llm_job(
                    text,
                    model_input.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                    model_input.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                    model_input.get('llm_embedding', torch.zeros(0, 192)),
                    segment_uuid
                )
                
                # 获取当前段的tokens
                segment_tokens = self.cosyvoice_model.tts_speech_token_dict[segment_uuid]
                segment_tokens_list.append(segment_tokens)
                segment_uuids.append(segment_uuid)
                total_token_count += len(segment_tokens)

            # 计算总时长和差异
            total_duration = total_token_count / self.Hz
            target_duration = sentence.target_duration / 1000 if sentence.target_duration else total_duration  # 毫秒转秒
            diff = total_duration - target_duration

            # 更新 model_input
            model_input['segment_speech_tokens'] = segment_tokens_list
            model_input['segment_uuids'] = segment_uuids
            model_input['uuid'] = main_uuid
            
            # 更新 Sentence 对象
            sentence.duration = total_duration * 1000  # 单位：毫秒
            sentence.diff = diff * 1000  # 单位：毫秒

            self.logger.info(f"TTS token 生成完成 (主UUID: {main_uuid}, 预期时长: {total_duration:.2f}s, 段落数: {len(segment_uuids)})")

        except Exception as e:
            self.logger.error(f"生成失败 (UUID: {main_uuid}): {str(e)}")
            with self.cosyvoice_model.lock:
                # 清理所有相关的UUID数据
                for segment_uuid in segment_uuids:
                    common_dicts = ['tts_speech_token_dict', 'llm_end_dict', 'hift_cache_dict']
                    for dict_name in common_dicts:
                        getattr(self.cosyvoice_model, dict_name).pop(segment_uuid, None)
                    if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                        self.cosyvoice_model.mel_overlap_dict.pop(segment_uuid, None)
            raise