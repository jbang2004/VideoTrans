import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
import torch
import os

class TTSTokenGenerator:
    def __init__(self, cosyvoice_model, Hz=25, max_workers=None):
        self.cosyvoice_model = cosyvoice_model.model
        self.Hz = Hz
        cpu_count = os.cpu_count()
        self.max_workers = min(max_workers or cpu_count, cpu_count)
        self.executor = None
        self.logger = logging.getLogger(__name__)

    async def tts_token_maker(self, batch_sentences):
        """
        批量为句子生成 TTS token。若在此阶段出现错误，会抛出异常。
        """
        try:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                self.logger.debug(f"创建新的线程池，max_workers={self.max_workers}")

            # 并发执行每个句子的 TTS token 生成
            tasks = [
                self._generate_single_async(sentence)
                for sentence in batch_sentences
            ]
            await asyncio.gather(*tasks)

            # 检查生成结果
            for sentence in batch_sentences:
                if not sentence.model_input.get('segment_speech_tokens'):
                    self.logger.error(f"TTS token 生成失败: {sentence.trans_text}")

        except Exception as e:
            self.logger.error(f"TTS token 生成失败: {str(e)}")
            raise

    async def _generate_single_async(self, sentence):
        """
        异步执行单个句子的 TTS token 生成
        """
        loop = asyncio.get_event_loop()
        this_uuid = str(uuid.uuid1())
        await loop.run_in_executor(
            self.executor,
            self._generate_tts_single,
            sentence,
            this_uuid
        )

    def _generate_tts_single(self, sentence, this_uuid=None):
        """
        核心逻辑：对某个句子生成 TTS token（CosyVoice 相关流程）。
        修改后的部分: 不再在此处计算 sentence.diff
        """
        model_input = sentence.model_input
        main_uuid = this_uuid or model_input.get('uuid') or str(uuid.uuid1())

        try:
            segment_tokens_list = []
            segment_uuids = []
            total_token_count = 0

            # 遍历分段文本
            for i, (text, text_len) in enumerate(zip(model_input['text'], model_input['text_len'])):
                segment_uuid = f"{main_uuid}_seg_{i}"

                # 避免字典残留
                with self.cosyvoice_model.lock:
                    self.cosyvoice_model.tts_speech_token_dict[segment_uuid] = []
                    self.cosyvoice_model.llm_end_dict[segment_uuid] = False
                    if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                        self.cosyvoice_model.mel_overlap_dict[segment_uuid] = None
                    self.cosyvoice_model.hift_cache_dict[segment_uuid] = None

                # LLM job 生成 tokens
                self.cosyvoice_model.llm_job(
                    text,
                    model_input.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                    model_input.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                    model_input.get('llm_embedding', torch.zeros(0, 192)),
                    segment_uuid
                )

                # 取生成后的 tokens
                segment_tokens = self.cosyvoice_model.tts_speech_token_dict[segment_uuid]
                segment_tokens_list.append(segment_tokens)
                segment_uuids.append(segment_uuid)
                total_token_count += len(segment_tokens)

            # 计算合并后的音频时长(秒)
            total_duration = total_token_count / self.Hz

            # 保留音频时长(ms)，供后续对齐逻辑使用
            sentence.duration = total_duration * 1000

            # 不再计算 sentence.diff = ...
            # 因为我们要让 DurationAligner 统一处理 diff

            # 更新 model_input
            model_input['segment_speech_tokens'] = segment_tokens_list
            model_input['segment_uuids'] = segment_uuids
            model_input['uuid'] = main_uuid

            self.logger.info(
                f"TTS token 生成完成 (主UUID: {main_uuid}, 估计时长: {total_duration:.2f}s, 段落数: {len(segment_uuids)})"
            )

        except Exception as e:
            self.logger.error(f"生成失败 (UUID: {main_uuid}): {str(e)}")
            # 清理临时dict
            with self.cosyvoice_model.lock:
                for seg_uuid in segment_uuids:
                    self.cosyvoice_model.tts_speech_token_dict.pop(seg_uuid, None)
                    self.cosyvoice_model.llm_end_dict.pop(seg_uuid, None)
                    self.cosyvoice_model.hift_cache_dict.pop(seg_uuid, None)
                    if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                        self.cosyvoice_model.mel_overlap_dict.pop(seg_uuid, None)
            raise