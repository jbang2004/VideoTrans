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

    async def tts_token_maker(self, sentences, reuse_uuid=False):
        """
        并发为句子生成 TTS token。
        :param reuse_uuid: 若为 True，则复用句子已有的 uuid；若没有，则自动生成。
                           若为 False，则为每个句子生成新的 uuid。
        """
        try:
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                self.logger.debug(f"创建线程池, max_workers={self.max_workers}")

            loop = asyncio.get_event_loop()
            tasks = []

            for s in sentences:
                current_uuid = (
                    s.model_input.get('uuid') if reuse_uuid and s.model_input.get('uuid')
                    else str(uuid.uuid1())
                )
                task = loop.run_in_executor(
                    self.executor,
                    self._generate_tts_single,
                    s,
                    current_uuid
                )
                tasks.append(task)

            processed = await asyncio.gather(*tasks)

            # 检查生成结果
            for sen in processed:
                if not sen.model_input.get('segment_speech_tokens'):
                    self.logger.error(f"TTS token 生成失败: {sen.trans_text}")

            return processed

        except Exception as e:
            self.logger.error(f"TTS token 生成失败: {e}")
            raise

    def _generate_tts_single(self, sentence, main_uuid):
        """
        同步方法, 生成分段 tokens, 计算时长
        """
        model_input = sentence.model_input
        segment_tokens_list = []
        segment_uuids = []
        total_token_count = 0

        try:
            for i, (text, text_len) in enumerate(zip(model_input['text'], model_input['text_len'])):
                seg_uuid = f"{main_uuid}_seg_{i}"
                with self.cosyvoice_model.lock:
                    self.cosyvoice_model.tts_speech_token_dict[seg_uuid] = []
                    self.cosyvoice_model.llm_end_dict[seg_uuid] = False
                    if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                        self.cosyvoice_model.mel_overlap_dict[seg_uuid] = None
                    self.cosyvoice_model.hift_cache_dict[seg_uuid] = None

                # LLM job 生成 tokens
                self.cosyvoice_model.llm_job(
                    text,
                    model_input.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32)),
                    model_input.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                    model_input.get('llm_embedding', torch.zeros(0, 192)),
                    seg_uuid
                )

                seg_tokens = self.cosyvoice_model.tts_speech_token_dict[seg_uuid]
                segment_tokens_list.append(seg_tokens)
                segment_uuids.append(seg_uuid)
                total_token_count += len(seg_tokens)

            total_duration_s = total_token_count / self.Hz
            sentence.duration = total_duration_s * 1000

            model_input['segment_speech_tokens'] = segment_tokens_list
            model_input['segment_uuids'] = segment_uuids
            model_input['uuid'] = main_uuid

            self.logger.info(
                f"TTS token 生成完成 (UUID={main_uuid}, 时长={total_duration_s:.2f}s, 段数={len(segment_uuids)})"
            )
            return sentence

        except Exception as e:
            self.logger.error(f"生成失败 (UUID={main_uuid}): {e}")
            # 失败时, 清理缓存
            with self.cosyvoice_model.lock:
                for seg_uuid in segment_uuids:
                    self.cosyvoice_model.tts_speech_token_dict.pop(seg_uuid, None)
                    self.cosyvoice_model.llm_end_dict.pop(seg_uuid, None)
                    self.cosyvoice_model.hift_cache_dict.pop(seg_uuid, None)
                    if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                        self.cosyvoice_model.mel_overlap_dict.pop(seg_uuid, None)
            raise
