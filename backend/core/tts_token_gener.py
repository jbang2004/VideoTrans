import logging
import asyncio
import uuid
import torch
import os

# [NEW] import concurrency
from utils import concurrency

class TTSTokenGenerator:
    def __init__(self, cosyvoice_model, Hz=25, max_workers=None):
        """
        Args:
            cosyvoice_model: CosyVoice model wrapper
            Hz: token频率
            max_workers: (废弃, 改统一线程池)
        """
        self.cosyvoice_model = cosyvoice_model.model
        self.Hz = Hz
        self.logger = logging.getLogger(__name__)

    async def tts_token_maker(self, sentences, reuse_uuid=False):
        """
        并发为句子生成 TTS token，不再自建线程池，而是统一 run_sync.
        """
        try:
            tasks = []
            for s in sentences:
                current_uuid = (
                    s.model_input.get('uuid') if reuse_uuid and s.model_input.get('uuid')
                    else str(uuid.uuid1())
                )
                # 创建异步任务
                tasks.append(asyncio.create_task(
                    self._generate_tts_single_async(s, current_uuid)
                ))

            processed = await asyncio.gather(*tasks)

            for sen in processed:
                if not sen.model_input.get('segment_speech_tokens'):
                    self.logger.error(f"TTS token 生成失败: {sen.trans_text}")

            return processed

        except Exception as e:
            self.logger.error(f"TTS token 生成失败: {e}")
            raise

    async def _generate_tts_single_async(self, sentence, main_uuid):
        """
        异步：实际调用 _generate_tts_single 同步逻辑 => concurrency.run_sync
        """
        return await concurrency.run_sync(self._generate_tts_single, sentence, main_uuid)

    def _generate_tts_single(self, sentence, main_uuid):
        """
        同步核心逻辑：对 sentence 做 TTS token 生成
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

                # 调用 cosyvoice_model.llm_job( ... ) 同步
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

            self.logger.debug(
                f"TTS token 生成完成 (UUID={main_uuid}, 时长={total_duration_s:.2f}s, "
                f"段数={len(segment_uuids)})"
            )
            return sentence

        except Exception as e:
            self.logger.error(f"生成失败 (UUID={main_uuid}): {e}")
            # 清理
            with self.cosyvoice_model.lock:
                for seg_uuid in segment_uuids:
                    self.cosyvoice_model.tts_speech_token_dict.pop(seg_uuid, None)
                    self.cosyvoice_model.llm_end_dict.pop(seg_uuid, None)
                    self.cosyvoice_model.hift_cache_dict.pop(seg_uuid, None)
                    if hasattr(self.cosyvoice_model, 'mel_overlap_dict'):
                        self.cosyvoice_model.mel_overlap_dict.pop(seg_uuid, None)
            raise