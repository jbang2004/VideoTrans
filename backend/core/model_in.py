import os
import logging
import torch
import numpy as np
import librosa
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ModelIn:
    def __init__(self, cosy_model,
                 max_workers: Optional[int] = None,
                 max_concurrent_tasks: int = 4):
        self.cosy_frontend = cosy_model.frontend
        self.cosy_sample_rate = cosy_model.sample_rate
        self.logger = logging.getLogger(__name__)

        self.speaker_cache = {}
        self.max_val = 0.8

        cpu_count = os.cpu_count() or 1
        self.max_workers = min(max_workers or cpu_count, cpu_count)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

        self.logger.info(
            f"ModelIn initialized with max_workers={self.max_workers}, "
            f"max_concurrent_tasks={max_concurrent_tasks}"
        )

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440):
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > self.max_val:
            speech = speech / speech.abs().max() * self.max_val
        
        speech = torch.concat([speech, torch.zeros(1, int(self.cosy_sample_rate * 0.2))], dim=1)
        return speech

    def _update_text_features_sync(self, sentence):
        try:
            tts_text = sentence.trans_text
            normalized_segments = self.cosy_frontend.text_normalize(tts_text, split=True)

            segment_tokens = []
            segment_token_lens = []

            for seg in normalized_segments:
                txt, txt_len = self.cosy_frontend._extract_text_token(seg)
                segment_tokens.append(txt)
                segment_token_lens.append(txt_len)

            sentence.model_input['text'] = segment_tokens
            sentence.model_input['text_len'] = segment_token_lens
            sentence.model_input['normalized_text_segments'] = normalized_segments

            self.logger.debug(f"成功更新文本特征: {normalized_segments}")
            return sentence
        except Exception as e:
            self.logger.error(f"更新文本特征失败: {str(e)}")
            raise

    def _process_sentence_sync(self, sentence, reuse_speaker=False, reuse_uuid=False):
        speaker_id = sentence.speaker_id

        # 1) Speaker处理
        if not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                processed_audio = self.postprocess(sentence.audio)
                self.speaker_cache[speaker_id] = self.cosy_frontend.frontend_cross_lingual(
                    "",
                    processed_audio,
                    self.cosy_sample_rate
                )
            speaker_features = self.speaker_cache[speaker_id].copy()
            sentence.model_input = speaker_features

        # 2) UUID处理
        if not reuse_uuid:
            sentence.model_input['uuid'] = ""

        # 3) 更新文本特征
        self._update_text_features_sync(sentence)
        return sentence

    async def _process_sentence_async(self, sentence, reuse_speaker=False, reuse_uuid=False):
        loop = asyncio.get_event_loop()
        async with self.semaphore:
            return await loop.run_in_executor(
                self.executor,
                self._process_sentence_sync,
                sentence,
                reuse_speaker,
                reuse_uuid
            )

    async def modelin_maker(self,
                            sentences,
                            reuse_speaker=False,
                            reuse_uuid=False,
                            batch_size=3):
        if not sentences:
            self.logger.warning("modelin_maker: 收到空的句子列表")
            return

        tasks = [
            asyncio.create_task(
                self._process_sentence_async(s, reuse_speaker, reuse_uuid)
            )
            for s in sentences
        ]

        try:
            results = []
            for i, task in enumerate(tasks, start=1):
                updated_sentence = await task
                results.append(updated_sentence)

                if i % batch_size == 0:
                    yield results
                    results = []

            if results:
                yield results

        except Exception as e:
            self.logger.error(f"modelin_maker处理失败: {str(e)}")
            raise

        finally:
            if not reuse_speaker:
                self.speaker_cache.clear()
                self.logger.debug("modelin_maker: 已清理 speaker_cache")
