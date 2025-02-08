import os
import logging
import torch
import numpy as np
import librosa
from typing import List, Optional
import asyncio
from .concurrency import run_sync
from shared.sentence_tools import Sentence

logger = logging.getLogger(__name__)

class ModelIn:
    """
    负责对句子进行文本正则化、token 提取、speaker 处理等，
    参考 backend/core/model_in.py 的实现。
    """
    def __init__(self, cosy_model, max_concurrent_tasks: int = 4):
        self.cosy_frontend = cosy_model.frontend
        self.cosy_sample_rate = cosy_model.sample_rate
        self.logger = logging.getLogger(__name__)
        self.speaker_cache = {}
        self.max_val = 0.8
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.logger.info(f"ModelIn initialized (max_concurrent_tasks={max_concurrent_tasks})")

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440):
        speech, _ = librosa.effects.trim(speech, top_db=top_db, frame_length=win_length, hop_length=hop_length)
        if speech.abs().max() > self.max_val:
            speech = speech / speech.abs().max() * self.max_val
        speech = torch.cat([speech, torch.zeros(1, int(self.cosy_sample_rate * 0.2))], dim=1)
        return speech

    def _update_text_features_sync(self, sentence: Sentence):
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
            self.logger.debug(f"Updated text features: {normalized_segments}")
            return sentence
        except Exception as e:
            self.logger.error(f"Failed to update text features: {e}")
            raise

    def _process_sentence_sync(self, sentence: Sentence, reuse_speaker=False, reuse_uuid=False):
        speaker_id = sentence.speaker_id
        if not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                processed_audio = self.postprocess(sentence.audio)
                self.speaker_cache[speaker_id] = self.cosy_frontend.frontend_cross_lingual("", processed_audio, self.cosy_sample_rate)
            speaker_features = self.speaker_cache[speaker_id].copy()
            sentence.model_input = speaker_features
        if not reuse_uuid:
            sentence.model_input['uuid'] = ""
        self._update_text_features_sync(sentence)
        return sentence

    async def _process_sentence_async(self, sentence: Sentence, reuse_speaker=False, reuse_uuid=False):
        async with self.semaphore:
            return await run_sync(self._process_sentence_sync, sentence, reuse_speaker, reuse_uuid)

    async def modelin_maker(self, sentences: List[Sentence], reuse_speaker=False, reuse_uuid=False, batch_size: int = 3):
        if not sentences:
            self.logger.warning("ModelIn: Empty sentence list")
            return
        tasks = [asyncio.create_task(self._process_sentence_async(s, reuse_speaker, reuse_uuid)) for s in sentences]
        results = []
        for i, task in enumerate(tasks, start=1):
            updated_sentence = await task
            results.append(updated_sentence)
            if i % batch_size == 0:
                yield results
                results = []
        if results:
            yield results
        if not reuse_speaker:
            self.speaker_cache.clear()
            self.logger.debug("Cleared speaker cache after modelin_maker")
