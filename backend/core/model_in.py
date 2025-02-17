# core/model_in.py
import logging
import asyncio
import torch
import numpy as np
import librosa
from typing import List, Optional

from services.cosyvoice.client import CosyVoiceClient
from utils import concurrency

class ModelIn:
    def __init__(self, cosyvoice_client: CosyVoiceClient, max_concurrent_tasks: int = 4):
        self.cosyvoice_client = cosyvoice_client
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

        self.speaker_cache = {}  # speaker_id => Features
        self.max_val = 0.8
        self.cosy_sample_rate = 24000

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440):
        """
        对音频进行trim、幅度归一化，并加0.2s静音
        """
        speech, _ = librosa.effects.trim(
            speech,
            top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if np.abs(speech).max() > self.max_val:
            speech = speech / np.abs(speech).max() * self.max_val

        # 结尾加0.2s静音
        pad_samples = int(self.cosy_sample_rate * 0.2)
        speech = np.concatenate([speech, np.zeros(pad_samples, dtype=speech.dtype)])
        return speech

    async def modelin_maker(self,
                            sentences: List,
                            reuse_speaker=False,
                            reuse_uuid=False,
                            batch_size=3):
        """
        给一批 sentence 做文本 + speaker 特征提取，并分批 yield
        """
        if not sentences:
            self.logger.warning("modelin_maker: 收到空句子列表")
            return

        tasks = []
        for s in sentences:
            tasks.append(asyncio.create_task(
                self._process_one_sentence_async(s, reuse_speaker, reuse_uuid)
            ))

        results_batch = []
        try:
            for i, task in enumerate(tasks, start=1):
                updated = await task
                if updated is not None:
                    results_batch.append(updated)
                if i % batch_size == 0:
                    yield results_batch
                    results_batch = []
            if results_batch:
                yield results_batch
        except Exception as e:
            self.logger.error(f"modelin_maker处理失败: {e}")
            raise
        finally:
            if not reuse_speaker:
                self.speaker_cache.clear()
                self.logger.debug("modelin_maker: 已清理 speaker_cache")

    async def _process_one_sentence_async(self, sentence, reuse_speaker, reuse_uuid):
        async with self.semaphore:
            return await concurrency.run_sync(
                self._process_one_sentence_sync, sentence, reuse_speaker, reuse_uuid
            )

    def _process_one_sentence_sync(self, sentence, reuse_speaker, reuse_uuid):
        """
        同步处理：提取文本特征 & speaker特征
        """
        # 1) speaker
        speaker_id = getattr(sentence, 'speaker_id', None)
        if speaker_id and not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                # postprocess
                audio_np = sentence.audio.squeeze(0).cpu().numpy() if isinstance(sentence.audio, torch.Tensor) else sentence.audio
                processed = self.postprocess(audio_np)
                processed_tensor = torch.from_numpy(processed).unsqueeze(0)

                speaker_feats = self.cosyvoice_client.extract_speaker_features(processed_tensor)
                self.speaker_cache[speaker_id] = speaker_feats

            sentence.model_input['speaker_features'] = self.speaker_cache[speaker_id]

        # 2) uuid
        if not reuse_uuid:
            sentence.model_input['uuid'] = ""

        # 3) 文本特征
        if 'text_features' not in sentence.model_input:
            text_feats = self.cosyvoice_client.normalize_text(sentence.trans_text or "")
            sentence.model_input['text_features'] = text_feats

        return sentence
