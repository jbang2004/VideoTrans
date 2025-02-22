import logging
import asyncio
import torch
import numpy as np
import librosa
from typing import List, Dict
import os
import uuid
import soundfile as sf

from services.cosyvoice.client import CosyVoiceClient

class ModelIn:
    def __init__(self, cosyvoice_client: CosyVoiceClient, max_concurrent_tasks: int = 4):
        self.cosyvoice_client = cosyvoice_client
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        # 缓存 speaker 特征提取任务，确保同一 speaker 只处理一次
        self.speaker_cache: Dict[str, asyncio.Task] = {}  # speaker_id -> asyncio.Task
        self.max_val = 0.8
        self.cosy_sample_rate = 24000

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440):
        """
        对音频进行 trim、幅度归一化，并在结尾加 0.2s 静音
        """
        speech, _ = librosa.effects.trim(
            speech,
            top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if np.abs(speech).max() > self.max_val:
            speech = speech / np.abs(speech).max() * self.max_val

        # 结尾加 0.2s 静音
        pad_samples = int(self.cosy_sample_rate * 0.2)
        speech = np.concatenate([speech, np.zeros(pad_samples, dtype=speech.dtype)])
        return speech

    async def get_speaker_features(self, speaker_id: str, sentence) -> str:
        """
        异步提取 speaker 特征，并将任务缓存，
        确保同一 speaker 只处理一次。返回UUID。
        """
        if speaker_id in self.speaker_cache:
            return await self.speaker_cache[speaker_id]

        async def extract_features() -> str:
            # 获取音频数据
            audio_np = (
                sentence.audio.squeeze(0).cpu().numpy()
                if isinstance(sentence.audio, torch.Tensor)
                else sentence.audio
            )
            postprocessed_audio = self.postprocess(audio_np)
            postprocessed_tensor = torch.from_numpy(postprocessed_audio).unsqueeze(0)

            # 获取UUID并提取特征
            uuid = await asyncio.to_thread(
                self.cosyvoice_client.normalize_text,
                ""  # 空文本，只用于获取UUID
            )
            success = await asyncio.to_thread(
                self.cosyvoice_client.extract_speaker_features,
                uuid,
                postprocessed_tensor,
                self.cosy_sample_rate
            )
            if not success:
                raise Exception("提取说话人特征失败")
            return uuid

        task = asyncio.create_task(extract_features())
        self.speaker_cache[speaker_id] = task
        return await task

    async def _process_one_sentence_async(self, sentence, reuse_speaker: bool, reuse_uuid: bool):
        async with self.semaphore:
            speaker_id = getattr(sentence, 'speaker_id', None)
            
            # 处理UUID
            if not reuse_uuid:
                # 获取新的UUID
                uuid = await asyncio.to_thread(
                    self.cosyvoice_client.normalize_text,
                    sentence.trans_text or ""
                )
                sentence.model_input['uuid'] = uuid
            
            # speaker 特征提取
            if speaker_id is not None and not reuse_speaker:
                speaker_uuid = await self.get_speaker_features(speaker_id, sentence)
                sentence.model_input['speaker_uuid'] = speaker_uuid

            return sentence

    async def modelin_maker(self, sentences: List, reuse_speaker: bool = False, reuse_uuid: bool = False, batch_size: int = 3):
        """
        对一批 sentence 进行文本与 speaker 特征提取，
        并按 batch_size 分批 yield 结果。
        """
        if not sentences:
            self.logger.warning("modelin_maker: 收到空句子列表")
            return

        tasks = [
            asyncio.create_task(self._process_one_sentence_async(s, reuse_speaker, reuse_uuid))
            for s in sentences
        ]

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
