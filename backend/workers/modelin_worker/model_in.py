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
        self.speaker_cache: Dict[str, asyncio.Task] = {}  # speaker_id -> asyncio.Task
        self.max_val = 0.8
        self.cosy_sample_rate = 24000

    def postprocess(self, speech, sr, top_db=60, hop_length=220, win_length=440):
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

        # 使用 sr 采样率计算静音尾音的长度
        # 因为我们期望输入的音频采样率为 sr
        pad_samples = int(sr * 0.2)  # 0.2秒的静音
        speech = np.concatenate([speech, np.zeros(pad_samples, dtype=speech.dtype)])
        return speech

    async def get_speaker_features(self, speaker_id: str, sentence) -> str:
        """
        异步提取说话人特征，返回说话人UUID
        """
        if speaker_id in self.speaker_cache:
            return await self.speaker_cache[speaker_id]

        async def extract_features() -> str:
            # 从文件加载音频
            if not hasattr(sentence, 'audio_path') or not sentence.audio_path:
                raise ValueError(f"缺少音频路径 (audio_path)，无法提取说话人特征: speaker_id={speaker_id}")
            audio_path = sentence.audio_path
                
            # 从文件路径加载音频
            audio_np, sr = sf.read(audio_path)
            
            # 处理音频，得到numpy array
            postprocessed_audio = self.postprocess(audio_np, sr)

            speaker_uuid = str(uuid.uuid4())
            success = await asyncio.to_thread(
                self.cosyvoice_client.extract_speaker_features,
                speaker_uuid,
                postprocessed_audio,  # 直接传递numpy array
                16000  # 使用 16000Hz 作为采样率参数
            )
            if not success:
                raise Exception("提取说话人特征失败")
            return speaker_uuid

        task = asyncio.create_task(extract_features())
        self.speaker_cache[speaker_id] = task
        return await task

    async def _process_one_sentence_async(self, sentence, reuse_speaker: bool):
        async with self.semaphore:
            # 获取 speaker_id
            speaker_id = getattr(sentence, 'speaker_id', None)
            
            # 处理文本UUID
            trans_text = sentence.trans_text or ""
                
            text_uuid = await asyncio.to_thread(
                self.cosyvoice_client.normalize_text,
                trans_text
            )
            
            # 更新 model_input
            sentence.model_input['text_uuid'] = text_uuid
            
            # 处理说话人UUID
            if speaker_id is not None and not reuse_speaker:
                speaker_uuid = await self.get_speaker_features(speaker_id, sentence)
                sentence.model_input['speaker_uuid'] = speaker_uuid
            elif 'speaker_uuid' not in sentence.model_input:
                raise ValueError("缺少说话人UUID且未提取新特征")

            return sentence

    async def modelin_maker(self, sentences: List, reuse_speaker: bool = False, batch_size: int = 3):
        """
        对一批 sentence 进行文本与说话人特征提取，按 batch_size 分批 yield 结果
        """
        if not sentences:
            self.logger.warning("modelin_maker: 收到空句子列表")
            return

        tasks = [
            asyncio.create_task(self._process_one_sentence_async(s, reuse_speaker))
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