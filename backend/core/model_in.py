import logging
import asyncio
import torch
import numpy as np
import librosa
from typing import List, Optional
import os
import uuid
import soundfile as sf

from services.cosyvoice.client import CosyVoiceClient

class ModelIn:
    def __init__(self, cosyvoice_client: CosyVoiceClient, max_concurrent_tasks: int = 4):
        self.cosyvoice_client = cosyvoice_client
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        # 用于缓存 speaker 特征提取任务，确保同一 speaker 只处理一次
        self.speaker_cache = {}  # speaker_id -> asyncio.Task
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

    async def get_speaker_features(self, speaker_id, sentence):
        """
        异步提取 speaker 特征，并将任务缓存，确保同一 speaker 只处理一次。
        如果同一 speaker 已有任务在执行，则等待任务完成后返回结果。
        """
        if speaker_id in self.speaker_cache:
            self.logger.info(f"等待 speaker_id={speaker_id} 的特征提取任务")
            return await self.speaker_cache[speaker_id]

        async def extract_features():
            self.logger.info(f"为 speaker_id={speaker_id} 提取新的特征")
            # 获取音频数据
            audio_np = (
                sentence.audio.squeeze(0).cpu().numpy()
                if isinstance(sentence.audio, torch.Tensor)
                else sentence.audio
            )
            processed = self.postprocess(audio_np)

            # 保存音频到本地（阻塞操作放入线程中执行）
            save_dir = "debug_audio"
            os.makedirs(save_dir, exist_ok=True)
            file_id = str(uuid.uuid4())[:8]
            audio_path = os.path.join(save_dir, f"{file_id}_speaker{speaker_id}.wav")
            await asyncio.to_thread(sf.write, audio_path, processed, 16000)

            processed_tensor = torch.from_numpy(processed).unsqueeze(0)
            # 提取 speaker 特征（阻塞操作放入线程中执行）
            speaker_feats = await asyncio.to_thread(self.cosyvoice_client.extract_speaker_features, processed_tensor)
            self.logger.info(
                f"特征提取完成: embedding长度={len(speaker_feats.embedding)}, "
                f"prompt_speech_feat长度={len(speaker_feats.prompt_speech_feat)}, "
                f"prompt_speech_token长度={len(speaker_feats.prompt_speech_token)}, "
                f"音频保存路径={audio_path}"
            )
            return speaker_feats

        # 创建任务并存入缓存
        task = asyncio.create_task(extract_features())
        self.speaker_cache[speaker_id] = task
        return await task

    async def _process_one_sentence_async(self, sentence, reuse_speaker, reuse_uuid):
        async with self.semaphore:
            speaker_id = getattr(sentence, 'speaker_id', None)
            self.logger.info("=== _process_one_sentence_async ===")
            self.logger.info(f"speaker_id: {speaker_id}, reuse_speaker: {reuse_speaker}")

            # 1) speaker 特征提取
            if speaker_id is not None and not reuse_speaker:
                speaker_feats = await self.get_speaker_features(speaker_id, sentence)
                sentence.model_input['speaker_features'] = speaker_feats
            else:
                if speaker_id is not None:
                    self.logger.info("跳过特征提取: reuse_speaker=True")
                else:
                    self.logger.info(f"跳过特征提取: speaker_id为None, {sentence.raw_text}")
            self.logger.info("================================")

            # 2) uuid 处理
            if not reuse_uuid:
                sentence.model_input['uuid'] = ""

            # 3) 文本特征提取
            if 'text_features' not in sentence.model_input:
                text_feats = await asyncio.to_thread(self.cosyvoice_client.normalize_text, sentence.trans_text or "")
                sentence.model_input['text_features'] = text_feats

            return sentence

    async def modelin_maker(self, sentences: List, reuse_speaker=False, reuse_uuid=False, batch_size=3):
        """
        对一批 sentence 进行文本与 speaker 特征提取，并按 batch_size 分批 yield 结果
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
                self.logger.debug("modelin_maker: 已清理 speaker_cache")
