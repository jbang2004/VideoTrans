import os
import logging
import torch
import numpy as np
import librosa
from typing import List, Optional
import asyncio
import ray

# [NEW] 统一使用 concurrency.run_sync
from utils import concurrency

class ModelIn:
    def __init__(self, cosyvoice_model_actor):
        """
        Args:
            cosyvoice_model_actor: CosyVoice模型Actor引用
        """
        self.cosyvoice_actor = cosyvoice_model_actor
        self.logger = logging.getLogger(__name__)

        # 获取采样率（同步调用，只在初始化时执行一次）
        self.cosy_sample_rate = ray.get(self.cosyvoice_actor.get_sample_rate.remote())
        self.speaker_cache = {}
        self.max_val = 0.8

        self.semaphore = asyncio.Semaphore(4)  # max_concurrent_tasks=4
        
        self.logger.info(f"ModelIn initialized with CosyVoice Actor")

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

    async def _update_text_features_sync(self, sentence):
        """
        更新文本特征（现在调用Actor）
        """
        try:
            tts_text = sentence.trans_text
            
            # 调用Actor的normalize_text方法
            normalized_segments = await concurrency.run_sync(
                lambda: ray.get(self.cosyvoice_actor.normalize_text.remote(tts_text, split=True))
            )
            
            segment_tokens = []
            segment_token_lens = []

            for seg in normalized_segments:
                # 调用Actor提取文本token
                txt, txt_len = await concurrency.run_sync(
                    lambda s=seg: ray.get(self.cosyvoice_actor.extract_text_tokens.remote(s))
                )
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

    async def _process_sentence_sync(self, sentence, reuse_speaker=False, reuse_uuid=False):
        """
        处理单个句子（现在调用Actor）
        """
        speaker_id = sentence.speaker_id

        # 1) Speaker处理
        if not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                # 调用Actor处理音频
                processed_audio = await concurrency.run_sync(
                    lambda: ray.get(self.cosyvoice_actor.postprocess_audio.remote(
                        sentence.audio, max_val=self.max_val
                    ))
                )
                
                # 调用Actor提取说话人特征
                self.speaker_cache[speaker_id] = await concurrency.run_sync(
                    lambda: ray.get(self.cosyvoice_actor.extract_speaker_features.remote(processed_audio))
                )
                
            speaker_features = self.speaker_cache[speaker_id].copy()
            sentence.model_input = speaker_features

        # 2) UUID处理
        if not reuse_uuid:
            sentence.model_input['uuid'] = ""

        # 3) 文本特征更新
        await self._update_text_features_sync(sentence)
        return sentence

    async def _process_sentence_async(self, sentence, reuse_speaker=False, reuse_uuid=False):
        """在异步方法中，对单个 sentence 做同步处理"""
        async with self.semaphore:
            # 确保sentence是真实对象，不是协程
            if asyncio.iscoroutine(sentence):
                sentence = await sentence  # 等待协程完成
            
            # 然后执行处理
            return await self._process_sentence_sync(sentence, reuse_speaker, reuse_uuid)

    async def modelin_maker(self,
                            sentences,
                            reuse_speaker=False,
                            reuse_uuid=False,
                            batch_size=3):
        """
        对一批 sentences 做 model_in 处理，分批 yield
        """
        if not sentences:
            self.logger.warning("modelin_maker: 收到空的句子列表")
            return

        tasks = []
        for s in sentences:
            tasks.append(
                asyncio.create_task(
                    self._process_sentence_async(s, reuse_speaker, reuse_uuid)
                )
            )

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
            # 不复用speaker时，清空cache
            if not reuse_speaker:
                self.speaker_cache.clear()
                self.logger.debug("modelin_maker: 已清理 speaker_cache")
