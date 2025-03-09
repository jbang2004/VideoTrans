import os
import logging
import torch
import numpy as np
import librosa
from typing import List, Optional, AsyncGenerator
import asyncio
import ray

# [NEW] 统一使用 concurrency.run_sync
from utils import concurrency

@ray.remote(num_gpus=0.1)  # 分配0.1个GPU资源
class ModelInActor:
    def __init__(self, cosyvoice_model_actor):
        """
        Args:
            cosyvoice_model_actor: CosyVoice模型Actor引用
        """
        self.cosyvoice_actor = cosyvoice_model_actor
        self.logger = logging.getLogger(__name__)

        # 获取采样率（同步调用，只在初始化时执行一次）
        self.cosy_sample_rate = ray.get(self.cosyvoice_actor.get_sample_rate.remote())
        self.speaker_cache = {}  # 存储speaker_id到特征缓存ID的映射
        self.max_val = 0.8

        self.semaphore = asyncio.Semaphore(4)  # max_concurrent_tasks=4
        
        self.logger.info(f"ModelInActor initialized with CosyVoice Actor")

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
        更新文本特征（使用Actor缓存）
        """
        try:
            tts_text = sentence.trans_text
            
            # 调用Actor提取文本token并缓存
            text_feature_id, normalized_segments = await concurrency.run_sync(
                lambda: ray.get(self.cosyvoice_actor.extract_text_tokens_and_cache.remote(tts_text))
            )
            
            # 保存文本特征ID
            sentence.model_input['text_feature_id'] = text_feature_id
            sentence.model_input['normalized_text_segments'] = normalized_segments

            self.logger.debug(f"成功更新文本特征: {normalized_segments}")
            return sentence
        except Exception as e:
            self.logger.error(f"更新文本特征失败: {str(e)}")
            raise

    async def _process_sentence_sync(self, sentence, reuse_speaker=False):
        """
        处理单个句子（使用Actor缓存）
        """
        speaker_id = sentence.speaker_id

        # 1) Speaker处理
        if not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                try:
                    # 处理音频并缓存
                    audio = sentence.audio
                    
                    # 调用Actor处理音频并缓存
                    processed_audio_id = await concurrency.run_sync(
                        lambda: ray.get(self.cosyvoice_actor.process_and_cache_audio.remote(
                            audio, max_val=self.max_val
                        ))
                    )
                    
                    # 调用Actor提取说话人特征并缓存
                    speaker_feature_id = await concurrency.run_sync(
                        lambda: ray.get(self.cosyvoice_actor.extract_speaker_features_and_cache.remote(processed_audio_id))
                    )
                    
                    # 保存特征ID到本地缓存
                    self.speaker_cache[speaker_id] = speaker_feature_id
                    
                except Exception as e:
                    self.logger.error(f"处理音频或提取特征失败: {str(e)}")
                    raise
                
            # 保存speaker_feature_id到sentence
            sentence.model_input['speaker_feature_id'] = self.speaker_cache[speaker_id]

        # 3) 文本特征更新
        await self._update_text_features_sync(sentence)
        return sentence

    async def _process_sentence_async(self, sentence, reuse_speaker=False):
        """在异步方法中，对单个 sentence 做同步处理"""
        async with self.semaphore:
            # 确保sentence是真实对象，不是协程
            if asyncio.iscoroutine(sentence):
                sentence = await sentence  # 等待协程完成
            
            # 然后执行处理
            return await self._process_sentence_sync(sentence, reuse_speaker)

    async def modelin_maker(self,
                            sentences,
                            reuse_speaker=False,
                            batch_size=3):
        """
        对一批 sentences 做 model_in 处理，分批 yield
        类似于 translator_actor.translate_sentences，返回 ObjectRef
        """
        if not sentences:
            self.logger.warning("modelin_maker: 收到空的句子列表")
            return

        # 不再需要ensure_cpu_tensors
        self.logger.debug(f"处理 {len(sentences)} 个句子")

        tasks = []
        for s in sentences:
            tasks.append(
                asyncio.create_task(
                    self._process_sentence_async(s, reuse_speaker)
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
            # 不再清理说话人特征，只在需要时清空本地缓存
            if not reuse_speaker:
                # 只清空本地缓存映射，不清理Actor中的特征
                self.speaker_cache.clear()
                self.logger.debug("modelin_maker: 已清理本地speaker_cache映射（不清理特征）")
