import logging
import asyncio
import torch
import numpy as np
import librosa
from typing import List, Optional

# 用于在异步方法中调用同步函数
from utils import concurrency

class ModelIn:
    def __init__(self, cosyvoice_client, max_concurrent_tasks: int = 4):
        """
        Args:
            cosyvoice_client: 连接 gRPC 的客户端封装 (CosyVoiceClient)
            max_concurrent_tasks: 同时处理多少个 sentence
        """
        self.cosyvoice_client = cosyvoice_client
        # 这里可从 client 或者自行固定
        self.cosy_sample_rate = 24000

        self.logger = logging.getLogger(__name__)

        self.speaker_cache = {}
        self.max_val = 0.8

        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

        self.logger.info(
            f"ModelIn initialized (max_concurrent_tasks={max_concurrent_tasks})"
        )

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440):
        """
        与原先相同，对音频进行trim和幅度归一化，并在结尾加0.2s静音
        """
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > self.max_val:
            speech = speech / speech.abs().max() * self.max_val

        # 结尾添加0.2s静音
        pad_samples = int(self.cosy_sample_rate * 0.2)
        # 这里 speech 可能是 2D Tensor: shape=[1, time]
        # 如果是一维，可以先确保维度
        if speech.ndim == 1:
            speech = speech.unsqueeze(0)
        speech = torch.cat([speech, torch.zeros((1, pad_samples), dtype=speech.dtype)], dim=1)

        return speech

    def _update_text_features_sync(self, sentence):
        """同步：调用gRPC的normalize_text，更新sentence.model_input"""
        try:
            tts_text = sentence.trans_text
            result = self.cosyvoice_client.normalize_text(tts_text)

            # 直接更新model_input，client已经处理好了数据格式
            sentence.model_input.update(result)

            self.logger.debug(f"成功更新文本特征: {sentence.model_input['normalized_text_segments']}")
            return sentence
        except Exception as e:
            self.logger.error(f"更新文本特征失败: {str(e)}")
            raise

    def _process_sentence_sync(self, sentence, reuse_speaker=False, reuse_uuid=False):
        """同步处理单个sentence的speaker特征和文本特征"""
        speaker_id = sentence.speaker_id

        # 1) 处理speaker特征
        if not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                try:
                    # 音频预处理
                    processed_audio = self.postprocess(sentence.audio)
                    
                    # 提取特征 - 直接使用原始特征
                    features = self.cosyvoice_client.extract_speaker_features(processed_audio)
                    self.speaker_cache[speaker_id] = features
                    
                except Exception as e:
                    self.logger.error(f"处理说话人特征失败 (speaker_id={speaker_id}): {str(e)}")
                    raise

            speaker_features = self.speaker_cache[speaker_id]
            
            # 直接使用原始特征
            sentence.model_input.update(speaker_features)
            
            # 准备prompt_text (空)
            sentence.model_input['prompt_text'] = []
            sentence.model_input['prompt_text_len'] = 0
            
            self.logger.debug(f"成功更新speaker特征")

        # 2) 处理uuid
        if not reuse_uuid:
            sentence.model_input['uuid'] = ""

        # 3) 文本特征更新 - 使用_update_text_features_sync
        sentence = self._update_text_features_sync(sentence)
        
        return sentence

    async def _process_sentence_async(self, sentence, reuse_speaker=False, reuse_uuid=False):
        """
        在异步场景下，对单个 sentence 做同步处理 => concurrency.run_sync
        """
        async with self.semaphore:
            return await concurrency.run_sync(
                self._process_sentence_sync,
                sentence, reuse_speaker, reuse_uuid
            )

    async def modelin_maker(self,
                            sentences,
                            reuse_speaker=False,
                            reuse_uuid=False,
                            batch_size=3):
        """
        对一批 sentences 进行 model_in 处理（文本+speaker），分批 yield
        """
        if not sentences:
            self.logger.warning("modelin_maker: 收到空的句子列表")
            return

        tasks = []
        for s in sentences:
            task = asyncio.create_task(
                self._process_sentence_async(s, reuse_speaker, reuse_uuid)
            )
            tasks.append(task)

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
            # 若不复用speaker，则最后清理缓存
            if not reuse_speaker:
                self.speaker_cache.clear()
                self.logger.debug("modelin_maker: 已清理 speaker_cache")
