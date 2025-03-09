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
from utils.tensor_utils import ensure_cpu_tensors

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
        self.speaker_cache = {}
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
        import torch
        
        speaker_id = sentence.speaker_id

        # 1) Speaker处理
        if not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                try:
                    # 确保音频在CPU上
                    sentence = ensure_cpu_tensors(sentence, path="sentence")
                    audio = sentence.audio
                    
                    # 调用Actor处理音频
                    processed_audio = await concurrency.run_sync(
                        lambda: ray.get(self.cosyvoice_actor.postprocess_audio.remote(
                            audio, max_val=self.max_val
                        ))
                    )
                    
                    # 调用Actor提取说话人特征
                    self.speaker_cache[speaker_id] = await concurrency.run_sync(
                        lambda: ray.get(self.cosyvoice_actor.extract_speaker_features.remote(processed_audio))
                    )
                except RuntimeError as e:
                    error_msg = str(e)
                    if "torch.cuda.is_available() is False" in error_msg:
                        self.logger.error(f"CUDA设备不可用错误: {error_msg}")
                        # 尝试重新初始化cosyvoice_actor
                        self.logger.info("尝试重新获取采样率...")
                        self.cosy_sample_rate = ray.get(self.cosyvoice_actor.get_sample_rate.remote())
                        # 重试处理音频，确保使用CPU
                        sentence = ensure_cpu_tensors(sentence, path="sentence_retry")
                        audio = sentence.audio
                        
                        processed_audio = await concurrency.run_sync(
                            lambda: ray.get(self.cosyvoice_actor.postprocess_audio.remote(
                                audio, max_val=self.max_val
                            ))
                        )
                        
                        # 重试提取特征
                        self.speaker_cache[speaker_id] = await concurrency.run_sync(
                            lambda: ray.get(self.cosyvoice_actor.extract_speaker_features.remote(processed_audio))
                        )
                    elif "Input type" in error_msg and "weight type" in error_msg and "should be the same" in error_msg:
                        self.logger.error(f"设备不匹配错误: {error_msg}")
                        # 确保音频在CPU上
                        sentence = ensure_cpu_tensors(sentence, path="sentence_device_mismatch")
                        audio = sentence.audio
                        
                        # 重新处理音频
                        processed_audio = await concurrency.run_sync(
                            lambda: ray.get(self.cosyvoice_actor.postprocess_audio.remote(
                                audio, max_val=self.max_val
                            ))
                        )
                        
                        # 确保处理后的音频在CPU上
                        if isinstance(processed_audio, torch.Tensor) and processed_audio.is_cuda:
                            processed_audio = processed_audio.cpu()
                        
                        # 重试提取特征
                        self.speaker_cache[speaker_id] = await concurrency.run_sync(
                            lambda: ray.get(self.cosyvoice_actor.extract_speaker_features.remote(processed_audio))
                        )
                    else:
                        raise
                except TypeError as e:
                    if "can't convert cuda:0 device type tensor to numpy" in str(e):
                        self.logger.error(f"CUDA张量转换错误: {str(e)}")
                        # 确保音频在CPU上
                        sentence = ensure_cpu_tensors(sentence, path="sentence_type_error")
                        audio = sentence.audio
                        
                        processed_audio = await concurrency.run_sync(
                            lambda: ray.get(self.cosyvoice_actor.postprocess_audio.remote(
                                audio, max_val=self.max_val
                            ))
                        )
                        
                        # 重试提取特征
                        self.speaker_cache[speaker_id] = await concurrency.run_sync(
                            lambda: ray.get(self.cosyvoice_actor.extract_speaker_features.remote(processed_audio))
                        )
                    else:
                        raise
                
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
        类似于 translator_actor.translate_sentences，返回 ObjectRef
        """
        if not sentences:
            self.logger.warning("modelin_maker: 收到空的句子列表")
            return

        # 确保所有句子的张量都在CPU上
        sentences = ensure_cpu_tensors(sentences, path="modelin_sentences")
        self.logger.debug(f"已将所有张量移动到CPU，处理 {len(sentences)} 个句子")

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
