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
        """
        Args:
            cosy_model: 带有 frontend 等属性的cosyvoice对象
            max_workers: 线程池线程数，不指定时默认为 CPU 核心数
            max_concurrent_tasks: 同时并发处理的句子数上限 (Semaphore)
        """
        self.cosy_frontend = cosy_model.frontend
        self.cosy_sample_rate = cosy_model.sample_rate
        self.logger = logging.getLogger(__name__)

        self.speaker_cache = {}
        self.max_val = 0.8  # 最大音量阈值

        cpu_count = os.cpu_count() or 1
        self.max_workers = min(max_workers or cpu_count, cpu_count)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

        self.logger.info(
            f"ModelIn initialized with max_workers={self.max_workers}, "
            f"max_concurrent_tasks={max_concurrent_tasks}"
        )

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440):
        """
        音频后处理: 去除多余静音, 音量限制, 拼接短静音等
        """
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > self.max_val:
            speech = speech / speech.abs().max() * self.max_val
        
        # 拼接 0.2s 静音
        speech = torch.concat([speech, torch.zeros(1, int(self.cosy_sample_rate * 0.2))], dim=1)
        return speech

    def _update_text_features_sync(self, sentence):
        """
        同步方法: 更新文本特征到 sentence.model_input
        """
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
        """
        同步核心方法, 通过布尔参数控制:
          1) reuse_speaker=False => 如果 speaker_id 不在缓存中, 则执行 postprocess + frontend_cross_lingual
          2) reuse_uuid=False    => 重置 model_input['uuid'] = "", 适合“首次处理”
             reuse_uuid=True     => 保留/复用原来的 UUID

        整个流程:
          - 如果不复用 speaker, 检查 speaker_cache, 做音频处理 + 前端特征
          - 根据 reuse_uuid 决定是否重置 sentence.model_input['uuid']
          - 更新文本特征
        """
        speaker_id = sentence.speaker_id

        # 1) Speaker处理
        if not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                # 首次处理或无缓存时
                processed_audio = self.postprocess(sentence.audio)
                self.speaker_cache[speaker_id] = self.cosy_frontend.frontend_cross_lingual(
                    "",  # 空字符串占位
                    processed_audio,
                    self.cosy_sample_rate
                )
            # 将speaker缓存复制到model_input
            speaker_features = self.speaker_cache[speaker_id].copy()
            sentence.model_input = speaker_features

        # 2) UUID处理
        if not reuse_uuid:
            # 不复用 => 重置
            sentence.model_input['uuid'] = ""

        # 3) 更新文本特征
        self._update_text_features_sync(sentence)
        return sentence

    async def _process_sentence_async(self, sentence, reuse_speaker=False, reuse_uuid=False):
        """
        将同步逻辑封装到线程池中执行
        """
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
        """
        统一的对外接口，用于处理句子（首次 or 仅更新文本特征）:
          - reuse_speaker=False, reuse_uuid=False 表示“首次处理”(含 speaker)
          - reuse_speaker=True,  reuse_uuid=True  表示“仅更新文本特征”

        其他组合也可自由使用:
          - reuse_speaker=True,  reuse_uuid=False => 复用 speaker, 但重新分配 UUID
          - reuse_speaker=False, reuse_uuid=True  => 不复用 speaker(重做前端), 但保留 UUID

        常见用法示例:
          1) 首次处理:
             async for batch in self.modelin_maker(sentences, batch_size=3):
                 yield batch
          2) 只更新文本特征(重试场景):
             async for batch in self.modelin_maker(sentences, reuse_speaker=True, reuse_uuid=True, batch_size=3):
                 ...
        """
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
            # 如果不复用 speaker, 说明是“首次完整处理”，用完后清缓存
            if not reuse_speaker:
                self.speaker_cache.clear()
                self.logger.debug("modelin_maker: 已清理 speaker_cache")
