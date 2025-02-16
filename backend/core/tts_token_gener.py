# core/tts_token_gener.py

import logging
import asyncio
import uuid
from utils import concurrency
import torch
import numpy as np

# 假设在 services/cosyvoice 下有 client.py
from services.cosyvoice.client import CosyVoiceClient

class TTSTokenGenerator:
    def __init__(self, cosyvoice_client: CosyVoiceClient):
        """
        Args:
            cosyvoice_client: gRPC 客户端封装
        """
        self.cosyvoice_client = cosyvoice_client
        self.logger = logging.getLogger(__name__)

    async def tts_token_maker(self, sentences, reuse_uuid=False):
        """
        对一批 Sentence 异步生成 TTS token
        """
        try:
            tasks = []
            for s in sentences:
                current_uuid = (
                    s.model_input.get('uuid')
                    if reuse_uuid and s.model_input.get('uuid')
                    else str(uuid.uuid1())
                )
                tasks.append(
                    asyncio.create_task(
                        self._generate_tts_single_async(s, current_uuid)
                    )
                )
            # 并发执行
            processed = await asyncio.gather(*tasks)

            # 检查结果
            for sen in processed:
                # 若没有 segment_speech_tokens，说明生成失败
                if not sen.model_input.get('segment_speech_tokens'):
                    self.logger.error(f"TTS token 生成失败: {sen.trans_text or '(空)'}")

            return processed

        except Exception as e:
            self.logger.error(f"TTS token 生成失败: {e}")
            raise

    async def _generate_tts_single_async(self, sentence, main_uuid):
        """
        在 asyncio 环境下包装实际的 _generate_tts_single
        """
        return await concurrency.run_sync(
            self._generate_tts_single, sentence, main_uuid
        )

    def _generate_tts_single(self, sentence, main_uuid):
        """
        生成单个句子的TTS tokens
        """
        model_input = sentence.model_input

        # 1) 检查文本分段
        token_tensors = model_input.get('text', [])
        token_lengths = model_input.get('text_len', [])

        if not token_tensors or not token_lengths:
            self.logger.warning(f"[TTS Token] 未检测到文本分段 => 生成空 token (UUID={main_uuid})")
            model_input['segment_speech_tokens'] = []
            model_input['segment_uuids'] = []
            model_input['uuid'] = main_uuid
            sentence.duration = sentence.target_duration
            return sentence

        # 2) 准备数据 - 确保所有数据都是Python原生类型
        # 处理text_segments
        text_segments = []
        for tokens in token_tensors:
            if isinstance(tokens, (torch.Tensor, np.ndarray)):
                tokens = tokens.ravel().tolist()
            elif not isinstance(tokens, list):
                tokens = list(tokens)
            text_segments.append(tokens)

        # 打印数据类型信息
        self.logger.info("========== TTS Token生成数据类型 ==========")
        self.logger.info(f"text_segments type: {type(text_segments)}")
        for i, seg in enumerate(text_segments):
            self.logger.info(f"segment {i} type: {type(seg)}, content: {seg[:10]}...")
        self.logger.info(f"features type: {type(model_input.get('features'))}")
        self.logger.info("==========================================")

        # 准备TTS Token生成上下文
        tts_token_context = {
            'prompt_text': model_input.get('prompt_text', []),
            'prompt_text_len': model_input.get('prompt_text_len', 0),
            'features': model_input.get('features')  # 直接传递完整的features对象
        }

        # 3) 调用gRPC服务
        resp = self.cosyvoice_client.generate_tts_tokens(
            uuid=main_uuid,
            text_segments=[list(map(int, seg)) for seg in text_segments],  # 确保所有元素都是int类型
            tts_token_context=tts_token_context
        )

        # 4) 处理响应
        model_input['segment_speech_tokens'] = [seg['tokens'] for seg in resp['segments']]
        model_input['segment_uuids'] = [seg['uuid'] for seg in resp['segments']]
        model_input['uuid'] = main_uuid
        
        # 5) 设置时长
        sentence.duration = resp['duration_ms']

        self.logger.debug(
            f"[TTS Token] (UUID={main_uuid}) => 共 {len(model_input['segment_uuids'])} 段, "
            f"时长估计={sentence.duration/1000:.2f}s"
        )
        return sentence
