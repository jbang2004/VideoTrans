# services/cosyvoice/service.py

import logging
import numpy as np
import torch
import grpc
from concurrent import futures
import uuid
import os
import soundfile as sf

from .proto import cosyvoice_pb2
from .proto import cosyvoice_pb2_grpc
from .cache import FeaturesCache, TextFeatureData, SpeakerFeatureData

# 假设我们使用 CosyVoice2 作为模型
from models.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2

logger = logging.getLogger(__name__)

class CosyVoiceServiceServicer(cosyvoice_pb2_grpc.CosyVoiceServiceServicer):
    def __init__(self, model_path="models/CosyVoice/pretrained_models/CosyVoice2-0.5B"):
        try:
            self.cosyvoice = CosyVoice2(model_path)
            self.frontend = self.cosyvoice.frontend
            self.model = self.cosyvoice.model
            self.sample_rate = self.cosyvoice.sample_rate
            self.cache = FeaturesCache()
            
            logger.info('CosyVoice服务初始化成功')
        except Exception as e:
            logger.error(f'CosyVoice服务初始化失败: {e}')
            raise

    def _generate_uuid(self) -> str:
        """生成唯一的UUID"""
        return str(uuid.uuid4())

    def NormalizeText(self, request, context):
        """
        文本标准化，生成文本特征数据并缓存
        """
        try:
            text = request.text or ""
            normalized_texts = self.frontend.text_normalize(text, split=True, text_frontend=False)
            text_tokens = []
            
            for seg in normalized_texts:
                tokens, _ = self.frontend._extract_text_token(seg)
                text_tokens.append(tokens)

            text_uuid = self._generate_uuid()
            self.cache.update_text_features(text_uuid, normalized_texts, text_tokens)

            return cosyvoice_pb2.NormalizeTextResponse(text_uuid=text_uuid)
        except Exception as e:
            logger.error(f"NormalizeText失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.NormalizeTextResponse()

    def ExtractSpeakerFeatures(self, request, context):
        """
        提取说话人特征并缓存
        """
        try:
            speaker_uuid = request.speaker_uuid
            # 转换为tensor，确保数组可写
            audio_np = np.frombuffer(request.audio, dtype=np.float32).copy()
            audio = torch.from_numpy(audio_np).unsqueeze(0)  # [1, T]

            # 使用tensor调用frontend，这里的sample_rate是目标采样率
            features = self.frontend.frontend_cross_lingual("", audio, request.sample_rate)
            
            # 直接存储完整的特征字典
            success = self.cache.update_speaker_features(speaker_uuid, features)

            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("更新说话人特征缓存失败")
                return cosyvoice_pb2.ExtractSpeakerFeaturesResponse()

            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse(success=True)
        except Exception as e:
            logger.error(f"ExtractSpeakerFeatures失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse()

    def GenerateTTSTokens(self, request, context):
        """
        使用文本UUID和说话人UUID生成TTS tokens并更新缓存
        """
        try:
            text_uuid = request.text_uuid
            speaker_uuid = request.speaker_uuid
            text_features = self.cache.get_text(text_uuid)
            speaker_features = self.cache.get_speaker(speaker_uuid)

            if not text_features:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Text UUID {text_uuid} 不存在")
                return cosyvoice_pb2.GenerateTTSTokensResponse()
            if not speaker_features:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Speaker UUID {speaker_uuid} 不存在")
                return cosyvoice_pb2.GenerateTTSTokensResponse()

            total_duration_ms = 0
            tts_tokens = []
            segment_uuids = []

            for i, text_tokens in enumerate(text_features.text_tokens):
                seg_uuid = f"{text_uuid}_seg_{i}"
                segment_uuids.append(seg_uuid)

                with self.model.lock:
                    self.model.tts_speech_token_dict[seg_uuid] = []
                    self.model.llm_end_dict[seg_uuid] = False

                try:
                    self.model.llm_job(
                        text=text_tokens,
                        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                        llm_prompt_speech_token=speaker_features.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                        llm_embedding=speaker_features.get('llm_embedding', torch.zeros(0, 192)),
                        uuid=seg_uuid
                    )

                    seg_tokens = self.model.tts_speech_token_dict[seg_uuid]
                    tts_tokens.append(seg_tokens)
                    total_duration_ms += len(seg_tokens) / 25.0 * 1000

                finally:
                    self.model.tts_speech_token_dict.pop(seg_uuid, None)
                    self.model.llm_end_dict.pop(seg_uuid, None)

            success = self.cache.update_tts_tokens(text_uuid, tts_tokens, segment_uuids)
            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("更新TTS tokens缓存失败")
                return cosyvoice_pb2.GenerateTTSTokensResponse()

            return cosyvoice_pb2.GenerateTTSTokensResponse(
                duration_ms=int(total_duration_ms),
                success=True
            )

        except Exception as e:
            logger.error(f"GenerateTTSTokens失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.GenerateTTSTokensResponse()

    def Token2Wav(self, request, context):
        """
        使用文本UUID和说话人UUID将TTS tokens转换为音频
        """
        try:
            text_uuid = request.text_uuid
            speaker_uuid = request.speaker_uuid
            text_features = self.cache.get_text(text_uuid)
            speaker_features = self.cache.get_speaker(speaker_uuid)

            if not text_features:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Text UUID {text_uuid} 不存在")
                return cosyvoice_pb2.Token2WavResponse()
            if not speaker_features:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Speaker UUID {speaker_uuid} 不存在")
                return cosyvoice_pb2.Token2WavResponse()

            speed = request.speed
            audio_pieces = []
            total_duration_sec = 0.0

            for seg_tokens, seg_uuid in zip(text_features.tts_tokens, text_features.tts_segment_uuids):
                if not seg_tokens:
                    continue

                self.model.hift_cache_dict[seg_uuid] = None
                try:
                    seg_audio_out = self.model.token2wav(
                        token=torch.tensor(seg_tokens).unsqueeze(0),
                        prompt_token=speaker_features.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32)),
                        prompt_feat=speaker_features.get('prompt_speech_feat', torch.zeros(1, 0, 80)),
                        embedding=speaker_features.get('llm_embedding', torch.zeros(0)),
                        uuid=seg_uuid,
                        token_offset=0,
                        finalize=True,
                        speed=speed
                    )
                finally:
                    self.model.hift_cache_dict.pop(seg_uuid, None)

                # 处理音频：移除batch维度，如果是多通道则取平均
                seg_audio = seg_audio_out.cpu().numpy()
                if seg_audio.ndim > 1:
                    seg_audio = seg_audio.mean(axis=0)  # 如果还有多个通道，取平均得到单通道
                audio_pieces.append(seg_audio)
                total_duration_sec += len(seg_audio) / self.sample_rate

            if not audio_pieces:
                logger.warning(f"Token2Wav: Text UUID {text_uuid} 未生成任何音频")
                return cosyvoice_pb2.Token2WavResponse()

            final_audio = np.concatenate(audio_pieces)
            audio_int16 = (final_audio * (2**15)).astype(np.int16).tobytes()

            return cosyvoice_pb2.Token2WavResponse(
                audio=audio_int16,
                duration_sec=total_duration_sec
            )

        except Exception as e:
            logger.error(f"Token2Wav失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.Token2WavResponse()

    def Cleanup(self, request, context):
        """
        清理指定UUID的文本或说话人特征
        """
        try:
            uuid = request.uuid
            is_speaker = request.is_speaker
            self.cache.delete(uuid, is_speaker)
            return cosyvoice_pb2.CleanupResponse(success=True)
        except Exception as e:
            logger.error(f"Cleanup失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.CleanupResponse(success=False)

def serve(args):
    host = getattr(args, 'host', '0.0.0.0')
    port = getattr(args, 'port', 50052)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cosyvoice_pb2_grpc.add_CosyVoiceServiceServicer_to_server(
        CosyVoiceServiceServicer(args.model_dir), server
    )
    address = f'{host}:{port}'
    server.add_insecure_port(address)
    server.start()
    logger.info(f'CosyVoice服务已启动: {address}')
    server.wait_for_termination()