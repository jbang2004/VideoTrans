# services/cosyvoice/service.py

import logging
import numpy as np
import torch
import grpc
from concurrent import futures
import uuid

from .proto import cosyvoice_pb2
from .proto import cosyvoice_pb2_grpc
from .cache import FeaturesCache, FeatureData

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
        文本标准化，生成特征数据并缓存
        """
        try:
            text = request.text or ""
            normalized_texts = self.frontend.text_normalize(text, split=True, text_frontend=True)
            text_tokens = []
            
            # 提取文本tokens，保持原有维度
            for seg in normalized_texts:
                tokens, _ = self.frontend._extract_text_token(seg)  # 已经是 [batch_size, seq_len]
                text_tokens.append(tokens)

            # 生成UUID并缓存特征
            new_uuid = self._generate_uuid()
            self.cache.update_text_features(new_uuid, normalized_texts, text_tokens)

            return cosyvoice_pb2.NormalizeTextResponse(uuid=new_uuid)
        except Exception as e:
            logger.error(f"NormalizeText失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.NormalizeTextResponse()

    def ExtractSpeakerFeatures(self, request, context):
        """
        提取说话人特征并更新缓存
        """
        try:
            uuid = request.uuid
            if not self.cache.exists(uuid):
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"UUID {uuid} 不存在")
                return cosyvoice_pb2.ExtractSpeakerFeaturesResponse()

            # 处理音频数据
            audio_np = np.frombuffer(request.audio, dtype=np.float32).copy()
            audio = torch.from_numpy(audio_np).unsqueeze(0)  # [1, T]

            # 提取特征 - 直接使用模型输出的tensor，维度已经正确
            result = self.frontend.frontend_cross_lingual("", audio, request.sample_rate)
            
            # 更新缓存 - 直接存储tensor，维度已经正确
            success = self.cache.update_speaker_features(
                uuid,
                embedding=result['llm_embedding'],  # [1, embedding_dim]
                prompt_feat=result['prompt_speech_feat'],  # [1, time, 80]
                prompt_feat_len=int(result['prompt_speech_feat_len'].item()),
                prompt_token=result['flow_prompt_speech_token'],  # [1, seq_len]
                prompt_token_len=int(result['flow_prompt_speech_token_len'].item())
            )

            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("更新缓存失败")
                return cosyvoice_pb2.ExtractSpeakerFeaturesResponse()

            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse(success=True)
        except Exception as e:
            logger.error(f"ExtractSpeakerFeatures失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse()

    def GenerateTTSTokens(self, request, context):
        """
        生成TTS tokens并更新缓存
        """
        try:
            uuid = request.uuid
            features = self.cache.get(uuid)
            if not features:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"UUID {uuid} 不存在")
                return cosyvoice_pb2.GenerateTTSTokensResponse()

            total_duration_ms = 0
            tts_tokens = []
            segment_uuids = []

            # 处理每段文本
            for i, text_tokens in enumerate(features.text_tokens):
                seg_uuid = f"{uuid}_seg_{i}"
                segment_uuids.append(seg_uuid)

                # 初始化LLM状态
                self.model.tts_speech_token_dict[seg_uuid] = []
                self.model.llm_end_dict[seg_uuid] = False

                try:
                    # 生成TTS tokens，使用默认空特征值替换不存在的特征
                    self.model.llm_job(
                        text=text_tokens,
                        prompt_text=torch.zeros(1, 0, dtype=torch.int32),
                        llm_prompt_speech_token=features.prompt_speech_token if features.prompt_speech_token is not None 
                            else torch.zeros(1, 0, dtype=torch.int32),
                        llm_embedding=features.embedding if features.embedding is not None 
                            else torch.zeros(0, 192),
                        uuid=seg_uuid
                    )

                    # 获取生成的tokens
                    seg_tokens = self.model.tts_speech_token_dict[seg_uuid]
                    tts_tokens.append(seg_tokens)

                    # 估算时长
                    total_duration_ms += len(seg_tokens) / 25.0 * 1000

                finally:
                    # 清理LLM状态
                    self.model.tts_speech_token_dict.pop(seg_uuid, None)
                    self.model.llm_end_dict.pop(seg_uuid, None)

            # 更新缓存
            success = self.cache.update_tts_tokens(uuid, tts_tokens, segment_uuids)
            if not success:
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("更新缓存失败")
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
        将缓存中的TTS tokens转换为音频
        """
        try:
            uuid = request.uuid
            features = self.cache.get(uuid)
            if not features:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"UUID {uuid} 不存在")
                return cosyvoice_pb2.Token2WavResponse()

            speed = request.speed
            audio_pieces = []
            total_duration_sec = 0.0

            # 处理每个TTS segment
            for seg_tokens, seg_uuid in zip(features.tts_tokens, features.tts_segment_uuids):
                if not seg_tokens:
                    continue

                # 设置缓存
                self.model.hift_cache_dict[seg_uuid] = None
                try:
                    # 使用默认空特征值替换不存在的特征
                    seg_audio_out = self.model.token2wav(
                        token=torch.tensor(seg_tokens).unsqueeze(0),
                        prompt_token=features.prompt_speech_token if features.prompt_speech_token is not None 
                            else torch.zeros(1, 0, dtype=torch.int32),
                        prompt_feat=features.prompt_speech_feat if features.prompt_speech_feat is not None 
                            else torch.zeros(1, 0, 80),
                        embedding=features.embedding if features.embedding is not None 
                            else torch.zeros(0),
                        uuid=seg_uuid,
                        token_offset=0,
                        finalize=True,
                        speed=speed
                    )
                finally:
                    self.model.hift_cache_dict.pop(seg_uuid, None)

                # 处理音频
                seg_audio = seg_audio_out.cpu().numpy().squeeze()
                if seg_audio.ndim > 1:
                    seg_audio = seg_audio.mean(axis=0)
                audio_pieces.append(seg_audio)
                total_duration_sec += len(seg_audio) / self.sample_rate

            if not audio_pieces:
                logger.warning(f"Token2Wav: UUID {uuid} 未生成任何音频")
                return cosyvoice_pb2.Token2WavResponse()

            # 合并音频
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
        清理缓存中的特征数据
        """
        try:
            uuid = request.uuid
            self.cache.delete(uuid)
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
