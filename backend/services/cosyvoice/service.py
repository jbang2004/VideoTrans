import logging
import numpy as np
import torch
import grpc
from concurrent import futures
import threading  # 导入 threading 模块

from .proto import cosyvoice_pb2
from .proto import cosyvoice_pb2_grpc
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
            self.lock = threading.Lock()  # 创建锁
            logger.info('CosyVoice服务初始化成功')
        except Exception as e:
            logger.error(f'CosyVoice服务初始化失败: {e}')
            raise

    # ========== 1) NormalizeText ==========
    def NormalizeText(self, request, context):
        try:
            text = request.text or ""
            normalized_texts = self.frontend.text_normalize(text, split=True, text_frontend=True)

            features_msg = cosyvoice_pb2.Features()
            for seg in normalized_texts:
                features_msg.normalized_text_segments.append(seg)
                # 用模型的分词器抽取 tokens
                tokens, token_len = self.frontend._extract_text_token(seg)
                # 这里每段 tokens 存到一个 TextSegment
                seg_msg = features_msg.text_segments.add()
                seg_msg.tokens.extend(tokens.reshape(-1).tolist())

            return cosyvoice_pb2.NormalizeTextResponse(features=features_msg)
        except Exception as e:
            logger.error(f"NormalizeText失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.NormalizeTextResponse()

    # ========== 2) ExtractSpeakerFeatures ==========
    def ExtractSpeakerFeatures(self, request, context):
        try:
            audio_int16 = request.audio
            sr = self.sample_rate if self.sample_rate is not None else request.sample_rate

            audio_np = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / (2**15)
            audio = torch.from_numpy(audio_np).unsqueeze(0)

            # 示例: zero-shot / cross-lingual 提取
            result = self.frontend.frontend_cross_lingual("", audio, sr)

            emb = result['llm_embedding'].reshape(-1).tolist()
            ps_feat = result['prompt_speech_feat'].reshape(-1).tolist()
            ps_feat_len = int(result['prompt_speech_feat_len'].item())
            ps_token = result['flow_prompt_speech_token'].reshape(-1).tolist()
            ps_token_len = int(result['flow_prompt_speech_token_len'].item())

            features_msg = cosyvoice_pb2.Features(
                embedding=emb,
                prompt_speech_feat=ps_feat,
                prompt_speech_feat_len=ps_feat_len,
                prompt_speech_token=ps_token,
                prompt_speech_token_len=ps_token_len
            )
            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse(features=features_msg)
        except Exception as e:
            logger.error(f"ExtractSpeakerFeatures失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse()

    # ========== 3) GenerateTTSTokens (分段) ==========
    def GenerateTTSTokens(self, request, context):
        """
        遍历 features_in.text_segments (二维), 对每段生成 TTS tokens
        并存储到 out_features.tts_segments
        """
        try:
            features_in = request.features
            base_uuid = request.uuid or "no_uuid"

            embedding_tensor = torch.tensor(features_in.embedding, dtype=torch.float32)
            prompt_token_tensor = torch.tensor(features_in.prompt_speech_token, dtype=torch.int32).unsqueeze(0)

            total_duration_ms = 0
            out_features = cosyvoice_pb2.Features()

            # 遍历每段文本
            for i, text_seg in enumerate(features_in.text_segments):
                seg_tokens = text_seg.tokens
                if not seg_tokens:
                    continue
                seg_uuid = f"{base_uuid}_seg_{i}"

                text_tensor = torch.tensor(seg_tokens, dtype=torch.int32).unsqueeze(0)

                # 加锁
                with self.lock:
                    # 创建字典 (如果不存在)
                    if seg_uuid not in self.model.tts_speech_token_dict:
                        self.model.tts_speech_token_dict[seg_uuid] = []
                    if seg_uuid not in self.model.llm_end_dict:
                        self.model.llm_end_dict[seg_uuid] = False

                # 调用 LLM 生成 tokens
                try:
                    self.model.llm_job(
                        text=text_tensor,
                        prompt_text=torch.zeros((1, 0), dtype=torch.int32),
                        llm_prompt_speech_token=prompt_token_tensor,
                        llm_embedding=embedding_tensor,
                        uuid=seg_uuid
                    )
                except Exception as e:
                    logger.error(f"LLM 生成 tokens 失败 (UUID: {seg_uuid}): {e}", exc_info=True)
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(f"LLM 生成 tokens 失败 (UUID: {seg_uuid}): {e}")
                    return cosyvoice_pb2.GenerateTTSTokensResponse()

                # 拿到生成的 tokens
                tts_tokens_segment = self.model.tts_speech_token_dict.get(seg_uuid, [])
                # 清理
                with self.lock:
                    self.model.tts_speech_token_dict.pop(seg_uuid, None)
                    self.model.llm_end_dict[seg_uuid] = True

                seg_duration_ms = len(tts_tokens_segment) / 25.0 * 1000
                total_duration_ms += seg_duration_ms

                # 写回 out_features.tts_segments
                seg_msg = out_features.tts_segments.add()
                seg_msg.uuid = seg_uuid
                seg_msg.tokens.extend(tts_tokens_segment)

            return cosyvoice_pb2.GenerateTTSTokensResponse(
                features=out_features,
                duration_ms=int(total_duration_ms)
            )
        except Exception as e:
            logger.error(f"GenerateTTSTokens失败: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.GenerateTTSTokensResponse()

    # ========== 4) Token2Wav (把分段 tokens 拼起来一次合成) ==========
    def Token2Wav(self, request, context):
        try:
            features = request.features
            speed = request.speed

            # 把所有段 tokens 拼成一个大列表
            all_tokens = []
            for seg in features.tts_segments:
                all_tokens.extend(seg.tokens)

            embedding_tensor = torch.tensor(features.embedding, dtype=torch.float32)
            if len(features.prompt_speech_feat) > 0:
                feat_tensor = torch.tensor(features.prompt_speech_feat, dtype=torch.float32).reshape(1, -1, 80)
            else:
                feat_tensor = torch.zeros((1,0,80), dtype=torch.float32)

            if not all_tokens:
                logger.warning("Token2Wav: 未发现 tokens")
                return cosyvoice_pb2.Token2WavResponse()

            token_tensor = torch.tensor(all_tokens, dtype=torch.int32).unsqueeze(0)

            # 合成
            this_uuid = "token2wav_uuid"
            audio_out = self.model.token2wav(
                token=token_tensor,
                prompt_token=torch.zeros((1,0), dtype=torch.int32),
                prompt_feat=feat_tensor,
                embedding=embedding_tensor,
                uuid=this_uuid,
                token_offset=0,
                finalize=True,
                speed=speed
            )

            audio_out = audio_out.cpu().numpy().squeeze()
            if audio_out.ndim > 1:
                audio_out = audio_out.mean(axis=0)

            audio_int16 = (audio_out * (2**15)).astype(np.int16).tobytes()
            duration_sec = len(audio_out) / self.sample_rate

            return cosyvoice_pb2.Token2WavResponse(
                audio=audio_int16,
                duration_sec=duration_sec
            )
        except Exception as e:
            logger.error(f"Token2Wav失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.Token2WavResponse()

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
