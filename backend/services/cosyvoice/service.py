# services/cosyvoice/service.py

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
        """
        执行文本标准化
        """
        try:
            text = request.text or ""
            normalized_texts = self.frontend.text_normalize(text, split=True, text_frontend=True)

            text_features_msg = cosyvoice_pb2.Features()
            for seg in normalized_texts:
                text_features_msg.normalized_text_segments.append(seg)
                # 用模型的分词器抽取 tokens
                tokens, token_len = self.frontend._extract_text_token(seg)
                # 这里每段 tokens 存到一个 TextSegment
                text_seg_msg = text_features_msg.text_segments.add()
                text_seg_msg.tokens.extend(tokens.reshape(-1).tolist())

            return cosyvoice_pb2.NormalizeTextResponse(features=text_features_msg)
        except Exception as e:
            logger.error(f"NormalizeText失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.NormalizeTextResponse()

    # ========== 2) ExtractSpeakerFeatures ==========
    def ExtractSpeakerFeatures(self, request, context):
        """
        接收音频并提取说话人相关的特征 (embedding、prompt_speech_feat、prompt_speech_token 等)
        """
        try:
            # 创建一个可写的数组副本
            audio_np = np.frombuffer(request.audio, dtype=np.float32).copy()
            audio = torch.from_numpy(audio_np).unsqueeze(0)

            # 示例: zero-shot / cross-lingual 提取
            result = self.frontend.frontend_cross_lingual("", audio, self.sample_rate)

            speaker_features_msg = cosyvoice_pb2.Features(
                embedding=result['llm_embedding'].reshape(-1).tolist(),
                prompt_speech_feat=result['prompt_speech_feat'].reshape(-1).tolist(),
                prompt_speech_feat_len=int(result['prompt_speech_feat_len'].item()),
                prompt_speech_token=result['flow_prompt_speech_token'].reshape(-1).tolist(),
                prompt_speech_token_len=int(result['flow_prompt_speech_token_len'].item())
            )
            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse(features=speaker_features_msg)
        except Exception as e:
            logger.error(f"ExtractSpeakerFeatures失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse()

    # ========== 3) GenerateTTSTokens (分段) ==========
    def GenerateTTSTokens(self, request, context):
        """
        将输入文本特征转换为TTS tokens，并估算时长
        """
        try:
            features_in = request.features
            base_uuid = request.uuid or "no_uuid"
            
            # 1. 准备输出特征，复制输入的speaker相关特征
            tts_out_features = cosyvoice_pb2.Features()
            tts_out_features.embedding.extend(features_in.embedding)
            tts_out_features.prompt_speech_feat.extend(features_in.prompt_speech_feat)
            tts_out_features.prompt_speech_feat_len = features_in.prompt_speech_feat_len
            tts_out_features.prompt_speech_token.extend(features_in.prompt_speech_token)
            tts_out_features.prompt_speech_token_len = features_in.prompt_speech_token_len

            # 2. 准备模型输入
            embedding_tensor = torch.tensor(features_in.embedding, dtype=torch.float32).unsqueeze(0)
            prompt_token_tensor = torch.tensor(features_in.prompt_speech_token, dtype=torch.int32).unsqueeze(0)

            total_duration_ms = 0
            # 3. 处理每段文本
            for i, text_seg in enumerate(features_in.text_segments):
                if not text_seg.tokens:
                    continue

                seg_uuid = f"{base_uuid}_seg_{i}"
                text_tensor = torch.tensor(text_seg.tokens, dtype=torch.int32).unsqueeze(0)

                # 初始化LLM状态
                with self.lock:
                    self.model.tts_speech_token_dict[seg_uuid] = []
                    self.model.llm_end_dict[seg_uuid] = False

                try:
                    # 生成TTS tokens
                    self.model.llm_job(
                        text=text_tensor,
                        prompt_text=torch.zeros((1, 0), dtype=torch.int32),
                        llm_prompt_speech_token=prompt_token_tensor,
                        llm_embedding=embedding_tensor,
                        uuid=seg_uuid
                    )

                    # 获取生成的tokens并添加到输出
                    tts_tokens = self.model.tts_speech_token_dict[seg_uuid]
                    
                    tts_seg_msg = tts_out_features.tts_segments.add()
                    tts_seg_msg.uuid = seg_uuid
                    tts_seg_msg.tokens.extend(tts_tokens)  # 直接使用list

                    # 估算时长（每25个token约1秒）
                    total_duration_ms += len(tts_tokens) / 25.0 * 1000

                finally:
                    # 清理LLM状态
                    with self.lock:
                        self.model.tts_speech_token_dict.pop(seg_uuid, None)
                        self.model.llm_end_dict.pop(seg_uuid, None)

            return cosyvoice_pb2.GenerateTTSTokensResponse(
                features=tts_out_features,
                duration_ms=int(total_duration_ms)
            )

        except Exception as e:
            logger.error(f"GenerateTTSTokens失败: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.GenerateTTSTokensResponse()

    # ========== 4) Token2Wav (把分段 tokens 拼起来一次合成) ==========
    def Token2Wav(self, request, context):
        """
        将 GenerateTTSTokens 里返回的多段 tokens (features.tts_segments) 分段调用 token2wav，
        使用每个seg中的uuid（如果存在）作为token2wav的uuid，然后将各段生成的音频合并。
        """
        try:
            features = request.features
            speed = request.speed

            # 转换 embedding
            if len(features.embedding) > 0:
                embedding_tensor = torch.tensor(features.embedding, dtype=torch.float32).unsqueeze(0)
            else:
                embedding_tensor = torch.zeros((1, 0), dtype=torch.float32)
                logger.warning("Token2Wav: embedding 为空, 使用零向量代替.")

            # 转换 prompt_speech_feat
            if len(features.prompt_speech_feat) > 0:
                feat_tensor = torch.tensor(features.prompt_speech_feat, dtype=torch.float32).reshape(1, -1, 80)
            else:
                feat_tensor = torch.zeros((1, 0, 80), dtype=torch.float32)

            prompt_token_tensor = torch.tensor(features.prompt_speech_token, dtype=torch.int32).unsqueeze(0)

            audio_pieces = []
            total_duration_sec = 0.0

            # 针对每个tts segment单独生成音频
            for seg in features.tts_segments:
                tokens = seg.tokens
                if not tokens:
                    continue

                # 使用 seg 中的 uuid，如果为空则采用默认值
                seg_uuid = seg.uuid if seg.uuid else "token2wav_uuid"

                # 直接将tokens转为tensor
                token_tensor = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0)

                # 设置缓存，保证线程安全
                with self.model.lock:
                    self.model.hift_cache_dict[seg_uuid] = None
                try:
                    seg_audio_out = self.model.token2wav(
                        token=token_tensor,
                        prompt_token=prompt_token_tensor,
                        prompt_feat=feat_tensor,
                        embedding=embedding_tensor,
                        uuid=seg_uuid,
                        token_offset=0,
                        finalize=True,
                        speed=speed
                    )
                except Exception as e:
                    error_msg = f"Token2Wav合成失败 (UUID: {seg_uuid}): {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(error_msg)
                    return cosyvoice_pb2.Token2WavResponse()
                finally:
                    with self.model.lock:
                        self.model.hift_cache_dict.pop(seg_uuid, None)

                # 处理生成的音频：转换为 numpy 数组、去除无用维度，如果多通道取均值
                seg_audio = seg_audio_out.cpu().numpy().squeeze()
                if seg_audio.ndim > 1:
                    seg_audio = seg_audio.mean(axis=0)
                audio_pieces.append(seg_audio)
                total_duration_sec += len(seg_audio) / self.sample_rate

            if not audio_pieces:
                logger.warning("Token2Wav: 未生成任何音频，可能缺少有效 tokens")
                return cosyvoice_pb2.Token2WavResponse()

            # 合并所有段音频
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
