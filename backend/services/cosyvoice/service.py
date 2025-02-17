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
        """
        接收音频并提取说话人相关的特征 (embedding、prompt_speech_feat、prompt_speech_token 等)
        """
        try:
            audio_int16 = request.audio
            sr = request.sample_rate

            # === 打印输入音频信息 ===
            logger.info("=== ExtractSpeakerFeatures 输入音频信息 ===")
            audio_np = np.frombuffer(audio_int16, dtype=np.int16).astype(np.float32) / (2**15)
            logger.info(f"输入音频: shape={audio_np.shape}, sr={sr}Hz")
            logger.info(f"音频统计: min={audio_np.min():.4f}, max={audio_np.max():.4f}, mean={audio_np.mean():.4f}, std={audio_np.std():.4f}")
            
            audio = torch.from_numpy(audio_np).unsqueeze(0)
            logger.info(f"处理后音频tensor shape: {audio.shape}")
            logger.info("================================")

            # 示例: zero-shot / cross-lingual 提取
            result = self.frontend.frontend_cross_lingual("", audio, sr)

            # === 打印提取的特征信息 ===
            logger.info("=== ExtractSpeakerFeatures 输出特征信息 ===")
            
            # 1. embedding
            emb = result['llm_embedding'].reshape(-1).tolist()
            logger.info(f"llm_embedding: shape={result['llm_embedding'].shape}, len={len(emb)}")
            logger.info(f"embedding前5个值: {emb[:5]}")
            
            # 2. prompt_speech_feat
            ps_feat = result['prompt_speech_feat'].reshape(-1).tolist()
            logger.info(f"prompt_speech_feat: shape={result['prompt_speech_feat'].shape}, len={len(ps_feat)}")
            logger.info(f"prompt_speech_feat_len: {result['prompt_speech_feat_len'].item()}")
            
            # 3. prompt_speech_token
            ps_token = result['flow_prompt_speech_token'].reshape(-1).tolist()
            logger.info(f"flow_prompt_speech_token: shape={result['flow_prompt_speech_token'].shape}, len={len(ps_token)}")
            logger.info(f"flow_prompt_speech_token_len: {result['flow_prompt_speech_token_len'].item()}")
            logger.info("================================")

            features_msg = cosyvoice_pb2.Features(
                embedding=emb,
                prompt_speech_feat=ps_feat,
                prompt_speech_feat_len=int(result['prompt_speech_feat_len'].item()),
                prompt_speech_token=ps_token,
                prompt_speech_token_len=int(result['flow_prompt_speech_token_len'].item())
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
        并存储到 out_features.tts_segments；也把原先的embedding等字段复制到 out_features 中。
        """
        try:
            features_in = request.features
            base_uuid = request.uuid or "no_uuid"

            # # === 打印输入features的详细信息 ===
            # logger.info("=== GenerateTTSTokens features_in详细信息 ===")
            
            # # 1. 文本相关信息
            # logger.info(f"normalized_text_segments数量: {len(features_in.normalized_text_segments)}")
            # for i, text in enumerate(features_in.normalized_text_segments):
            #     logger.info(f"  第{i+1}段标准化文本: {text}")
            
            # logger.info(f"text_segments数量: {len(features_in.text_segments)}")
            # for i, seg in enumerate(features_in.text_segments):
            #     logger.info(f"  第{i+1}段:")
            #     logger.info(f"    tokens长度: {len(seg.tokens)}")
            #     if len(seg.tokens) > 0:
            #         logger.info(f"    tokens前5个值: {seg.tokens[:5]}")
            
            # # 2. embedding信息
            # logger.info(f"embedding长度: {len(features_in.embedding)}")
            # if len(features_in.embedding) > 0:
            #     logger.info(f"embedding前5个值: {features_in.embedding[:5]}")
            
            # # 3. prompt speech相关信息
            # logger.info(f"prompt_speech_feat长度: {len(features_in.prompt_speech_feat)}")
            # logger.info(f"prompt_speech_feat_len: {features_in.prompt_speech_feat_len}")
            # logger.info(f"prompt_speech_token长度: {len(features_in.prompt_speech_token)}")
            # logger.info(f"prompt_speech_token_len: {features_in.prompt_speech_token_len}")
            
            # logger.info(f"base_uuid: {base_uuid}")
            # logger.info("================================")

            # 这里原先只取了 text_segments 的 tokens, 并未处理embedding
            # 如果你要后面 Token2Wav 用到embedding, 就在 out_features 里保留:
            out_features = cosyvoice_pb2.Features()

            # === 把输入的 embedding & prompt_speech_* 也写到 out_features ===
            out_features.embedding.extend(features_in.embedding)
            out_features.prompt_speech_feat.extend(features_in.prompt_speech_feat)
            out_features.prompt_speech_feat_len = features_in.prompt_speech_feat_len
            out_features.prompt_speech_token.extend(features_in.prompt_speech_token)
            out_features.prompt_speech_token_len = features_in.prompt_speech_token_len

            total_duration_ms = 0

            # 遍历每段文本
            for i, text_seg in enumerate(features_in.text_segments):
                seg_tokens = text_seg.tokens
                if not seg_tokens:
                    continue

                seg_uuid = f"{base_uuid}_seg_{i}"
                text_tensor = torch.tensor(seg_tokens, dtype=torch.int32).unsqueeze(0)

                # 由于 CosyVoice2Model 的 llm_job 会把 tokens 存进 self.model.tts_speech_token_dict[seg_uuid],
                # 我们要先初始化一下
                with self.lock:
                    if seg_uuid not in self.model.tts_speech_token_dict:
                        self.model.tts_speech_token_dict[seg_uuid] = []
                    if seg_uuid not in self.model.llm_end_dict:
                        self.model.llm_end_dict[seg_uuid] = False

                # 调用 LLM 生成 tokens (同步方式)
                try:
                    # 这里可能需要embedding, prompt等：你原先写死了 embedding_tensor 为空
                    # 不过如果只是演示分段tokens, 也可以不传embedding. 视具体业务需要
                    # 例：embedding等都用上:
                    embedding_tensor = torch.tensor(features_in.embedding, dtype=torch.float32)
                    prompt_token_tensor = torch.tensor(features_in.prompt_speech_token, dtype=torch.int32).unsqueeze(0)

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

                # 粗略估计时长
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
        """
        将 GenerateTTSTokens 里返回的多段 tokens (features.tts_segments) 合并，
        结合embedding等信息，调用 model.token2wav 完成一次音频合成。
        """
        try:
            features = request.features
            speed = request.speed

            # # === 打印features的详细信息 ===
            # logger.info("=== Token2Wav features详细信息 ===")
            
            # # 1. embedding信息
            # logger.info(f"embedding长度: {len(features.embedding)}")
            # if len(features.embedding) > 0:
            #     logger.info(f"embedding前5个值: {features.embedding[:5]}")
            
            # # 2. prompt speech相关信息
            # logger.info(f"prompt_speech_feat长度: {len(features.prompt_speech_feat)}")
            # logger.info(f"prompt_speech_feat_len: {features.prompt_speech_feat_len}")
            # logger.info(f"prompt_speech_token长度: {len(features.prompt_speech_token)}")
            # logger.info(f"prompt_speech_token_len: {features.prompt_speech_token_len}")
            
            # # 3. tts segments信息
            # logger.info(f"tts_segments数量: {len(features.tts_segments)}")
            # for i, seg in enumerate(features.tts_segments):
            #     logger.info(f"  第{i+1}段:")
            #     logger.info(f"    UUID: {seg.uuid}")
            #     logger.info(f"    tokens长度: {len(seg.tokens)}")
            #     if len(seg.tokens) > 0:
            #         logger.info(f"    tokens前5个值: {seg.tokens[:5]}")
            
            # logger.info(f"speed: {speed}")
            # logger.info("================================")

            # 收集所有段 tokens
            all_tokens = []
            for seg in features.tts_segments:
                all_tokens.extend(seg.tokens)

            # 先把 embedding 转成 Torch 张量
            embedding_tensor = torch.tensor(features.embedding, dtype=torch.float32)
            if embedding_tensor.numel() > 0:
                # 如果非空，补一个 batch 维度 => [1, N]
                embedding_tensor = embedding_tensor.unsqueeze(0)
            else:
                # 如果真的没有embedding, 你可以选择报错，或用一个零向量
                # 这里随便给了一个 [1, 192]，具体大小看你的模型需要
                embedding_tensor = torch.zeros((1, 192), dtype=torch.float32)
                logger.warning("Token2Wav: embedding 为空, 使用零向量代替.")

            # prompt_speech_feat 如果有，则 reshape(1, -1, 80); 否则 (1,0,80)
            if len(features.prompt_speech_feat) > 0:
                feat_tensor = torch.tensor(features.prompt_speech_feat, dtype=torch.float32).reshape(1, -1, 80)
            else:
                feat_tensor = torch.zeros((1, 0, 80), dtype=torch.float32)

            if not all_tokens:
                logger.warning("Token2Wav: 未发现 tokens")
                return cosyvoice_pb2.Token2WavResponse()

            token_tensor = torch.tensor(all_tokens, dtype=torch.int32).unsqueeze(0)
            prompt_token_tensor = torch.tensor(features.prompt_speech_token, dtype=torch.int32).unsqueeze(0)

            # 调用 model 进行一次性合成
            this_uuid = "token2wav_uuid"
            audio_out = self.model.token2wav(
                token=token_tensor,
                prompt_token=prompt_token_tensor,
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
