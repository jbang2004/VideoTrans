import logging
import numpy as np
import torch
import grpc
from concurrent import futures

from .proto import cosyvoice_pb2
from .proto import cosyvoice_pb2_grpc
from models.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2

logger = logging.getLogger(__name__)

class CosyVoiceServiceServicer(cosyvoice_pb2_grpc.CosyVoiceServiceServicer):
    def __init__(self, model_path="models/CosyVoice/pretrained_models/CosyVoice2-0.5B"):
        try:
            self.cosyvoice = CosyVoice2(model_path)
            self.frontend = self.cosyvoice.frontend
            self.model = self.cosyvoice.model
            self.sample_rate = self.cosyvoice.sample_rate
            self.max_val = 0.8
            logger.info('CosyVoice服务初始化成功')
        except Exception as e:
            logger.error(f'CosyVoice服务初始化失败: {e}')
            raise

    def NormalizeText(self, request, context):
        try:
            # 文本标准化
            normalized_segments = self.frontend.text_normalize(request.text, split=True)
            
            # 构建响应
            segments = []
            for text in normalized_segments:
                tokens, token_len = self.frontend._extract_text_token(text)
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.cpu().numpy()
                if isinstance(tokens, np.ndarray):
                    tokens = tokens.ravel()
                segments.append(
                    cosyvoice_pb2.NormalizeTextResponse.Segment(
                        text=text,
                        tokens=tokens,
                        length=token_len.item()
                    )
                )
            
            return cosyvoice_pb2.NormalizeTextResponse(segments=segments)
        except Exception as e:
            logger.error(f"文本标准化失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.NormalizeTextResponse()

    def GenerateTTSTokens(self, request, context):
        try:
            # 打印输入数据的形状和类型
            logger.info("========== GenerateTTSTokens 输入数据 ==========")
            logger.info(f"prompt_text: shape={len(request.tts_token_context.prompt_text)}")
            logger.info(f"text_segments: count={len(request.text_segments)}")
            logger.info("=============================================")
            
            # 从features对象中提取所需数据
            tts_token_context = request.tts_token_context
            if hasattr(tts_token_context, 'features') and tts_token_context.features:
                features = tts_token_context.features
                prompt_speech_token = features.prompt_speech_token
                prompt_speech_token_len = features.prompt_speech_token_len
                embedding = features.embedding
            else:
                # 使用默认值
                prompt_speech_token = []
                prompt_speech_token_len = 0
                embedding = []
            
            segments = []
            total_duration = 0
            
            # 处理每个文本段
            for i, text_segment in enumerate(request.text_segments):
                seg_uuid = f"{request.uuid}_seg_{i}"
                
                try:
                    # 转换文本tokens为tensor
                    text_tensor = torch.tensor(text_segment, dtype=torch.int32).reshape(1, -1)
                    
                    # 转换特征数据为tensor
                    prompt_speech_token_tensor = torch.tensor(prompt_speech_token, dtype=torch.int32).reshape(1, -1)
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
                    
                    # 调用llm_job
                    self.model.llm_job(
                        text=text_tensor,
                        prompt_text=torch.tensor(tts_token_context.prompt_text, dtype=torch.int32).reshape(1, -1),
                        llm_prompt_speech_token=prompt_speech_token_tensor,
                        llm_embedding=embedding_tensor,
                        uuid=seg_uuid
                    )
                    
                    # 获取生成结果
                    tokens = self.model.tts_speech_token_dict[seg_uuid]
                    segments.append(
                        cosyvoice_pb2.GenerateTTSTokensResponse.Segment(
                            uuid=seg_uuid,
                            tokens=tokens
                        )
                    )
                    
                    # 估算时长
                    total_duration += len(tokens) / 25.0 * 1000  # 25Hz采样率
                    
                finally:
                    # 清理模型状态
                    if seg_uuid in self.model.tts_speech_token_dict:
                        self.model.tts_speech_token_dict.pop(seg_uuid)
            
            return cosyvoice_pb2.GenerateTTSTokensResponse(
                segments=segments,
                duration_ms=total_duration
            )
        except Exception as e:
            logger.error(f"TTS Token生成失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.GenerateTTSTokensResponse()

    def Token2Wav(self, request, context):
        try:
            # 打印输入数据的形状和类型
            logger.info("========== Token2Wav 输入数据 ==========")
            logger.info(f"prompt_token: shape={len(request.speaker.prompt_token)}")
            logger.info(f"prompt_feat: shape={len(request.speaker.prompt_feat)}")
            logger.info(f"embedding: shape={len(request.speaker.embedding)}")
            logger.info(f"tokens_list: count={len(request.tokens_list)}")
            logger.info(f"speed: value={request.speed}")
            logger.info("=======================================")
            
            # 转换输入数据为正确的格式
            prompt_token = torch.tensor(request.speaker.prompt_token, dtype=torch.int32).reshape(1, -1)
            prompt_feat = torch.tensor(request.speaker.prompt_feat, dtype=torch.float32).reshape(1, -1, 80)
            embedding = torch.tensor(request.speaker.embedding, dtype=torch.float32)
            
            # 初始化最终音频
            final_audio = np.zeros(0, dtype=np.float32)
            
            # 获取token列表
            tokens_list = request.tokens_list
            uuids_list = request.uuids_list
            
            # 逐段生成音频
            for tokens, segment_uuid in zip(tokens_list, uuids_list):
                # 转换tokens为tensor
                token_tensor = torch.tensor(tokens, dtype=torch.int32).reshape(1, -1)
                
                # 初始化模型状态
                self.model.flow_cache_dict[segment_uuid] = torch.zeros(1, 80, 0, 2)
                self.model.hift_cache_dict[segment_uuid] = None
                self.model.mel_overlap_dict[segment_uuid] = torch.zeros(1, 80, 0)
                
                try:
                    # 生成当前段音频
                    segment_audio = self.model.token2wav(
                        token=token_tensor,
                        prompt_token=prompt_token,
                        prompt_feat=prompt_feat,
                        embedding=embedding,
                        uuid=segment_uuid,
                        token_offset=0,
                        finalize=True,
                        speed=request.speed
                    )
                    
                    # 处理音频格式
                    segment_audio = segment_audio.cpu().numpy()
                    if segment_audio.ndim > 1:
                        segment_audio = segment_audio.mean(axis=0)
                    
                    # 拼接到最终音频
                    final_audio = np.concatenate([final_audio, segment_audio])
                    
                finally:
                    # 清理模型状态
                    self.model.flow_cache_dict.pop(segment_uuid, None)
                    self.model.hift_cache_dict.pop(segment_uuid, None)
                    self.model.mel_overlap_dict.pop(segment_uuid, None)
            
            # 转换为int16
            audio_int16 = (final_audio * (2**15)).astype(np.int16)
            
            return cosyvoice_pb2.Token2WavResponse(
                audio=audio_int16.tobytes(),
                duration_sec=len(final_audio) / self.sample_rate
            )
        except Exception as e:
            logger.error(f"音频生成失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.Token2WavResponse()

    def ExtractSpeakerFeatures(self, request, context):
        """提取说话人特征，返回原始特征"""
        try:
            # 转换音频格式
            audio_np = np.frombuffer(request.audio, dtype=np.int16).astype(np.float32) / (2**15)
            audio = torch.from_numpy(audio_np).unsqueeze(0)
            
            # 提取特征
            features = self.frontend.frontend_cross_lingual("", audio, request.sample_rate)
            
            # 打印特征形状信息
            logger.info("========== ExtractSpeakerFeatures 特征形状 ==========")
            logger.info(f"llm_embedding: shape={features['llm_embedding'].shape}")
            logger.info(f"prompt_speech_feat: shape={features['prompt_speech_feat'].shape}")
            logger.info(f"flow_prompt_speech_token: shape={features['flow_prompt_speech_token'].shape}")
            logger.info("=================================================")
            
            # 转换为numpy array并转为list
            embedding = features['llm_embedding'].cpu().numpy().ravel().tolist()
            prompt_speech_feat = features['prompt_speech_feat'].cpu().numpy().ravel().tolist()
            prompt_speech_token = features['flow_prompt_speech_token'].cpu().numpy().ravel().tolist()
            
            # 使用顶层Features消息类型
            features_proto = cosyvoice_pb2.Features(
                embedding=embedding,
                prompt_speech_feat=prompt_speech_feat,
                prompt_speech_token=prompt_speech_token,
                prompt_speech_feat_len=int(features['prompt_speech_feat_len'].item()),
                prompt_speech_token_len=int(features['flow_prompt_speech_token_len'].item())
            )
            
            # 返回响应
            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse(
                features=features_proto
            )
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return cosyvoice_pb2.ExtractSpeakerFeaturesResponse()


def serve(args):
    """启动服务
    Args:
        args: 命令行参数，包含host和port
    """
    host = args.host if hasattr(args, 'host') else '[::]'
    port = args.port if hasattr(args, 'port') else 50052
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cosyvoice_pb2_grpc.add_CosyVoiceServiceServicer_to_server(
        CosyVoiceServiceServicer(args.model_dir), server
    )
    
    # 使用标准的IPv4格式
    if host == '[:::]' or host == '[::]':
        host = '0.0.0.0'
        
    address = f'{host}:{port}'
    server.add_insecure_port(address)
    server.start()
    logger.info(f'CosyVoice服务启动于 {address}')
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
