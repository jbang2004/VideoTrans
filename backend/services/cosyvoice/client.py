import grpc
import numpy as np
import torch
from .proto import cosyvoice_pb2
from .proto import cosyvoice_pb2_grpc
import logging

logger = logging.getLogger(__name__)

class CosyVoiceClient:
    def __init__(self, address="localhost:50052"):
        self.channel = grpc.insecure_channel(address)
        self.stub = cosyvoice_pb2_grpc.CosyVoiceServiceStub(self.channel)
        
    def normalize_text(self, text: str) -> dict:
        """文本标准化，直接返回服务端响应"""
        try:
            response = self.stub.NormalizeText(
                cosyvoice_pb2.NormalizeTextRequest(text=text)
            )
            return {
                'normalized_text_segments': [seg.text for seg in response.segments],
                'text': [seg.tokens for seg in response.segments],
                'text_len': [seg.length for seg in response.segments]
            }
        except Exception as e:
            logger.error(f"文本标准化失败: {e}")
            raise

    def generate_tts_tokens(self, uuid: str, text_segments: list, tts_token_context: dict) -> dict:
        """生成TTS Tokens，直接传递数据"""
        try:
            # 打印输入数据
            logger.info("========== GenerateTTSTokens Client输入数据 ==========")
            logger.info(f"uuid: {uuid}")
            logger.info(f"text_segments type: {type(text_segments)}")
            for i, seg in enumerate(text_segments):
                logger.info(f"segment {i} type: {type(seg)}, content: {seg[:10]}...")
            logger.info(f"tts_token_context keys: {tts_token_context.keys()}")
            logger.info("=================================================")

            # 创建上下文，直接传递features
            context_proto = cosyvoice_pb2.GenerateTTSTokensRequest.TTSTokenContext(
                prompt_text=tts_token_context.get('prompt_text', []),
                prompt_text_len=tts_token_context.get('prompt_text_len', 0),
                features=tts_token_context.get('features', None)
            )
            
            # 创建请求并发送
            request = cosyvoice_pb2.GenerateTTSTokensRequest(
                uuid=str(uuid),
                text_segments=text_segments,
                tts_token_context=context_proto
            )
            
            response = self.stub.GenerateTTSTokens(request)
            return {
                'segments': [{
                    'uuid': seg.uuid,
                    'tokens': seg.tokens
                } for seg in response.segments],
                'duration_ms': response.duration_ms
            }
        except Exception as e:
            logger.error(f"TTS Token生成失败: {e}")
            raise

    def token2wav(self, tokens_list: list, uuids_list: list, speaker_info: dict, speed: float = 1.0) -> dict:
        """Token转换为音频，直接传递数据"""
        try:
            speaker_proto = cosyvoice_pb2.Token2WavRequest.SpeakerInfo(
                prompt_token=speaker_info.get('prompt_token', []),
                prompt_feat=speaker_info.get('prompt_feat', []),
                embedding=speaker_info.get('embedding', [])
            )
            
            response = self.stub.Token2Wav(
                cosyvoice_pb2.Token2WavRequest(
                    tokens_list=tokens_list,
                    uuids_list=uuids_list,
                    speed=speed,
                    speaker=speaker_proto
                )
            )
            
            return {
                'audio': np.frombuffer(response.audio, dtype=np.int16).astype(np.float32) / (2**15),
                'duration_sec': response.duration_sec
            }
        except Exception as e:
            logger.error(f"音频生成失败: {e}")
            raise

    def extract_speaker_features(self, audio_tensor: torch.Tensor) -> dict:
        """提取说话人特征，直接返回服务端响应的features对象"""
        try:
            audio_np = audio_tensor.squeeze(0).cpu().numpy()
            audio_int16 = (audio_np * (2**15)).astype(np.int16).tobytes()
            
            response = self.stub.ExtractSpeakerFeatures(
                cosyvoice_pb2.ExtractSpeakerFeaturesRequest(
                    audio=audio_int16,
                    sample_rate=24000
                )
            )
            
            return {
                'features': response.features  # 直接返回整个features对象
            }
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            raise 