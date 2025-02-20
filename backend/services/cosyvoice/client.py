import grpc
import numpy as np
import torch
import logging

from .proto import cosyvoice_pb2
from .proto import cosyvoice_pb2_grpc

logger = logging.getLogger(__name__)

class CosyVoiceClient:
    def __init__(self, address="localhost:50052"):
        self.channel = grpc.insecure_channel(address)
        self.stub = cosyvoice_pb2_grpc.CosyVoiceServiceStub(self.channel)

    def normalize_text(self, text: str):
        """文本标准化"""
        try:
            req = cosyvoice_pb2.NormalizeTextRequest(text=text)
            resp = self.stub.NormalizeText(req)
            return resp.features
        except Exception as e:
            logger.error(f"NormalizeText调用失败: {e}")
            raise

    def extract_speaker_features(self, audio_tensor: torch.Tensor, sr=24000):
        """提取说话人特征"""
        try:
            # 保持原始格式，直接转换为numpy数组并转为二进制
            audio_np = audio_tensor.squeeze(0).cpu().numpy()
            req = cosyvoice_pb2.ExtractSpeakerFeaturesRequest(
                audio=audio_np.tobytes(),
                sample_rate=sr
            )
            resp = self.stub.ExtractSpeakerFeatures(req)
            return resp.features
        except Exception as e:
            logger.error(f"ExtractSpeakerFeatures调用失败: {e}")
            raise

    def generate_tts_tokens(self, features, uuid=""):
        """生成TTS tokens(分段)"""
        try:
            req = cosyvoice_pb2.GenerateTTSTokensRequest(features=features, uuid=uuid)
            resp = self.stub.GenerateTTSTokens(req)
            return resp.features, resp.duration_ms
        except Exception as e:
            logger.error(f"GenerateTTSTokens调用失败: {e}")
            raise

    def token2wav(self, features, speed=1.0):
        """tokens 转音频"""
        try:
            req = cosyvoice_pb2.Token2WavRequest(features=features, speed=speed)
            resp = self.stub.Token2Wav(req)
            audio_np = np.frombuffer(resp.audio, dtype=np.int16).astype(np.float32) / (2**15)
            return audio_np, resp.duration_sec
        except Exception as e:
            logger.error(f"Token2Wav调用失败: {e}")
            raise
