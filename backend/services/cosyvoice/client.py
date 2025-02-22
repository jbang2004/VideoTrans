import grpc
import numpy as np
import torch
import logging
from typing import Tuple, Optional

from .proto import cosyvoice_pb2
from .proto import cosyvoice_pb2_grpc

logger = logging.getLogger(__name__)

class CosyVoiceClient:
    def __init__(self, address="localhost:50052"):
        self.channel = grpc.insecure_channel(address)
        self.stub = cosyvoice_pb2_grpc.CosyVoiceServiceStub(self.channel)

    def normalize_text(self, text: str) -> str:
        """
        文本标准化，返回UUID
        """
        try:
            req = cosyvoice_pb2.NormalizeTextRequest(text=text)
            resp = self.stub.NormalizeText(req)
            return resp.uuid
        except Exception as e:
            logger.error(f"NormalizeText调用失败: {e}")
            raise

    def extract_speaker_features(self, uuid: str, audio_tensor: torch.Tensor, sr: int = 24000) -> bool:
        """
        提取说话人特征并更新服务端缓存
        """
        try:
            audio_np = audio_tensor.squeeze(0).cpu().numpy()
            req = cosyvoice_pb2.ExtractSpeakerFeaturesRequest(
                uuid=uuid,
                audio=audio_np.tobytes(),
                sample_rate=sr
            )
            resp = self.stub.ExtractSpeakerFeatures(req)
            return resp.success
        except Exception as e:
            logger.error(f"ExtractSpeakerFeatures调用失败: {e}")
            raise

    def generate_tts_tokens(self, uuid: str) -> Tuple[int, bool]:
        """
        生成TTS tokens并返回预估时长
        """
        try:
            req = cosyvoice_pb2.GenerateTTSTokensRequest(uuid=uuid)
            resp = self.stub.GenerateTTSTokens(req)
            return resp.duration_ms, resp.success
        except Exception as e:
            logger.error(f"GenerateTTSTokens调用失败: {e}")
            raise

    def token2wav(self, uuid: str, speed: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        将TTS tokens转换为音频
        """
        try:
            req = cosyvoice_pb2.Token2WavRequest(uuid=uuid, speed=speed)
            resp = self.stub.Token2Wav(req)
            audio_np = np.frombuffer(resp.audio, dtype=np.int16).astype(np.float32) / (2**15)
            return audio_np, resp.duration_sec
        except Exception as e:
            logger.error(f"Token2Wav调用失败: {e}")
            raise

    def cleanup(self, uuid: str) -> bool:
        """
        清理服务端缓存
        """
        try:
            req = cosyvoice_pb2.CleanupRequest(uuid=uuid)
            resp = self.stub.Cleanup(req)
            return resp.success
        except Exception as e:
            logger.error(f"Cleanup调用失败: {e}")
            raise
