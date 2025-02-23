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
        文本标准化，返回文本UUID
        """
        try:
            req = cosyvoice_pb2.NormalizeTextRequest(text=text)
            resp = self.stub.NormalizeText(req)
            return resp.text_uuid
        except Exception as e:
            logger.error(f"NormalizeText调用失败: {e}")
            raise

    def extract_speaker_features(self, speaker_uuid: str, audio: np.ndarray, sr: int = 24000) -> bool:
        """
        提取说话人特征。
        :param speaker_uuid: 说话人标识
        :param audio: 输入音频（numpy.ndarray）
        :param sr: 采样率，默认24000
        :return: 是否成功
        """
        try:
            # 直接将numpy array转换为bytes
            audio_bytes = audio.tobytes()
            req = cosyvoice_pb2.ExtractSpeakerFeaturesRequest(
                speaker_uuid=speaker_uuid,
                audio=audio_bytes,
                sample_rate=sr
            )
            resp = self.stub.ExtractSpeakerFeatures(req)
            return resp.success
        except Exception as e:
            logger.error(f"ExtractSpeakerFeatures调用失败: {e}")
            raise

    def generate_tts_tokens(self, text_uuid: str, speaker_uuid: str) -> Tuple[int, bool]:
        """
        根据文本UUID和说话人UUID生成TTS tokens并返回预估时长
        """
        try:
            req = cosyvoice_pb2.GenerateTTSTokensRequest(text_uuid=text_uuid, speaker_uuid=speaker_uuid)
            resp = self.stub.GenerateTTSTokens(req)
            return resp.duration_ms, resp.success
        except Exception as e:
            logger.error(f"GenerateTTSTokens调用失败: {e}")
            raise

    def token2wav(self, text_uuid: str, speaker_uuid: str, speed: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        将文本UUID和说话人UUID转换为音频
        """
        try:
            req = cosyvoice_pb2.Token2WavRequest(text_uuid=text_uuid, speaker_uuid=speaker_uuid, speed=speed)
            resp = self.stub.Token2Wav(req)
            audio_np = np.frombuffer(resp.audio, dtype=np.int16).astype(np.float32) / (2**15)
            return audio_np, resp.duration_sec
        except Exception as e:
            logger.error(f"Token2Wav调用失败: {e}")
            raise

    def cleanup(self, uuid: str, is_speaker: bool = False) -> bool:
        """
        清理服务端缓存
        """
        try:
            req = cosyvoice_pb2.CleanupRequest(uuid=uuid, is_speaker=is_speaker)
            resp = self.stub.Cleanup(req)
            return resp.success
        except Exception as e:
            logger.error(f"Cleanup调用失败: {e}")
            raise