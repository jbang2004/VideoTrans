import threading
from typing import Dict, Optional, List, Any
import torch

class TextFeatureData:
    """存储文本特征数据"""
    def __init__(self):
        self.normalized_texts: List[str] = []
        self.text_tokens: List[torch.Tensor] = []  # [batch_size, seq_len]
        self.tts_tokens: List[List[int]] = []      # TTS tokens
        self.tts_segment_uuids: List[str] = []

class SpeakerFeatureData:
    """存储说话人特征数据，保持完整特征字典"""
    def __init__(self):
        self.features: Optional[Dict[str, Any]] = None  # 存储完整的特征字典

class FeaturesCache:
    def __init__(self):
        self._text_cache: Dict[str, TextFeatureData] = {}      # text_uuid -> TextFeatureData
        self._speaker_cache: Dict[str, SpeakerFeatureData] = {}  # speaker_uuid -> SpeakerFeatureData
        self._lock = threading.Lock()
    
    def get_text(self, text_uuid: str) -> Optional[TextFeatureData]:
        """获取文本特征数据"""
        with self._lock:
            return self._text_cache.get(text_uuid)
    
    def get_speaker(self, speaker_uuid: str) -> Optional[Dict[str, Any]]:
        """获取说话人特征字典"""
        with self._lock:
            data = self._speaker_cache.get(speaker_uuid)
            return data.features if data else None
    
    def set_text(self, text_uuid: str, data: TextFeatureData) -> None:
        """设置文本特征数据"""
        with self._lock:
            self._text_cache[text_uuid] = data
    
    def set_speaker(self, speaker_uuid: str, features: Dict[str, Any]) -> None:
        """设置说话人特征字典"""
        with self._lock:
            data = SpeakerFeatureData()
            data.features = features
            self._speaker_cache[speaker_uuid] = data
    
    def update_text_features(self, text_uuid: str, normalized_texts: List[str], text_tokens: List[torch.Tensor]) -> bool:
        """更新文本特征"""
        with self._lock:
            if text_uuid not in self._text_cache:
                data = TextFeatureData()
                data.normalized_texts = normalized_texts
                data.text_tokens = text_tokens
                self._text_cache[text_uuid] = data
            else:
                self._text_cache[text_uuid].normalized_texts = normalized_texts
                self._text_cache[text_uuid].text_tokens = text_tokens
            return True
    
    def update_speaker_features(self, speaker_uuid: str, features: Dict[str, Any]) -> bool:
        """更新说话人特征字典"""
        with self._lock:
            data = SpeakerFeatureData()
            data.features = features
            self._speaker_cache[speaker_uuid] = data
            return True
    
    def update_tts_tokens(self, text_uuid: str, tokens: List[List[int]], segment_uuids: List[str]) -> bool:
        """更新TTS tokens"""
        with self._lock:
            if text_uuid not in self._text_cache:
                return False
            data = self._text_cache[text_uuid]
            data.tts_tokens = tokens
            data.tts_segment_uuids = segment_uuids
            return True
    
    def delete(self, uuid: str, is_speaker: bool = False) -> None:
        """删除特征数据"""
        with self._lock:
            if is_speaker:
                self._speaker_cache.pop(uuid, None)
            else:
                self._text_cache.pop(uuid, None)
    
    def exists_text(self, text_uuid: str) -> bool:
        """检查文本UUID是否存在"""
        with self._lock:
            return text_uuid in self._text_cache
    
    def exists_speaker(self, speaker_uuid: str) -> bool:
        """检查说话人UUID是否存在"""
        with self._lock:
            return speaker_uuid in self._speaker_cache