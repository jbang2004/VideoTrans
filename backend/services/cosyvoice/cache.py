import threading
from typing import Dict, Optional, List
import torch

class FeatureData:
    """服务端特征数据存储结构，直接存储模型输出的数据，保持原有维度"""
    def __init__(self):
        # 文本相关
        self.normalized_texts: List[str] = []
        self.text_tokens: List[torch.Tensor] = []  # [batch_size, seq_len]
        
        # 说话人相关 - 保持原有维度
        self.embedding: Optional[torch.Tensor] = None  # [1, embedding_dim]
        self.prompt_speech_feat: Optional[torch.Tensor] = None  # [1, time, 80]
        self.prompt_speech_feat_len: int = 0
        self.prompt_speech_token: Optional[torch.Tensor] = None  # [1, seq_len]
        self.prompt_speech_token_len: int = 0
        
        # TTS tokens
        self.tts_tokens: List[List[int]] = []  # 直接存储模型生成的token列表
        self.tts_segment_uuids: List[str] = []

class FeaturesCache:
    def __init__(self):
        self._cache: Dict[str, FeatureData] = {}
        self._lock = threading.Lock()
    
    def get(self, uuid: str) -> Optional[FeatureData]:
        """获取特征数据"""
        with self._lock:
            return self._cache.get(uuid)
    
    def set(self, uuid: str, data: FeatureData) -> None:
        """设置特征数据"""
        with self._lock:
            self._cache[uuid] = data
    
    def update_text_features(self, uuid: str, normalized_texts: List[str], text_tokens: List[torch.Tensor]) -> bool:
        """更新文本特征，text_tokens应该已经是正确的维度 [batch_size, seq_len]"""
        with self._lock:
            if uuid not in self._cache:
                data = FeatureData()
                data.normalized_texts = normalized_texts
                data.text_tokens = text_tokens
                self._cache[uuid] = data
            else:
                self._cache[uuid].normalized_texts = normalized_texts
                self._cache[uuid].text_tokens = text_tokens
            return True
    
    def update_speaker_features(self, uuid: str, 
                              embedding: torch.Tensor,  # [1, embedding_dim]
                              prompt_feat: torch.Tensor,  # [1, time, 80]
                              prompt_feat_len: int,
                              prompt_token: torch.Tensor,  # [1, seq_len]
                              prompt_token_len: int) -> bool:
        """更新说话人特征，保持输入tensor的原有维度"""
        with self._lock:
            if uuid not in self._cache:
                return False
            data = self._cache[uuid]
            data.embedding = embedding
            data.prompt_speech_feat = prompt_feat
            data.prompt_speech_feat_len = prompt_feat_len
            data.prompt_speech_token = prompt_token
            data.prompt_speech_token_len = prompt_token_len
            return True
    
    def update_tts_tokens(self, uuid: str, tokens: List[List[int]], segment_uuids: List[str]) -> bool:
        """更新TTS tokens"""
        with self._lock:
            if uuid not in self._cache:
                return False
            data = self._cache[uuid]
            data.tts_tokens = tokens
            data.tts_segment_uuids = segment_uuids
            return True
    
    def delete(self, uuid: str) -> None:
        """删除特征数据"""
        with self._lock:
            self._cache.pop(uuid, None)
    
    def exists(self, uuid: str) -> bool:
        """检查UUID是否存在"""
        with self._lock:
            return uuid in self._cache