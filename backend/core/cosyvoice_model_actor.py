import ray
import torch
import logging
import threading
import sys
import uuid
import numpy as np

@ray.remote(num_gpus=0.9)
class CosyVoiceModelActor:
    """
    将CosyVoice模型封装为Ray Actor，提供所有与模型相关的操作
    """
    def __init__(self, model_path):
        """初始化CosyVoice模型服务"""
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化CosyVoice模型Actor: {model_path}")
        
        # 导入Config并设置系统路径
        from config import Config
        config = Config()
        
        # 添加系统路径，与api.py中相同
        for path in config.SYSTEM_PATHS:
            if path not in sys.path:
                sys.path.append(path)
                self.logger.info(f"添加系统路径: {path}")
        
        # 导入并加载模型
        try:
            from models.CosyVoice.cosyvoice.cli.cosyvoice import CosyVoice2
            self.cosyvoice = CosyVoice2(model_path)
            self.model = self.cosyvoice.model
            self.sample_rate = self.cosyvoice.sample_rate
            self.frontend = self.cosyvoice.frontend
            
            self.logger.info("CosyVoice模型加载完成")
        except Exception as e:
            self.logger.error(f"CosyVoice模型加载失败: {str(e)}")
            self.logger.error(f"当前系统路径: {sys.path}")
            raise
            
        # 添加特征缓存
        self.feature_cache_lock = threading.Lock()
        self.speaker_feature_cache = {}  # 说话人特征缓存
        self.text_feature_cache = {}     # 文本特征缓存
        self.processed_audio_cache = {}  # 处理后的音频缓存
        self.tts_token_cache = {}        # TTS token特征缓存
        
    def get_model_info(self):
        """获取模型基本信息"""
        return {
            "sample_rate": self.sample_rate,
        }
        
    def get_sample_rate(self):
        """返回模型采样率"""
        return self.sample_rate
        
    def get_frontend(self):
        """获取模型frontend（用于ModelIn）"""
        return self.frontend
        
    def postprocess_audio(self, speech_tensor, top_db=60, hop_length=220, win_length=440, max_val=0.8):
        """处理音频"""
        import librosa
        speech, _ = librosa.effects.trim(
            speech_tensor, 
            top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > max_val:
            speech = speech / speech.abs().max() * max_val
        
        speech = torch.concat([speech, torch.zeros(1, int(self.sample_rate * 0.2))], dim=1)
        return speech
        
    def process_and_cache_audio(self, audio, max_val=0.8):
        """处理音频并缓存结果，返回缓存ID"""
        # 生成唯一ID
        cache_id = str(uuid.uuid4())
        
        # 处理音频
        processed_audio = self.postprocess_audio(audio, max_val=max_val)
        
        # 缓存处理后的音频
        with self.feature_cache_lock:
            self.processed_audio_cache[cache_id] = processed_audio
            
        return cache_id
        
    def extract_speaker_features_and_cache(self, processed_audio_id):
        """提取说话人特征并缓存，返回缓存ID"""
        # 生成唯一ID
        cache_id = str(uuid.uuid4())
        
        # 从缓存获取处理后的音频
        with self.feature_cache_lock:
            if processed_audio_id not in self.processed_audio_cache:
                raise ValueError(f"处理后的音频ID不存在: {processed_audio_id}")
            processed_audio = self.processed_audio_cache[processed_audio_id]
        
        # 提取说话人特征
        speaker_features = self.frontend.frontend_cross_lingual(
            "",
            processed_audio,
            self.sample_rate
        )
        
        # 缓存说话人特征
        with self.feature_cache_lock:
            self.speaker_feature_cache[cache_id] = speaker_features
            
        return cache_id
        
    def get_speaker_features(self, cache_id):
        """获取缓存的说话人特征"""
        with self.feature_cache_lock:
            if cache_id not in self.speaker_feature_cache:
                raise ValueError(f"说话人特征ID不存在: {cache_id}")
            return self.speaker_feature_cache[cache_id]
            
    def normalize_text(self, text, split=True):
        """文本正则化"""
        return self.frontend.text_normalize(text, split=split)
        
    def extract_text_tokens_and_cache(self, text):
        """提取文本token并缓存，返回缓存ID和规范化的文本段落"""
        # 生成唯一ID
        cache_id = str(uuid.uuid4())
        
        # 文本正则化
        normalized_segments = self.normalize_text(text, split=True)
        
        # 提取文本token
        segment_tokens = []
        segment_token_lens = []
        
        for seg in normalized_segments:
            txt, txt_len = self.extract_text_tokens(seg)
            segment_tokens.append(txt)
            segment_token_lens.append(txt_len)
            
        # 缓存文本特征
        with self.feature_cache_lock:
            self.text_feature_cache[cache_id] = {
                'text': segment_tokens,
                'text_len': segment_token_lens,
                'normalized_text_segments': normalized_segments
            }
            
        return cache_id, normalized_segments
        
    def get_text_features(self, cache_id):
        """获取缓存的文本特征"""
        with self.feature_cache_lock:
            if cache_id not in self.text_feature_cache:
                raise ValueError(f"文本特征ID不存在: {cache_id}")
            return self.text_feature_cache[cache_id]
        
    def extract_text_tokens(self, text):
        """提取文本token"""
        return self.frontend._extract_text_token(text)
    
    # ==== TTS Token生成功能 ====
    def generate_tts_tokens_and_cache(self, text_feature_id, speaker_feature_id, tts_token_id=None):
        """
        生成TTS Tokens并缓存，返回缓存ID和时长
        如果提供了tts_token_id，则复用此ID，否则生成新ID
        """
        # 生成或复用唯一ID
        cache_id = tts_token_id if tts_token_id else str(uuid.uuid4())
        
        # 从缓存获取特征
        with self.feature_cache_lock:
            if text_feature_id not in self.text_feature_cache:
                raise ValueError(f"文本特征ID不存在: {text_feature_id}")
            if speaker_feature_id not in self.speaker_feature_cache:
                raise ValueError(f"说话人特征ID不存在: {speaker_feature_id}")
                
            text_features = self.text_feature_cache[text_feature_id]
            speaker_features = self.speaker_feature_cache[speaker_feature_id]
            
            # 如果已存在此ID的TTS token，先清理
            if cache_id in self.tts_token_cache:
                # 清理所有相关的segment uuid
                for seg_uuid in self.tts_token_cache[cache_id]['segment_uuids']:
                    self.cleanup_tts_token(seg_uuid)
            
        # 准备参数
        segment_tokens_list = []
        segment_uuids = []
        
        for i, (text, text_len) in enumerate(zip(text_features['text'], text_features['text_len'])):
            current_seg_uuid = f"{cache_id}_seg_{i}"
            
            with self.model.lock:
                self.model.tts_speech_token_dict[current_seg_uuid] = []
                self.model.llm_end_dict[current_seg_uuid] = False
                if hasattr(self.model, 'mel_overlap_dict'):
                    self.model.mel_overlap_dict[current_seg_uuid] = None
                self.model.hift_cache_dict[current_seg_uuid] = None
                
            # 从speaker_features中获取必要的参数
            prompt_text = speaker_features.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32))
            llm_prompt_speech_token = speaker_features.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32))
            llm_embedding = speaker_features.get('llm_embedding', torch.zeros(0, 192))
            
            # 调用模型生成token
            self.model.llm_job(
                text,
                prompt_text,
                llm_prompt_speech_token,
                llm_embedding,
                current_seg_uuid
            )
            
            # 获取生成的tokens
            tokens = self.model.tts_speech_token_dict[current_seg_uuid]
            segment_tokens_list.append(tokens)
            segment_uuids.append(current_seg_uuid)
        
        # 计算总token数和时长
        total_token_count = sum(len(tokens) for tokens in segment_tokens_list)
        total_duration_ms = total_token_count / 25 * 1000  # 25Hz转换为毫秒
        
        # 缓存TTS token特征
        with self.feature_cache_lock:
            self.tts_token_cache[cache_id] = {
                'segment_speech_tokens': segment_tokens_list,
                'segment_uuids': segment_uuids,
                'duration': total_duration_ms
            }
            
        return cache_id, total_duration_ms
    
    def get_tts_tokens(self, cache_id):
        """获取缓存的TTS token特征"""
        with self.feature_cache_lock:
            if cache_id not in self.tts_token_cache:
                raise ValueError(f"TTS token特征ID不存在: {cache_id}")
            return self.tts_token_cache[cache_id]
    
    def cleanup_tts_token(self, seg_uuid):
        """清理单个TTS token缓存"""
        with self.model.lock:
            self.model.tts_speech_token_dict.pop(seg_uuid, None)
            self.model.llm_end_dict.pop(seg_uuid, None)
            self.model.hift_cache_dict.pop(seg_uuid, None)
            if hasattr(self.model, 'mel_overlap_dict'):
                self.model.mel_overlap_dict.pop(seg_uuid, None)
    
    def cleanup_tts_tokens(self, cache_id):
        """清理TTS token缓存"""
        with self.feature_cache_lock:
            if cache_id in self.tts_token_cache:
                # 清理所有相关的segment uuid
                for seg_uuid in self.tts_token_cache[cache_id]['segment_uuids']:
                    self.cleanup_tts_token(seg_uuid)
                # 从缓存中移除
                self.tts_token_cache.pop(cache_id, None)
    
    # ==== 音频生成功能 ====
    def generate_audio(self, tts_token_id, speaker_feature_id, speed=1.0):
        """生成音频，使用缓存的特征"""
        # 从缓存获取特征
        with self.feature_cache_lock:
            if tts_token_id not in self.tts_token_cache:
                raise ValueError(f"TTS token特征ID不存在: {tts_token_id}")
            if speaker_feature_id not in self.speaker_feature_cache:
                raise ValueError(f"说话人特征ID不存在: {speaker_feature_id}")
                
            tts_tokens = self.tts_token_cache[tts_token_id]
            speaker_features = self.speaker_feature_cache[speaker_feature_id]
            
        # 准备参数
        prompt_token = speaker_features.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32))
        prompt_feat = speaker_features.get('prompt_speech_feat', torch.zeros(1, 0, 80))
        embedding = speaker_features.get('flow_embedding', torch.zeros(0))
        
        segment_audio_list = []
        
        for i, (tokens, segment_uuid) in enumerate(zip(tts_tokens['segment_speech_tokens'], tts_tokens['segment_uuids'])):
            if not tokens:
                segment_audio_list.append(np.zeros(0, dtype=np.float32))
                continue
                
            # 准备token tensor
            token_tensor = torch.tensor(tokens).unsqueeze(dim=0)
            
            token2wav_kwargs = {
                'token': token_tensor,
                'token_offset': 0,
                'finalize': True,
                'prompt_token': prompt_token,
                'prompt_feat': prompt_feat,
                'embedding': embedding,
                'uuid': segment_uuid,
                'speed': speed
            }
            
            segment_output = self.model.token2wav(**token2wav_kwargs)
            segment_audio = segment_output.cpu().numpy()
            
            # 如果是多通道，转单通道
            if segment_audio.ndim > 1:
                segment_audio = segment_audio.mean(axis=0)
                
            segment_audio_list.append(segment_audio)
        
        # 拼接所有段落的音频
        if segment_audio_list:
            final_audio = np.concatenate(segment_audio_list)
        else:
            final_audio = np.zeros(0, dtype=np.float32)
            
        return final_audio
        
    def cleanup_feature_cache(self, cache_ids=None, skip_speaker_features=True):
        """
        清理特征缓存
        
        Args:
            cache_ids: 要清理的缓存ID列表，None表示清理所有缓存
            skip_speaker_features: 是否跳过清理说话人特征，默认为True
        """
        with self.feature_cache_lock:
            if cache_ids is None:
                # 清理所有缓存
                # 先清理TTS token缓存
                for cache_id in list(self.tts_token_cache.keys()):
                    self.cleanup_tts_tokens(cache_id)
                
                # 再清理其他缓存
                if not skip_speaker_features:
                    self.speaker_feature_cache.clear()
                    self.logger.info("已清理所有说话人特征缓存")
                
                self.text_feature_cache.clear()
                self.processed_audio_cache.clear()
                self.tts_token_cache.clear()
                self.logger.info(f"已清理所有特征缓存 (跳过说话人特征: {skip_speaker_features})")
            else:
                # 清理指定的缓存
                for cache_id in cache_ids:
                    # 检查是否是TTS token缓存
                    if cache_id in self.tts_token_cache:
                        self.cleanup_tts_tokens(cache_id)
                    
                    # 清理其他缓存，可选跳过说话人特征
                    if not skip_speaker_features:
                        self.speaker_feature_cache.pop(cache_id, None)
                    
                    self.text_feature_cache.pop(cache_id, None)
                    self.processed_audio_cache.pop(cache_id, None)
                
                self.logger.info(f"已清理指定的特征缓存: {cache_ids} (跳过说话人特征: {skip_speaker_features})") 