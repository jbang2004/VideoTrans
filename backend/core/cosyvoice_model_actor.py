import ray
import torch
import logging
import threading
import sys

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
        
    def extract_speaker_features(self, processed_audio):
        """提取说话人特征"""
        return self.frontend.frontend_cross_lingual(
            "",
            processed_audio,
            self.sample_rate
        )
        
    def normalize_text(self, text, split=True):
        """文本正则化"""
        return self.frontend.text_normalize(text, split=split)
        
    def extract_text_tokens(self, text):
        """提取文本token"""
        return self.frontend._extract_text_token(text)
    
    # ==== TTS Token生成功能 ====
    def generate_tts_tokens(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, seg_uuid):
        """生成TTS Tokens"""
        with self.model.lock:
            self.model.tts_speech_token_dict[seg_uuid] = []
            self.model.llm_end_dict[seg_uuid] = False
            if hasattr(self.model, 'mel_overlap_dict'):
                self.model.mel_overlap_dict[seg_uuid] = None
            self.model.hift_cache_dict[seg_uuid] = None
            
        # 调用模型生成token
        self.model.llm_job(
            text,
            prompt_text,
            llm_prompt_speech_token,
            llm_embedding,
            seg_uuid
        )
        
        # 返回生成的tokens
        return self.model.tts_speech_token_dict[seg_uuid]
    
    def cleanup_tts_tokens(self, seg_uuid):
        """清理TTS token缓存"""
        with self.model.lock:
            self.model.tts_speech_token_dict.pop(seg_uuid, None)
            self.model.llm_end_dict.pop(seg_uuid, None)
            self.model.hift_cache_dict.pop(seg_uuid, None)
            if hasattr(self.model, 'mel_overlap_dict'):
                self.model.mel_overlap_dict.pop(seg_uuid, None)
    
    # ==== 音频生成功能 ====
    def generate_audio(self, tokens, token_offset, uuid, prompt_token, prompt_feat, embedding, speed=1.0):
        """生成音频"""
        token2wav_kwargs = {
            'token': tokens,
            'token_offset': token_offset,
            'finalize': True,
            'prompt_token': prompt_token,
            'prompt_feat': prompt_feat,
            'embedding': embedding,
            'uuid': uuid,
            'speed': speed
        }
        
        segment_output = self.model.token2wav(**token2wav_kwargs)
        return segment_output.cpu().numpy() 