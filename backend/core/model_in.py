import logging
import torch
import numpy as np
import librosa
from typing import List
import torchaudio

class ModelIn:
    def __init__(self, cosy_frontend):
        self.cosy_frontend = cosy_frontend
        self.last_speaker_id = None
        self.logger = logging.getLogger(__name__)
        self.speaker_cache = {}
        self.max_val = 0.8  # 最大音量阈值
        self.target_sr = 22050  # 目标采样率

    def postprocess(self, speech, top_db=60, hop_length=220, win_length=440):
        """音频后处理
        
        Args:
            speech: 输入音频
            top_db: 静音判定阈值
            hop_length: 帧移步长
            win_length: 分析窗口长度
        """
        speech, _ = librosa.effects.trim(
            speech, top_db=top_db,
            frame_length=win_length,
            hop_length=hop_length
        )
        if speech.abs().max() > self.max_val:
            speech = speech / speech.abs().max() * self.max_val
        speech = torch.concat([speech, torch.zeros(1, int(self.target_sr * 0.2))], dim=1)
        return speech

    async def modelin_maker(self, sentences, batch_size: int = 3):
        """处理Sensevoice的输出
        
        Args:
            sentences: 待处理的句子列表
            batch_size: 批处理大小，默认3
        """
        if not sentences:
            self.logger.error("sensevoice_output为空")
            return

        try:
            modelined_batch = []
            
            for sentence in sentences:
                speaker_id = sentence.speaker_id

                # 检查缓存中是否已存在该说话人的处理结果
                if speaker_id not in self.speaker_cache:
                    prompt_speech = sentence.audio
                    # 使用 postprocess 处理音频,此时的sentence.audio是16k的
                    prompt_speech_16k = self.postprocess(prompt_speech)
                    prompt_speech_22050 = torchaudio.transforms.Resample(
                        orig_freq=16000, new_freq=22050)(prompt_speech_16k)
                    
                    self.speaker_cache[speaker_id] = {
                        'speech_token': self.cosy_frontend._extract_speech_token(prompt_speech_16k),
                        'embedding': self.cosy_frontend._extract_spk_embedding(prompt_speech_16k),
                        'speech_feat': self.cosy_frontend._extract_speech_feat(prompt_speech_22050)
                    }
                    
                cached_features = self.speaker_cache[speaker_id]
                speech_token, speech_token_len = cached_features['speech_token']
                embedding = cached_features['embedding']
                prompt_speech_feat, prompt_speech_feat_len = cached_features['speech_feat']

                # 文本处理
                tts_text = sentence.trans_text
                tts_text = self.cosy_frontend.text_normalize(tts_text, split=False)
                print("成功添加model_in句子:",tts_text)
                text_token, text_token_len = self.cosy_frontend._extract_text_token(tts_text)

                model_input = {
                    'text': text_token,
                    'text_len': text_token_len,
                    'flow_prompt_speech_token': speech_token,
                    'flow_prompt_speech_token_len': speech_token_len,
                    'prompt_speech_feat': prompt_speech_feat,
                    'prompt_speech_feat_len': prompt_speech_feat_len,
                    'llm_embedding': embedding,
                    'flow_embedding': embedding,
                    'tts_speech_token': [],
                    'uuid': '',
                }

                sentence.model_input = model_input
                modelined_batch.append(sentence)
                
                # 当批次达到指定大小时，yield整个批次
                if len(modelined_batch) >= batch_size:
                    yield modelined_batch
                    modelined_batch = []
            
            # 处理剩余的句子
            if modelined_batch:
                yield modelined_batch

        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            raise
            
        finally:
            # 处理完成后清理缓存
            self.speaker_cache.clear()