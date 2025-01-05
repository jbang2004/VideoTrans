import logging
import torch
import numpy as np
import librosa
from typing import List
import torchaudio

class ModelIn:
    def __init__(self, cosy_model):
        self.cosy_frontend = cosy_model.frontend
        self.cosy_sample_rate = cosy_model.sample_rate
        self.last_speaker_id = None
        self.logger = logging.getLogger(__name__)
        self.speaker_cache = {}
        self.max_val = 0.8  # 最大音量阈值

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
        speech = torch.concat([speech, torch.zeros(1, int(self.cosy_sample_rate * 0.2))], dim=1)
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
                    
                    # 使用 frontend_cross_lingual 方法准备数据
                    self.speaker_cache[speaker_id] = self.cosy_frontend.frontend_cross_lingual(
                        "", # 空字符串作为占位符，实际文本稍后添加
                        prompt_speech_16k,
                        self.cosy_sample_rate
                    )
                
                # 获取缓存的基础数据
                cached_features = self.speaker_cache[speaker_id].copy()
                
                # 处理当前句子的文本
                tts_text = sentence.trans_text
                tts_text = self.cosy_frontend.text_normalize(tts_text, split=False)
                print("成功添加model_in句子:", tts_text)
                
                # 更新文本相关的特征
                text_token, text_token_len = self.cosy_frontend._extract_text_token(tts_text)
                cached_features['text'] = text_token
                cached_features['text_len'] = text_token_len
                
                # 添加uuid字段
                cached_features['tts_speech_token'] = []
                cached_features['uuid'] = ''

                sentence.model_input = cached_features
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