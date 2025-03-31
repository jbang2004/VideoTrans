import os
import sys
import logging
import torch
import numpy as np
import librosa
from typing import List, Optional
import ray
from ray import serve
import uuid
import asyncio
from config import Config

@serve.deployment(
    name="model_in_maker"
)
class ModelInMaker:
    def __init__(self):
        """初始化Frontend模块，独立加载CosyVoiceFrontEnd"""
        self.logger = logging.getLogger(__name__)
        
        # 使用配置中的目标采样率和其他配置
        self.config = Config()
        self.speaker_cache = {}  # 存储speaker_id到特征文件路径的映射
        self.max_val = 0.8
        
        # 添加系统路径
        for path in self.config.SYSTEM_PATHS:
            if path not in sys.path:
                sys.path.append(path)
                self.logger.info(f"添加系统路径: {path}")
        
        # 创建特征存储目录 - 修改为使用TASKS_DIR
        self.feature_dir = os.path.join(str(self.config.TASKS_DIR), "features")
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # 加载Frontend模块
        self._load_frontend()
        
    def _load_frontend(self):
        """加载CosyVoiceFrontEnd模块"""
        try:
            # 直接使用Python模块导入方式
            from models.CosyVoice.cosyvoice.cli.frontend import CosyVoiceFrontEnd
            
            # 获取配置文件路径 - 使用相对路径
            yaml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "models", "CosyVoice", "pretrained_models", 
                                     "CosyVoice2-0.5B", "cosyvoice.yaml")
            
            # 检查文件存在性
            if not os.path.exists(yaml_path):
                self.logger.error(f"找不到CosyVoice配置文件: {yaml_path}")
                raise FileNotFoundError(f"找不到CosyVoice配置文件: {yaml_path}")
            
            # 构建其他文件的路径 - 基于同一目录
            model_dir = os.path.dirname(yaml_path)
            
            # 加载配置 - 添加overrides参数，模拟原始CosyVoice2行为
            with open(yaml_path, 'r') as f:
                from hyperpyyaml import load_hyperpyyaml
                configs = load_hyperpyyaml(f, overrides={
                    'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')
                })
            
            self.frontend = CosyVoiceFrontEnd(
                configs['get_tokenizer'],
                configs['feat_extractor'],
                f'{model_dir}/campplus.onnx',
                f'{model_dir}/speech_tokenizer_v2.onnx',
                f'{model_dir}/spk2info.pt',
                configs['allowed_special']
            )
            
            self.sample_rate = configs['sample_rate']
            self.logger.info("Frontend模块加载完成")
        except Exception as e:
            self.logger.error(f"Frontend模块加载失败: {str(e)}")
            raise

    def _update_text_features(self, sentence):
        """
        更新文本特征（保存到本地文件）
        """
        try:
            tts_text = sentence.trans_text
            
            # 文本正则化
            normalized_segments = self.frontend.text_normalize(tts_text, split=True)
            
            # 生成特征ID并保存到文件
            text_feature_id = str(uuid.uuid4())
            feature_path = os.path.join(self.feature_dir, f"text_{text_feature_id}.pt")
            
            # 提取文本token
            segment_tokens = []
            segment_token_lens = []
            
            for seg in normalized_segments:
                txt, txt_len = self.frontend._extract_text_token(seg)
                segment_tokens.append(txt)
                segment_token_lens.append(txt_len)
            
            # 保存到文件
            text_features = {
                'text': segment_tokens,
                'text_len': segment_token_lens,
                'normalized_text_segments': normalized_segments
            }
            torch.save(text_features, feature_path)
            
            # 保存文本特征路径
            sentence.model_input['text_feature_path'] = feature_path
            sentence.model_input['normalized_text_segments'] = normalized_segments

            self.logger.debug(f"成功更新文本特征: {normalized_segments}")
            return sentence
        except Exception as e:
            self.logger.error(f"更新文本特征失败: {str(e)}")
            raise

    def _modelin_sentence(self, sentence, reuse_speaker=False):
        """
        对单个句子进行模型输入处理（保存到本地文件）
        """
        speaker_id = sentence.speaker_id

        # 1) Speaker处理
        if not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                try:
                    # 准备音频
                    audio = sentence.audio
                    
                    # 创建处理后的音频文件
                    audio_id = str(uuid.uuid4())
                    processed_audio_path = os.path.join(self.feature_dir, f"audio_{audio_id}.pt")
                    
                    # 处理音频
                    speech_tensor = audio
                    processed_audio, _ = librosa.effects.trim(
                        speech_tensor.numpy().flatten(), 
                        top_db=60,
                        frame_length=440,
                        hop_length=220
                    )
                    processed_audio = torch.tensor(processed_audio).unsqueeze(0)
                    
                    if processed_audio.abs().max() > self.max_val:
                        processed_audio = processed_audio / processed_audio.abs().max() * self.max_val
                    
                    processed_audio = torch.concat([processed_audio, torch.zeros(1, int(self.sample_rate * 0.2))], dim=1)
                    
                    # 保存处理后的音频
                    torch.save(processed_audio, processed_audio_path)
                    
                    # 提取说话人特征
                    speaker_feature_id = str(uuid.uuid4())
                    speaker_feature_path = os.path.join(self.feature_dir, f"speaker_{speaker_feature_id}.pt")
                    
                    # 使用空文本进行跨语言特征提取
                    speaker_features = self.frontend.frontend_cross_lingual(
                        "",
                        processed_audio,
                        self.sample_rate
                    )
                    
                    # 保存说话人特征
                    torch.save(speaker_features, speaker_feature_path)
                    
                    # 保存特征路径到本地缓存
                    self.speaker_cache[speaker_id] = speaker_feature_path
                    
                    # 显式删除大型音频数据，释放内存
                    del processed_audio
                    del speech_tensor
                    
                except Exception as e:
                    self.logger.error(f"模型输入音频处理失败: {str(e)}")
                    raise
                
            # 保存speaker_feature_path到sentence
            sentence.model_input['speaker_feature_path'] = self.speaker_cache[speaker_id]
            
            # 特征提取完成后删除audio数据，释放内存
            if hasattr(sentence, 'audio') and sentence.audio is not None:
                del sentence.audio
                sentence.audio = None
                self.logger.debug(f"已删除句子的原始音频数据，释放内存")

        # 2) 文本特征更新
        return self._update_text_features(sentence)

    async def modelin_maker(self, sentences, reuse_speaker=False, batch_size=3):
        """
        对一批 sentences 做 model_in 处理，分批 yield
        """
        if not sentences:
            self.logger.warning("modelin_maker: 收到空的句子列表")
            return

        self.logger.debug(f"模型输入处理 {len(sentences)} 个句子")

        results = []
        for i, s in enumerate(sentences, start=1):
            # 使用asyncio.to_thread包装同步函数调用
            modelin_sentence = await asyncio.to_thread(self._modelin_sentence, s, reuse_speaker)
            results.append(modelin_sentence)

            if i % batch_size == 0:
                yield results
                results = []

        if results:
            yield results

        if not reuse_speaker:
            self.speaker_cache.clear()
            self.logger.debug("modelin_maker: 已清理本地speaker_cache映射")