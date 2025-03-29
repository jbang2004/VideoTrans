import os
import sys
import logging
import torch
import numpy as np
import librosa
from typing import List, Optional, Dict, Any
import ray
from ray import serve
import uuid
import asyncio  # 添加 asyncio 导入
from config import Config

@serve.deployment(
    name="model_in_maker",
    health_check_timeout_s=120,  # 将健康检查超时时间从默认的30秒增加到120秒
    health_check_period_s=30     # 将健康检查周期从默认的10秒增加到30秒
)
class ModelInMaker:
    def __init__(self):
        """初始化Frontend模块，独立加载CosyVoiceFrontEnd"""
        self.logger = logging.getLogger(__name__)
        
        # 使用配置中的目标采样率和其他配置
        self.config = Config()
        self.speaker_cache = {}  # 存储speaker_id到特征的映射
        self.max_val = 0.8
        
        # 添加系统路径
        for path in self.config.SYSTEM_PATHS:
            if path not in sys.path:
                sys.path.append(path)
                self.logger.info(f"添加系统路径: {path}")
        
        # 创建特征存储目录(仍然需要用于其他用途)
        self.feature_dir = os.path.join(str(self.config.TASKS_DIR), "features")
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # 加载Frontend模块
        self._load_frontend()
        
    def _to_cpu(self, obj):
        """递归地将所有PyTorch张量移到CPU"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, list):
            return [self._to_cpu(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._to_cpu(item) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._to_cpu(v) for k, v in obj.items()}
        else:
            return obj
        
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
        更新文本特征（直接存储到sentence）
        """
        try:
            tts_text = sentence.trans_text
            
            # 文本正则化
            normalized_segments = self.frontend.text_normalize(tts_text, split=True)
            
            # 提取文本token
            segment_tokens = []
            segment_token_lens = []
            
            for seg in normalized_segments:
                txt, txt_len = self.frontend._extract_text_token(seg)
                segment_tokens.append(txt)
                segment_token_lens.append(txt_len)
            
            # 准备特征数据
            text_features = {
                'text': segment_tokens,
                'text_len': segment_token_lens,
                'normalized_text_segments': normalized_segments
            }
            
            # 确保所有张量都在CPU上，然后存储到sentence
            text_features = self._to_cpu(text_features)
            sentence.model_input['text_features'] = text_features

            self.logger.debug(f"成功更新文本特征: {normalized_segments}")
            return sentence
        except Exception as e:
            self.logger.error(f"更新文本特征失败: {str(e)}")
            raise

    def _modelin_sentence(self, sentence, reuse_speaker=False):
        """
        对单个句子进行模型输入处理（直接存储数据）
        """
        speaker_id = sentence.speaker_id

        # 1) Speaker处理
        if not reuse_speaker:
            if speaker_id not in self.speaker_cache:
                try:
                    # 准备音频
                    audio = sentence.audio
                    if audio is None:
                         raise ValueError("Sentence object missing 'audio' data needed for speaker embedding.")
                         
                    # 处理音频 (只在内存中处理)
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
                    
                    # 使用空文本进行跨语言特征提取
                    speaker_features = self.frontend.frontend_cross_lingual(
                        "",
                        processed_audio, # 使用内存中的张量
                        self.sample_rate
                    )
                    
                    # 确保特征在CPU上
                    speaker_features = self._to_cpu(speaker_features)
                    
                    # 直接存储特征到缓存，不使用ray.put
                    self.speaker_cache[speaker_id] = speaker_features
                    
                    # 显式删除大型音频数据，释放内存
                    del processed_audio
                    del speech_tensor
                    del audio # 删除原始音频
                    sentence.audio = None # 确保sentence对象中也没有引用
                    
                except Exception as e:
                    self.logger.error(f"模型输入音频处理失败: {str(e)}")
                    raise
                
            # 直接存储speaker特征到sentence，不使用ray.put
            sentence.model_input['speaker_features'] = self.speaker_cache[speaker_id]
            
            # 特征提取完成后删除audio数据，释放内存
            if hasattr(sentence, 'audio') and sentence.audio is not None:
                del sentence.audio
                sentence.audio = None
                self.logger.debug(f"已删除句子的原始音频数据引用，释放内存")

        # 2) 文本特征更新
        return self._update_text_features(sentence) # 返回更新后的sentence

    # 改为异步方法
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
            try:
                # 调用同步处理函数，使用 asyncio.to_thread 包装
                modelin_sentence = await asyncio.to_thread(
                    self._modelin_sentence, 
                    s, 
                    reuse_speaker
                )
                results.append(modelin_sentence)
            except Exception as e:
                 self.logger.error(f"处理句子 {i} (_modelin_sentence) 失败: {e}")
                 # 选择跳过这个句子或返回原始句子
                 results.append(s) # 返回原始句子以便下游知道有失败

            if i % batch_size == 0:
                yield results
                results = []

        if results:
            yield results

        if not reuse_speaker:
            # 清理speaker缓存
            self.speaker_cache.clear()
            self.logger.debug("modelin_maker: 已清理本地speaker缓存")