import logging
import ray
from ray import serve
import os
import sys
import torch
import uuid
import threading
import numpy as np
from config import Config

@serve.deployment(
    name="tts_token_generator"
)
class TtsTokenGenerator:
    """TTS Token生成Actor，专注于LLM模型"""
    
    def __init__(self):
        """初始化LLM模型Actor"""
        self.logger = logging.getLogger("tts_token_generator")
        self.config = Config()
        
        # 设置系统路径
        for path in self.config.SYSTEM_PATHS:
            if path not in sys.path:
                sys.path.append(path)
                self.logger.info(f"添加系统路径: {path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fp16 = torch.cuda.is_available()
        
        # 特征存储目录
        self.feature_dir = os.path.join(self.config.TASKS_DIR, "features")
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # 加载LLM模型
        self._load_llm_model()
        
    def _load_llm_model(self):
        """只加载LLM模型"""
        try:
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
                self.configs = load_hyperpyyaml(f, overrides={
                    'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')
                })
            
            # 直接导入模型类
            from models.CosyVoice.cosyvoice.cli.model import CosyVoice2Model
            
            # 创建模型实例
            self.model = CosyVoice2Model(
                self.configs['llm'],
                self.configs['flow'],
                self.configs['hift'],
                fp16=self.fp16
            )
            
            # 只加载LLM模型权重
            self.model.llm.load_state_dict(torch.load(f'{model_dir}/llm.pt', map_location=self.device), strict=True)
            self.model.llm.to(self.device).eval()
            
            # 设置FP16
            if self.fp16:
                self.model.llm.fp16 = True
                self.model.llm.half()
            
            # 添加必要的属性
            self.model.lock = threading.Lock()
            self.model.tts_speech_token_dict = {}
            self.model.llm_end_dict = {}
            
            self.logger.info("LLM模型加载完成")
        except Exception as e:
            self.logger.error(f"LLM模型加载失败: {str(e)}")
            import traceback
            self.logger.error(f"异常堆栈: {traceback.format_exc()}")
            raise
    
    def generate_tts_tokens(self, sentences):
        """生成TTS tokens并保存到文件"""
        if not sentences:
            self.logger.warning("generate_tts_tokens: 收到空的句子列表")
            return sentences
            
        self.logger.info(f"处理 {len(sentences)} 个句子生成TTS tokens")
        
        try:
            # 处理句子列表
            processed_sentences = []
            for sentence in sentences:
                try:
                    # 获取所需特征路径
                    model_input = sentence.model_input
                    text_feature_path = model_input.get('text_feature_path')
                    speaker_feature_path = model_input.get('speaker_feature_path')
                    
                    if not text_feature_path or not speaker_feature_path:
                        raise ValueError(f"缺少必要的特征路径: text_feature_path={text_feature_path}, speaker_feature_path={speaker_feature_path}")
                    
                    # 加载特征
                    text_features = torch.load(text_feature_path)
                    speaker_features = torch.load(speaker_feature_path)
                    
                    # 生成TTS token ID和文件路径
                    tts_token_id = str(uuid.uuid4())
                    tts_token_path = os.path.join(self.feature_dir, f"tts_token_{tts_token_id}.pt")
                    
                    # 处理每个文本段落，生成TTS tokens
                    segment_tokens_list = []
                    segment_uuids = []
                    
                    for i, (text, text_len) in enumerate(zip(text_features['text'], text_features['text_len'])):
                        current_seg_uuid = f"{tts_token_id}_seg_{i}"
                        
                        with self.model.lock:
                            self.model.tts_speech_token_dict[current_seg_uuid] = []
                            self.model.llm_end_dict[current_seg_uuid] = False
                        
                        # 获取必要的参数
                        prompt_text = speaker_features.get('prompt_text', torch.zeros(1, 0, dtype=torch.int32))
                        llm_prompt_speech_token = speaker_features.get('llm_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32))
                        llm_embedding = speaker_features.get('llm_embedding', torch.zeros(0, 192))
                        
                        # 调用LLM生成tokens
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
                    
                    # 保存TTS token特征到文件
                    tts_tokens = {
                        'segment_speech_tokens': segment_tokens_list,
                        'segment_uuids': segment_uuids,
                        'duration': total_duration_ms
                    }
                    torch.save(tts_tokens, tts_token_path)
                    
                    # 更新句子
                    model_input['tts_token_path'] = tts_token_path
                    sentence.duration = total_duration_ms
                    
                    self.logger.info(f"TTS token 生成完成 (ID={tts_token_id}, 时长={total_duration_ms:.2f}ms)")
                    processed_sentences.append(sentence)
                    
                    # 清理模型中的临时缓存
                    with self.model.lock:
                        for seg_uuid in segment_uuids:
                            self.model.tts_speech_token_dict.pop(seg_uuid, None)
                            self.model.llm_end_dict.pop(seg_uuid, None)
                    
                    # --- Memory cleanup for loop iteration ---
                    del text_features
                    del speaker_features
                    del tts_tokens
                    # --- End Memory cleanup ---
                    
                except Exception as e:
                    self.logger.error(f"句子处理失败: {e}")
                    # 仍然添加到结果列表，保持原始顺序
                    processed_sentences.append(sentence)
            
            self.logger.info(f"TTS tokens 生成完成，处理了 {len(processed_sentences)} 个句子")
            return processed_sentences
            
        except Exception as e:
            self.logger.error(f"批量生成TTS token失败: {e}")
            raise
        finally:
            # Ensure GPU cache is cleared after the batch processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.debug("TtsTokenGenerator: Cleared GPU cache.")