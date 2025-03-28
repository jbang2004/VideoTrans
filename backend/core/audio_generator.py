import logging
import numpy as np
import ray
from ray import serve
import torch
import os
import sys
import uuid
import threading
from config import Config

@serve.deployment(
    name="audio_generator"
)
class AudioGenerator:
    """音频生成Actor，专注于Flow和HiFT模型"""
    
    def __init__(self):
        """初始化Flow和HiFT模型Actor"""
        self.logger = logging.getLogger("audio_generator")
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
        
        # 加载Flow和HiFT模型
        self._load_flow_hift_models()
        
    def _load_flow_hift_models(self):
        """只加载Flow和HiFT模型"""
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
            
            # 只加载Flow模型权重
            self.model.flow.load_state_dict(torch.load(f'{model_dir}/flow.pt', map_location=self.device), strict=True)
            self.model.flow.to(self.device).eval()
            
            # 加载HiFT模型权重
            hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f'{model_dir}/hift.pt', map_location=self.device).items()}
            self.model.hift.load_state_dict(hift_state_dict, strict=True)
            self.model.hift.to(self.device).eval()
            
            # 可选：加载JIT编译的Flow模型
            if self.fp16 and torch.cuda.is_available():
                try:
                    flow_encoder = torch.jit.load(f'{model_dir}/flow.encoder.fp16.zip', map_location=self.device)
                    self.model.flow.encoder = flow_encoder
                except Exception as jit_e:
                    self.logger.warning(f"JIT模型加载失败: {str(jit_e)}")
            
            # 设置FP16
            if self.fp16:
                self.model.flow.fp16 = True
                self.model.flow.half()
            
            # 初始化必要的缓存
            self.this_uuid = str(uuid.uuid4())
            self.model.lock = threading.Lock()
            self.model.hift_cache_dict = {self.this_uuid: None}
            
            if hasattr(self.model, 'mel_overlap_dict'):
                self.model.mel_overlap_dict = {self.this_uuid: torch.zeros(1, 80, 0)}
            
            if hasattr(self.model, 'flow_cache_dict'):
                self.model.flow_cache_dict = {self.this_uuid: torch.zeros(1, 80, 0, 2)}
            
            self.sample_rate = self.configs['sample_rate']
            self.logger.info("Flow和HiFT模型加载完成")
        except Exception as e:
            self.logger.error(f"Flow和HiFT模型加载失败: {str(e)}")
            self.logger.error(f"异常详情: {str(e)}")
            self.logger.error(f"异常类型: {type(e)}")
            import traceback
            self.logger.error(f"异常堆栈: {traceback.format_exc()}")
            raise
    
    def generate_audio(self, sentences):
        """生成音频"""
        if not sentences:
            self.logger.warning("generate_audio: 收到空的句子列表")
            return sentences
            
        self.logger.info(f"开始处理 {len(sentences)} 个句子的音频生成")
        
        try:
            # 处理句子列表
            sentences_with_audio = []
            for sentence in sentences:
                try:
                    # 获取所需参数
                    model_input = sentence.model_input
                    tts_token_path = model_input.get('tts_token_path')
                    speaker_feature_path = model_input.get('speaker_feature_path')
                    
                    if not tts_token_path or not speaker_feature_path:
                        self.logger.info(f"缺少必要的参数，仅生成空波形 (TTS Token Path: {tts_token_path})")
                        sentence.generated_audio = np.zeros(0, dtype=np.float32)
                        sentences_with_audio.append(sentence)
                        continue
                    
                    # 加载特征
                    tts_tokens = torch.load(tts_token_path)
                    speaker_features = torch.load(speaker_feature_path)
                    
                    # 获取语速
                    speed = sentence.speed if hasattr(sentence, 'speed') and sentence.speed else 1.0
                    
                    # 准备参数
                    prompt_token = speaker_features.get('flow_prompt_speech_token', torch.zeros(1, 0, dtype=torch.int32))
                    prompt_feat = speaker_features.get('prompt_speech_feat', torch.zeros(1, 0, 80))
                    embedding = speaker_features.get('flow_embedding', torch.zeros(0))
                    
                    segment_audio_list = []
                    
                    for i, tokens in enumerate(tts_tokens['segment_speech_tokens']):
                        if not tokens:
                            segment_audio_list.append(np.zeros(0, dtype=np.float32))
                            continue
                        
                        # 准备token tensor
                        token_tensor = torch.tensor(tokens).unsqueeze(dim=0).to(self.device)
                        
                        # 使用model.token2wav方法生成音频
                        try:
                            # 准备通用的参数字典，始终包含token_offset
                            kwargs = {
                                'token': token_tensor,
                                'token_offset': 0,  # 总是包含token_offset参数
                                'prompt_token': prompt_token.to(self.device),
                                'prompt_feat': prompt_feat.to(self.device),
                                'embedding': embedding.to(self.device),
                                'uuid': self.this_uuid,
                                'finalize': True,
                                'speed': speed
                            }
                            
                            # 直接使用kwargs调用token2wav
                            segment_output = self.model.token2wav(**kwargs)
                        except Exception as model_error:
                            self.logger.error(f"音频生成错误: {model_error}")
                            # 生成一个空音频作为回退方案
                            segment_output = torch.zeros(1, 0)
                        
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
                    
                    # 添加首句静音
                    if hasattr(sentence, 'is_first') and sentence.is_first and hasattr(sentence, 'start') and sentence.start > 0:
                        silence_samples = int(sentence.start * self.sample_rate / 1000)
                        final_audio = np.concatenate([np.zeros(silence_samples, dtype=np.float32), final_audio])
                    
                    # 添加尾部静音
                    if hasattr(sentence, 'silence_duration') and sentence.silence_duration > 0:
                        silence_samples = int(sentence.silence_duration * self.sample_rate / 1000)
                        final_audio = np.concatenate([final_audio, np.zeros(silence_samples, dtype=np.float32)])
                    
                    # 更新句子
                    sentence.generated_audio = final_audio
                    self.logger.info(f"音频生成完成 (长度: {len(final_audio)}样本)")
                    sentences_with_audio.append(sentence)
                    
                except Exception as e:
                    self.logger.error(f"句子音频生成失败: {e}")
                    # 设置空音频
                    sentence.generated_audio = np.zeros(0, dtype=np.float32)
                    sentences_with_audio.append(sentence)
            
            self.logger.info(f"音频生成完成，处理了 {len(sentences_with_audio)} 个句子")
            return sentences_with_audio
            
        except Exception as e:
            self.logger.error(f"批量音频生成失败: {e}")
            raise
