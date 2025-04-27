# core/my_index_tts.py
import os
import sys
import logging
import asyncio
import gc
from typing import List, AsyncGenerator, Optional

import torch
import numpy as np
from ray import serve

# 全局 logger
logger = logging.getLogger(__name__)

@serve.deployment(
    name="my_index_tts",
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.4},
    max_ongoing_requests=2,
)
class MyIndexTTSDeployment:
    """
    Ray Serve 部署，提供流式 TTS 服务。
    """
    def __init__(self, config=None):
        # 确保在新进程中可以正确导入indextts模块
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        indextts_dir = os.path.join(backend_dir, 'models', 'IndexTTS')
        if indextts_dir not in sys.path:
            sys.path.insert(0, indextts_dir)
            logger.info(f"添加indextts模块路径: {indextts_dir}")
        
        # 记录当前sys.path用于调试
        logger.info(f"当前sys.path: {sys.path}")
        
        # 初始化配置和设备
        from config import Config
        self.config = config or Config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 定义模型和配置路径
        checkpoints_dir = os.path.join(backend_dir, 'models', 'IndexTTS', 'checkpoints')
        cfg_path = os.path.join(checkpoints_dir, 'config.yaml')
        model_dir = checkpoints_dir
        
        try:
            # 在确保路径设置后导入模型
            from models.IndexTTS.indextts.infer import IndexTTS
            logger.info(f"导入IndexTTS成功，加载模型路径：{cfg_path}")
            self.tts_model = IndexTTS(
                cfg_path=cfg_path,
                model_dir=model_dir,
                is_fp16=True,
                device=self.device
            )
            logger.info("IndexTTS模型加载成功")
        except Exception as e:
            logger.exception(f"IndexTTS初始化失败: {e}")
            raise
            
        # 配置样本率和批次大小
        self.sampling_rate = self.config.TARGET_SR
        self.batch_size = self.config.TTS_BATCH_SIZE
        self._lock = asyncio.Lock()

    def _clean_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def generate_audio_stream(
        self, sentences: List
    ) -> AsyncGenerator[List, None]:
        """
        接收翻译后的 Sentence 列表，生成音频并按批次返回。
        """
        if not sentences:
            logger.warning("TTS 收到空句子列表，跳过生成。")
            return

        # 批次收集并生成
        batch = []
        for sentence in sentences:
            try:
                # 在后台线程执行 infer，使用锁保护 GPU
                async with self._lock:
                    res = await asyncio.to_thread(
                        self.tts_model.infer,
                        sentence.audio,
                        sentence.trans_text,
                        None,
                        False
                    )
            except Exception as e:
                logger.error(f"TTS 错误：句子 {sentence.sentence_id}，{e}")
                res = None
            # 处理返回结果
            if res is None:
                sentence.generated_audio = None
                sentence.duration = 0.0
            else:
                sr, wav_np = res
                # 将 int16 波形值归一到 [-1,1]
                wav_flat = wav_np.flatten().astype(np.float32) / 32767.0
                sentence.generated_audio = wav_flat
                sentence.duration = len(wav_flat) / sr * 1000
            batch.append(sentence)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        # 输出剩余批次
        if batch:
            yield batch
        # 批次处理完毕后统一清理一次内存
        self._clean_memory()