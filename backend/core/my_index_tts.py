# core/my_index_tts.py
import os
import sys
import logging
import asyncio
import time
import io
from typing import List, AsyncGenerator, Optional
import gc
import torch
import numpy as np
import torchaudio
from ray import serve
from config import Config
from core.sentence_tools import Sentence

# 路径配置
_MY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MY_DIR)
INDEXTTS_DIR = os.path.join(_PROJECT_ROOT, 'models', 'IndexTTS')
if INDEXTTS_DIR not in sys.path:
    sys.path.insert(0, INDEXTTS_DIR)
CHECKPOINTS_DIR = os.path.join(_PROJECT_ROOT, 'models', 'IndexTTS', 'checkpoints')
CFG_PATH = os.path.join(CHECKPOINTS_DIR, 'config.yaml')
MODEL_DIR = CHECKPOINTS_DIR

# 导入自定义的 MyIndexTTS
from utils.index_tts_utils import MyIndexTTS

logger = logging.getLogger("ray.serve")

@serve.deployment(
    name="my_index_tts",
    # Configure resources based on IndexTTS needs (adjust GPU if required)
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.4}, # Example: Allocate more GPU if needed
    max_ongoing_requests=2, # Allow some concurrency
)
class MyIndexTTSDeployment:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 使用 utils.index_tts_utils 中定义的 MyIndexTTS
        self.tts_model = MyIndexTTS(cfg_path=CFG_PATH, model_dir=MODEL_DIR, is_fp16=True, device=self.device)
        self.sampling_rate = 24000  # 采样率固定为24kHz
        self.yield_batch_size = getattr(self.config, 'TTS_BATCH_SIZE', 8)
        self._tts_lock = asyncio.Lock()  # 添加类级别的锁

    def _clean_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_single_sentence_audio(self, sentence: Sentence) -> Optional[np.ndarray]:
        try:
            text = sentence.trans_text
            # 简化日志，只打印清理后的文本
            logger.info(f"TTS处理文本 (ID:{sentence.sentence_id}): {text}")
            
            # 检查音频文件路径
            if sentence.audio and os.path.exists(sentence.audio):
                audio_prompt = sentence.audio
                # 简化音频路径输出
                logger.debug(f"使用音频参考: {os.path.basename(audio_prompt)}")
            else:
                logger.warning(f"句子ID {sentence.sentence_id} 缺少音频参考文件，跳过TTS生成。")
                return None

            # 调用 infer 方法获取音频张量
            generated_audio_tensor = self.tts_model.infer(
                audio_prompt=audio_prompt,
                text=text
            )

            # 检查 infer 是否成功返回音频
            if generated_audio_tensor is not None and generated_audio_tensor.numel() > 0:
                # 将 Tensor 转换为 NumPy 数组
                generated_audio_np = generated_audio_tensor.numpy()
                # 只打印基本长度信息
                logger.debug(f"句子ID {sentence.sentence_id} TTS成功: 音频长度 {len(generated_audio_np)/self.sampling_rate:.2f}秒")
                return generated_audio_np
            else:
                logger.warning(f"句子ID {sentence.sentence_id}: TTS处理失败或返回空音频")
                return None
                
        except Exception as e:
            logger.error(f"生成句子ID {sentence.sentence_id} 音频时出错: {str(e)}")
            return None
        finally:
            self._clean_memory()

    async def generate_audio_stream(self, sentences: List[Sentence]) -> AsyncGenerator[List[Sentence], None]:
        """
        Receives a list of translated sentences, generates audio for each,
        updates their duration and generated_audio attributes, and yields them in batches.

        Args:
            sentences: List of Sentence objects with 'trans_text' and 'audio' populated.

        Yields:
            List[Sentence]: Batches of processed Sentence objects with 'generated_audio'
                            and 'duration' updated.
        """
        if not sentences:
            logger.warning("生成音频流收到空句子列表")
            return

        logger.info(f"开始处理 {len(sentences)} 个句子的TTS生成")
        tts_batch: List[Sentence] = []

        for i, sentence in enumerate(sentences):
            try:
                async with self._tts_lock:  # 锁保护整个TTS处理和清理过程
                    generated_wav_np = await asyncio.to_thread(
                        self._generate_single_sentence_audio, sentence
                    )
                    # 处理完后在锁内清理
                    self._clean_memory()

                if generated_wav_np is not None and len(generated_wav_np) > 0:
                    if generated_wav_np.ndim > 1:
                        generated_wav_np = generated_wav_np.flatten()
                    sentence.generated_audio = generated_wav_np
                    duration_ms = (len(generated_wav_np) / self.sampling_rate) * 1000
                    sentence.duration = duration_ms
                    # 简化日志，只打印ID和最终时长
                    logger.debug(f"句子ID {sentence.sentence_id} 处理完成: 时长 {duration_ms/1000:.2f}秒")
                else:
                    sentence.generated_audio = None
                    sentence.duration = 0.0
                    logger.warning(f"句子ID {sentence.sentence_id} 音频生成失败，时长设为0")

                tts_batch.append(sentence)

                if len(tts_batch) >= self.yield_batch_size:
                    yield tts_batch
                    logger.info(f"已生成一批 {len(tts_batch)} 个句子的音频")
                    tts_batch = []

            except Exception as e:
                logger.error(f"处理句子ID {sentence.sentence_id} 时出错: {str(e)}")
                sentence.generated_audio = None
                sentence.duration = 0.0
                tts_batch.append(sentence)
                if len(tts_batch) >= self.yield_batch_size:
                    yield tts_batch
                    logger.info(f"已生成一批 {len(tts_batch)} 个句子的音频 (含错误)")
                    tts_batch = []

            finally:
                # 移除不必要的处理时间日志，只保留在debug级别
                if (i+1) % 5 == 0:  # 每5个句子记录一次进度
                    logger.info(f"TTS进度: {i+1}/{len(sentences)}")

        if tts_batch:
            yield tts_batch
            logger.info(f"已生成最后一批 {len(tts_batch)} 个句子的音频")

        logger.info(f"所有 {len(sentences)} 个句子的TTS处理完成")