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

    def _clean_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate_single_sentence_audio(self, sentence: Sentence) -> Optional[np.ndarray]:
        try:
            text = sentence.trans_text
            logger.info(f"Origin text: {text}")
            
            # 检查音频文件路径
            if sentence.audio and os.path.exists(sentence.audio):
                audio_prompt = sentence.audio
                logger.info(f"使用保存的音频文件: {audio_prompt}")
            else:
                logger.warning(f"Sentence {sentence.sentence_id} missing audio prompt file, skipping TTS.")
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
                logger.debug(f"Sentence {sentence.sentence_id}: TTS successful, audio length {len(generated_audio_np)} samples.")
                return generated_audio_np
            else:
                logger.warning(f"Sentence {sentence.sentence_id}: TTS infer failed or returned empty audio.")
                return None
                
        except Exception as e:
            logger.error(f"Error generating audio for sentence {sentence.sentence_id}: {e}", exc_info=True)
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
            logger.warning("generate_audio_stream received an empty sentence list.")
            return

        logger.info(f"Received {len(sentences)} sentences for TTS processing.")
        tts_batch: List[Sentence] = []
        # start_time = time.time()  # 未用到可移除

        for i, sentence in enumerate(sentences):
            loop_start_time = time.time()
            try:
                # 使用 asyncio.to_thread 调用同步函数
                generated_wav_np = await asyncio.to_thread(
                    self._generate_single_sentence_audio, sentence
                )

                if generated_wav_np is not None and len(generated_wav_np) > 0:
                    if generated_wav_np.ndim > 1:
                        generated_wav_np = generated_wav_np.flatten()
                    sentence.generated_audio = generated_wav_np
                    duration_ms = (len(generated_wav_np) / self.sampling_rate) * 1000
                    sentence.duration = duration_ms
                    logger.debug(f"Sentence {sentence.sentence_id} processed. Duration: {duration_ms:.2f} ms")
                else:
                    sentence.generated_audio = None
                    sentence.duration = 0.0
                    logger.warning(f"Failed to generate audio for sentence {sentence.sentence_id}. Duration set to 0.")

                tts_batch.append(sentence)

                if len(tts_batch) >= self.yield_batch_size:
                    yield tts_batch
                    logger.info(f"Yielded batch of {len(tts_batch)} sentences.")
                    tts_batch = []

            except Exception as e:
                logger.error(f"Error processing sentence {sentence.sentence_id} in main loop: {e}", exc_info=True)
                sentence.generated_audio = None
                sentence.duration = 0.0
                tts_batch.append(sentence)
                if len(tts_batch) >= self.yield_batch_size:
                    yield tts_batch
                    logger.info(f"Yielded batch of {len(tts_batch)} sentences (containing error).")
                    tts_batch = []

            finally:
                self._clean_memory()
                loop_end_time = time.time()
                logger.debug(f"Sentence {i+1}/{len(sentences)} processing took {loop_end_time - loop_start_time:.2f}s")

        if tts_batch:
            yield tts_batch
            logger.info(f"Yielded final batch of {len(tts_batch)} sentences.")

        # end_time = time.time()  # 未用到可移除
        logger.info(f"Finished TTS processing for {len(sentences)} sentences.")