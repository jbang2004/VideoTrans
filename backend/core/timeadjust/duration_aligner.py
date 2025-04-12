from ray import serve
from ray.serve.handle import DeploymentHandle
import logging
import asyncio
from config import Config
from typing import List

logger = logging.getLogger("duration_aligner")

@serve.deployment(
    name="duration_aligner",
    ray_actor_options={"num_cpus": 0.25}
)
class DurationAligner:
    def __init__(self, simplifier_handle: DeploymentHandle, model_in_handle: DeploymentHandle, tts_token_gen_handle: DeploymentHandle):
        self.simplifier = simplifier_handle
        self.model_in = model_in_handle
        self.tts_token_gen = tts_token_gen_handle
        self.config = Config()
    async def __call__(self, sentences: List, max_speed: float = 1.1) -> List:
        if not sentences:
            logger.warning("align_durations: 收到空的句子列表")
            return sentences
        
        logger.info(f"开始处理 {len(sentences)} 个句子的时长对齐")
        try:
            # 直接使用asyncio.to_thread包装同步函数
            aligned_sentences = await asyncio.to_thread(_align_batch, sentences)
            fast_indices = [i for i, sentence in enumerate(aligned_sentences) if sentence.speed > max_speed]
            if fast_indices:
                fast_sentences = [aligned_sentences[idx] for idx in fast_indices]
                async for simplified_batch in self.simplifier.options(stream=True).simplify_sentences.remote(fast_sentences, target_speed=max_speed):
                    async for modelin_batch in self.model_in.options(stream=True).modelin_maker.remote(simplified_batch, batch_size=self.config.MODELIN_BATCH_SIZE):
                        refined_sentences = await self.tts_token_gen.generate_tts_tokens.remote(modelin_batch)
                if refined_sentences:
                    result_sentences = aligned_sentences.copy()
                    for i, orig_idx in enumerate(fast_indices):
                        if i < len(refined_sentences):
                            result_sentences[orig_idx] = refined_sentences[i]
                    logger.info("精简完成，进行最终对齐...")
                    # 直接使用asyncio.to_thread包装同步函数
                    return await asyncio.to_thread(_align_batch, result_sentences)
                else:
                    logger.warning("精简过程未能生成有效句子，保持原句子")
                    return aligned_sentences
            else:
                logger.info(f"时长对齐处理完成，共处理 {len(aligned_sentences)} 个句子")
                return aligned_sentences
        except Exception as e:
            logger.error(f"时长对齐处理失败: {e}")
            raise

def _align_batch(sentences):
    # 原有的 _align_batch 实现保持不变
    if not sentences:
        return sentences

    aligned_sentences = []
    for s in sentences:
        aligned_s = s
        aligned_s.diff = aligned_s.duration - aligned_s.target_duration
        aligned_sentences.append(aligned_s)

    total_diff_to_adjust = sum(s.diff for s in aligned_sentences)
    positive_diff_sum = sum(x.diff for x in aligned_sentences if x.diff > 0)
    negative_diff_sum_abs = sum(abs(x.diff) for x in aligned_sentences if x.diff < 0)
    current_time = aligned_sentences[0].start

    for s in aligned_sentences:
        s.adjusted_start = current_time
        diff = s.diff
        s.speed = 1.0
        s.silence_duration = 0.0
        s.adjusted_duration = s.duration

        if total_diff_to_adjust != 0:
            if total_diff_to_adjust > 0 and diff > 0:
                if positive_diff_sum > 0:
                    proportion = diff / positive_diff_sum
                    adjustment = total_diff_to_adjust * proportion
                    s.adjusted_duration = s.duration - adjustment
                    s.speed = s.duration / max(s.adjusted_duration, 0.001)
            elif total_diff_to_adjust < 0 and diff < 0:
                if negative_diff_sum_abs > 0:
                    proportion = abs(diff) / negative_diff_sum_abs
                    total_needed = abs(total_diff_to_adjust) * proportion
                    max_slowdown = s.duration * 0.07
                    slowdown = min(total_needed, max_slowdown)
                    s.adjusted_duration = s.duration + slowdown
                    s.speed = s.duration / max(s.adjusted_duration, 0.001)
                    s.silence_duration = total_needed - slowdown
                    if s.silence_duration > 0:
                        s.adjusted_duration += s.silence_duration

        s.diff = s.duration - s.adjusted_duration
        current_time += s.adjusted_duration
    
    return aligned_sentences