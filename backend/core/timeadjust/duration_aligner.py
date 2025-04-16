from ray import serve
from ray.serve.handle import DeploymentHandle
import logging
import asyncio
from config import Config
from core.sentence_tools import Sentence
from typing import List

logger = logging.getLogger("duration_aligner")

@serve.deployment(
    name="duration_aligner",
    ray_actor_options={"num_cpus": 0.25}
)
class DurationAligner:
    def __init__(self, simplifier_handle: DeploymentHandle, index_tts_handle: DeploymentHandle):
        # 从 pipeline_scheduler.py 中注入句柄
        self.simplifier = simplifier_handle # 指向 Translator (Simplifier)
        self.index_tts = index_tts_handle   # 指向 MyIndexTTSDeployment
        self.config = Config()
        logger.info("DurationAligner initialized.")

    async def __call__(self, sentences: List[Sentence], max_speed: float = 1.1) -> List[Sentence]:
        if not sentences:
            logger.warning("align_durations: Received an empty sentence list.")
            return sentences

        task_id = sentences[0].task_id if sentences else "unknown" # 获取 task_id 用于日志
        logger.info(f"[{task_id}] Starting duration alignment for {len(sentences)} sentences.")

        try:
            # 第一次对齐：计算初始速度和差异
            aligned_sentences = await asyncio.to_thread(_align_batch, sentences)
            if not aligned_sentences: # 检查对齐结果
                 logger.error(f"[{task_id}] Initial alignment (_align_batch) failed or returned empty list.")
                 return sentences # 返回原始句子或空列表

            # 找出语速过快的句子
            fast_indices = [i for i, sentence in enumerate(aligned_sentences) if sentence.speed > max_speed]

            if fast_indices:
                logger.info(f"[{task_id}] Found {len(fast_indices)} sentences exceeding max_speed ({max_speed}). Attempting simplification and regeneration.")
                fast_sentences = [aligned_sentences[idx] for idx in fast_indices]

                # 1. 调用 simplifier 获取简化后的句子
                simplified_results = []
                try:
                    # simplifier.simplify_sentences 也是一个异步生成器
                    # 我们需要收集所有简化后的句子
                    async for simplified_batch in self.simplifier.simplify_sentences.remote(fast_sentences, target_speed=max_speed):
                         if simplified_batch:
                              simplified_results.extend(simplified_batch)
                    # 注意：这里假设 simplify_sentences 在所有批次 yield 完毕后会正常结束循环
                except Exception as e_simplify:
                     logger.error(f"[{task_id}] Error during simplification call: {e_simplify}", exc_info=True)
                     # 简化失败，直接返回第一次对齐结果
                     return aligned_sentences

                if not simplified_results:
                     logger.warning(f"[{task_id}] Simplification did not return results for fast sentences. Skipping TTS regeneration.")
                     return aligned_sentences # 返回第一次对齐的结果

                # 确保简化后的句子数量与期望一致
                if len(simplified_results) != len(fast_sentences):
                    logger.warning(f"[{task_id}] Simplification returned {len(simplified_results)} sentences, expected {len(fast_sentences)}. Using original aligned sentences.")
                    return aligned_sentences


                logger.info(f"[{task_id}] Simplification returned {len(simplified_results)} sentences. Regenerating audio using streaming TTS.")

                # 2. --- 修改点: 使用 async for 循环处理 generate_audio_stream ---
                all_refined_sentences = [] # 用于收集所有TTS处理后的句子
                try:
                    async for tts_batch in self.index_tts.generate_audio_stream.remote(simplified_results):
                        if tts_batch:
                             all_refined_sentences.extend(tts_batch)
                except Exception as e_tts:
                     logger.error(f"[{task_id}] Error during TTS stream processing: {e_tts}", exc_info=True)
                     # TTS 流处理出错，返回第一次对齐结果
                     return aligned_sentences
                # -----------------------------------------------------------------

                # 3. 检查收集到的结果并进行替换
                if all_refined_sentences and len(all_refined_sentences) == len(fast_indices):
                    # 将精炼（重新生成音频）后的句子替换回原始列表
                    result_sentences = aligned_sentences.copy() # 创建副本以修改
                    successful_refinement = False
                    for i, orig_idx in enumerate(fast_indices):
                        # 检查精炼后的句子是否有有效的音频和时长
                        # 注意：索引 i 对应 all_refined_sentences 中的顺序
                        if i < len(all_refined_sentences) and \
                           all_refined_sentences[i].generated_audio is not None and \
                           all_refined_sentences[i].duration > 0:
                            result_sentences[orig_idx] = all_refined_sentences[i]
                            successful_refinement = True
                        else:
                            logger.warning(f"[{task_id}] Refined sentence at index {i} (original index {orig_idx}) lacks valid audio/duration or is missing. Keeping original aligned sentence.")

                    if not successful_refinement:
                         logger.warning(f"[{task_id}] None of the refined sentences had valid audio. Returning first alignment.")
                         return aligned_sentences

                    logger.info(f"[{task_id}] Refinement complete. Performing final alignment...")
                    # 对包含精炼句子的列表进行最终对齐
                    final_aligned_sentences = await asyncio.to_thread(_align_batch, result_sentences)
                    logger.info(f"[{task_id}] Final duration alignment completed after refinement.")
                    return final_aligned_sentences
                else:
                    # TTS 重新生成失败或返回数量不匹配
                    logger.warning(f"[{task_id}] TTS regeneration failed or returned incorrect number of sentences ({len(all_refined_sentences)} vs {len(fast_indices)} expected). Returning first alignment results.")
                    return aligned_sentences # 返回第一次对齐的结果

            else:
                # 没有过快的句子，直接返回第一次对齐的结果
                logger.info(f"[{task_id}] No sentences exceeded max_speed. Duration alignment complete.")
                return aligned_sentences

        except Exception as e:
            logger.exception(f"[{task_id}] Error during duration alignment: {e}")
            # 发生错误时，返回未经修改（或第一次对齐）的句子列表，避免中断流程
            # 检查 aligned_sentences 是否已定义且有效
            if 'aligned_sentences' in locals() and aligned_sentences:
                return aligned_sentences
            else:
                return sentences # Fallback to original sentences

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