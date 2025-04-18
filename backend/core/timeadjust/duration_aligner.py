from ray import serve
from ray.serve.handle import DeploymentHandle
import logging
import asyncio
from config import Config
from core.sentence_tools import Sentence
from typing import List

# 导入新的工具函数
from utils.duration_utils import apply_speed_and_silence, align_batch

logger = logging.getLogger(__name__)

@serve.deployment(
    name="duration_aligner",
    ray_actor_options={"num_cpus": 0.25}
)
class DurationAligner:
    def __init__(self, simplifier_handle: DeploymentHandle, index_tts_handle: DeploymentHandle):
        # 从 pipeline_scheduler.py 中注入句柄
        self.simplifier = simplifier_handle.options(stream=True) # 指向 Translator (Simplifier)，启用流式处理
        self.index_tts = index_tts_handle.options(stream=True)   # 指向 MyIndexTTSDeployment，启用流式处理
        self.config = Config()
        self.sample_rate = getattr(self.config, 'TARGET_SR', 24000)  # 获取采样率，默认24kHz
        logger.warning("时长对齐器已初始化")

    async def __call__(self, sentences: List[Sentence], max_speed: float = 1.1) -> List[Sentence]:
        """执行句子时长对齐"""
        if not sentences:
            logger.warning("时长对齐：收到空句子列表")
            return sentences

        task_id = sentences[0].task_id if sentences else "unknown" 
        logger.warning(f"[{task_id}] 开始为 {len(sentences)} 个句子进行时长对齐")

        try:
            # 第一次对齐：计算初始速度和差异
            aligned_sentences = await asyncio.to_thread(align_batch, sentences)
            if not aligned_sentences:
                 logger.error(f"[{task_id}] 初始对齐失败或返回空列表")
                 return sentences

            # 查找超过最大速度的句子并输出详细信息
            fast_indices = []
            for i, s in enumerate(aligned_sentences):
                if s.speed > max_speed:
                    fast_indices.append(i)
                    logger.warning(f"[{task_id}] 句子ID {s.sentence_id}: 速度过快 ({s.speed:.2f}x > {max_speed:.2f}x), "
                               f"原始时长: {s.duration:.2f}ms, 目标时长: {s.target_duration:.2f}ms")
                else:
                    logger.warning(f"[{task_id}] 句子ID {s.sentence_id}: 速度正常 ({s.speed:.2f}x), "
                               f"原始时长: {s.duration:.2f}ms, 目标时长: {s.target_duration:.2f}ms, "
                               f"将应用速度: {s.speed:.2f}x, 静音长度: {s.silence_duration:.2f}ms")

            # 处理过快的句子
            if fast_indices:
                # 提取过快的句子
                fast_sentences = [aligned_sentences[idx] for idx in fast_indices]
                logger.warning(f"[{task_id}] 发现 {len(fast_indices)} 个语速过快的句子 (>{max_speed}x)，尝试简化并重新生成")
                
                # 简化和重新生成音频
                result_sentences = await self._process_fast_sentences(task_id, aligned_sentences, fast_sentences, fast_indices, max_speed)
                return result_sentences
            else:
                # 没有过快的句子，直接应用变速和静音
                logger.warning(f"[{task_id}] 没有语速过快的句子，直接应用速度和静音调整")
                await apply_speed_and_silence(aligned_sentences, self.sample_rate)
                
                # 显示处理结果
                for s in aligned_sentences:
                    if hasattr(s, 'speed') and s.speed != 1.0:
                        logger.warning(f"[{task_id}] 句子ID {s.sentence_id} 速度调整成功: {s.speed:.2f}x")
                    if hasattr(s, 'silence_duration') and s.silence_duration > 0:
                        logger.warning(f"[{task_id}] 句子ID {s.sentence_id} 添加静音成功: {s.silence_duration:.2f}ms")
                
                logger.warning(f"[{task_id}] 时长对齐和音频调整完成")
                return aligned_sentences

        except Exception as e:
            logger.exception(f"[{task_id}] 时长对齐过程中发生错误: {e}")
            try:
                if 'aligned_sentences' in locals() and aligned_sentences:
                    await apply_speed_and_silence(aligned_sentences, self.sample_rate)
                    return aligned_sentences
            except Exception as e_apply:
                logger.error(f"[{task_id}] 应用速度和静音调整失败: {e_apply}")
            
            return sentences # 返回原始句子作为后备方案

    async def _process_fast_sentences(self, task_id, aligned_sentences, fast_sentences, fast_indices, max_speed):
        """处理语速过快的句子"""
        try:
            # 尝试简化文本
            simplified_results = await self._simplify_sentences(task_id, fast_sentences, max_speed)
            if not simplified_results:
                logger.warning(f"[{task_id}] 未能获取简化结果，应用速度和静音调整到初始对齐结果")
                await apply_speed_and_silence(aligned_sentences, self.sample_rate)
                return aligned_sentences
            
            # 重新生成音频
            all_refined_sentences = await self._regenerate_audio(task_id, simplified_results)
            if not all_refined_sentences or len(all_refined_sentences) != len(fast_indices):
                logger.warning(f"[{task_id}] TTS重新生成失败或返回数量不匹配 ({len(all_refined_sentences) if all_refined_sentences else 0} vs {len(fast_indices)})")
                await apply_speed_and_silence(aligned_sentences, self.sample_rate)
                return aligned_sentences
            
            # 将简化后的句子替换回原始列表
            result_sentences = aligned_sentences.copy()
            successful_refinement = False
            
            for i, orig_idx in enumerate(fast_indices):
                if i < len(all_refined_sentences) and all_refined_sentences[i].generated_audio is not None and all_refined_sentences[i].duration > 0:
                    result_sentences[orig_idx] = all_refined_sentences[i]
                    logger.warning(f"[{task_id}] 句子ID {all_refined_sentences[i].sentence_id} 简化并重新生成成功，"
                               f"原始时长: {aligned_sentences[orig_idx].duration:.2f}ms → 新时长: {all_refined_sentences[i].duration:.2f}ms")
                    successful_refinement = True
                else:
                    logger.warning(f"[{task_id}] 句子索引 {i} (原始索引 {orig_idx}) 缺少有效音频/时长或丢失，保留原始对齐句子")
            
            if not successful_refinement:
                logger.warning(f"[{task_id}] 没有句子成功完成简化和重新生成，应用速度和静音调整到初始对齐结果")
                await apply_speed_and_silence(aligned_sentences, self.sample_rate)
                return aligned_sentences
            
            # 重新对齐和应用变速静音
            logger.warning(f"[{task_id}] 简化和重新生成完成，执行最终对齐...")
            final_aligned_sentences = await asyncio.to_thread(align_batch, result_sentences)
            
            logger.warning(f"[{task_id}] 最终对齐后应用速度和静音调整")
            await apply_speed_and_silence(final_aligned_sentences, self.sample_rate)
            
            # 显示处理结果
            for s in final_aligned_sentences:
                if hasattr(s, 'speed') and s.speed != 1.0:
                    logger.warning(f"[{task_id}] 句子ID {s.sentence_id} 最终速度调整: {s.speed:.2f}x")
                if hasattr(s, 'silence_duration') and s.silence_duration > 0:
                    logger.warning(f"[{task_id}] 句子ID {s.sentence_id} 最终静音添加: {s.silence_duration:.2f}ms")
            
            logger.warning(f"[{task_id}] 最终时长对齐和音频调整完成")
            return final_aligned_sentences
            
        except Exception as e:
            logger.exception(f"[{task_id}] 处理快速句子时出错: {e}")
            await apply_speed_and_silence(aligned_sentences, self.sample_rate)
            return aligned_sentences

    async def _simplify_sentences(self, task_id, fast_sentences, max_speed):
        """简化句子文本"""
        simplified_results = []
        try:
            logger.warning(f"[{task_id}] 开始简化 {len(fast_sentences)} 个句子...")
            async for simplified_batch in self.simplifier.simplify_sentences.remote(fast_sentences, target_speed=max_speed):
                if simplified_batch:
                    simplified_results.extend(simplified_batch)
            logger.warning(f"[{task_id}] 获得 {len(simplified_results)} 个简化后的句子")
            return simplified_results
        except Exception as e:
            logger.error(f"[{task_id}] 简化调用期间出错: {e}", exc_info=True)
            return []

    async def _regenerate_audio(self, task_id, simplified_results):
        """重新生成音频"""
        all_refined_sentences = []
        try:
            logger.warning(f"[{task_id}] 使用流式TTS重新生成 {len(simplified_results)} 个句子的音频...")
            async for tts_batch in self.index_tts.generate_audio_stream.remote(simplified_results):
                if tts_batch:
                    all_refined_sentences.extend(tts_batch)
            logger.warning(f"[{task_id}] 成功重新生成 {len(all_refined_sentences)} 个句子的音频")
            return all_refined_sentences
        except Exception as e:
            logger.error(f"[{task_id}] TTS流处理期间出错: {e}", exc_info=True)
            return []