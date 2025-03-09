import logging
import ray
import torch
from core.tts_token_gener import generate_tts_tokens

class DurationAligner:
    def __init__(self, model_in_actor=None, simplifier=None, cosyvoice_actor=None, max_speed=1.1):
        """
        model_in_actor：生成模型Actor接口，用于更新文本特征  
        simplifier：简化处理接口（TranslatorActor）  
        cosyvoice_actor：CosyVoice模型Actor接口，用于生成TTS token
        max_speed：语速阈值，超过该速率的句子需要进行简化
        """
        self.model_in_actor = model_in_actor
        self.simplifier = simplifier
        self.cosyvoice_actor = cosyvoice_actor  # 替代tts_token_gener
        self.max_speed = max_speed
        self.logger = logging.getLogger(__name__)

    async def align_durations(self, sentences):
        """
        对整批句子进行时长对齐，并检查是否需要精简（语速过快）。
        
        Args:
            sentences: 要处理的句子列表
        """
        if not sentences:
            return

        # 第一次对齐
        self._align_batch(sentences)

        # 查找语速过快的句子（speed > max_speed）的索引
        fast_indices = [i for i, sentence in enumerate(sentences) if sentence.speed > self.max_speed]
        if fast_indices:
            self.logger.info(f"{len(fast_indices)} 个句子语速过快, 正在精简...")
            
            # 调用_retry_sentences_batch处理这些句子，传入原始列表和需要精简的索引
            success = await self._retry_sentences_batch(sentences, fast_indices)
            
            if success:
                # 若精简文本成功，再次对齐
                self.logger.info("精简文本成功，再次对齐...")
                self._align_batch(sentences)
            else:
                self.logger.warning("精简过程失败, 保持原结果")

    def _align_batch(self, sentences):
        """
        同批次句子进行时长对齐。
        """
        if not sentences:
            return

        # 计算每句需要调整的时间差
        for s in sentences:
            s.diff = s.duration - s.target_duration

        total_diff_to_adjust = sum(s.diff for s in sentences)
        positive_diff_sum = sum(x.diff for x in sentences if x.diff > 0)
        negative_diff_sum_abs = sum(abs(x.diff) for x in sentences if x.diff < 0)
        current_time = sentences[0].start

        for s in sentences:
            s.adjusted_start = current_time
            diff = s.diff
            # 确保初始speed不为0
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

            self.logger.info(
                f"对齐后: {s.trans_text}, duration: {s.duration}, "
                f"target_duration: {s.target_duration}, diff: {s.diff}, "
                f"speed: {s.speed}, silence_duration: {s.silence_duration}"
            )

    async def _retry_sentences_batch(self, all_sentences, fast_indices):
        """
        对语速过快的句子执行精简 + 更新 TTS token，并直接更新原始句子列表。
        
        Args:
            all_sentences: 所有句子的列表
            fast_indices: 需要精简的句子索引列表
            
        Returns:
            bool: 是否成功精简
        """
        try:
            # 提取需要精简的句子
            fast_sentences = [all_sentences[idx] for idx in fast_indices]
            self.logger.debug(f"处理 {len(fast_sentences)} 个语速过快的句子")
            
            # 记录原始文本，用于日志
            initial_texts = {idx: all_sentences[idx].trans_text for idx in fast_indices}
            
            # 1. 使用TranslatorActor对语速过快的句子进行精简
            simplified_ref = None
            async for batch_ref in self.simplifier.simplify_sentences.remote(
                fast_sentences,
                target_speed=self.max_speed
            ):
                simplified_ref = batch_ref
            
            if simplified_ref is None:
                self.logger.warning("简化过程未返回任何结果")
                return False
                
            # 2. 使用model_in_actor处理简化后的句子
            refined_sentences = []
            for batch_ref in self.model_in_actor.modelin_maker.remote(
                simplified_ref,
                reuse_speaker=True,
                batch_size=3
            ):
                # 3. 使用Ray Task生成TTS token，传递batch_ref而不是获取结果
                tts_token_ref = generate_tts_tokens.remote(
                    batch_ref,  # 直接传递引用，不使用ray.get()
                    self.cosyvoice_actor
                )
                
                # 获取处理后的句子
                refined_batch = ray.get(tts_token_ref)
                refined_sentences.extend(refined_batch)
            
            # 4. 直接更新原始句子列表
            if refined_sentences:
                for i, orig_idx in enumerate(fast_indices):
                    if i < len(refined_sentences):  # 防止索引越界
                        # 记录文本变化
                        old_text = initial_texts[orig_idx]
                        new_text = refined_sentences[i].trans_text
                        
                        # 更新原始列表中的句子
                        all_sentences[orig_idx] = refined_sentences[i]
                        
                        self.logger.info(f"更新句子[{orig_idx}]: {old_text} -> {new_text}")
                
                return True
            else:
                self.logger.warning("没有获取到精简后的句子")
                return False
                
        except Exception as e:
            self.logger.error(f"_retry_sentences_batch 出错: {e}")
            return False
