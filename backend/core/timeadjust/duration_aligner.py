import logging
import ray

class DurationAligner:
    def __init__(self, model_in=None, simplifier=None, tts_token_gener=None, max_speed=1.1):
        """
        model_in：生成模型接口，用于更新文本特征  
        simplifier：简化处理接口（TranslatorActor）  
        tts_token_gener：TTS token 生成接口  
        max_speed：语速阈值，超过该速率的句子需要进行简化
        """
        self.model_in = model_in
        self.simplifier = simplifier
        self.tts_token_gener = tts_token_gener
        self.max_speed = max_speed
        self.logger = logging.getLogger(__name__)

    async def align_durations(self, sentences):
        """
        对整批句子进行时长对齐，并检查是否需要精简（语速过快）。
        """
        if not sentences:
            return

        # 第一次对齐
        self._align_batch(sentences)

        # 查找语速过快的句子（speed > max_speed）
        retry_sentences = [s for s in sentences if s.speed > self.max_speed]
        if retry_sentences:
            self.logger.info(f"{len(retry_sentences)} 个句子语速过快, 正在精简...")
            success = await self._retry_sentences_batch(retry_sentences)
            if success:
                # 若精简文本成功，再次对齐
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

    async def _retry_sentences_batch(self, sentences):
        """
        对语速过快的句子执行精简 + 更新 TTS token。
        使用TranslatorActor而不是原来的Translator
        """
        try:
            # 1. 使用TranslatorActor对语速过快的句子进行精简
            async for simplified_batch_ref in self.simplifier.simplify_sentences.remote(
                sentences,
                target_speed=self.max_speed
            ):
                simplified_batch = await simplified_batch_ref
            
            if not simplified_batch:
                self.logger.warning("简化结果为空")
                return False
                
            # 2. 批量更新文本特征（复用 speaker 与 uuid）
            async for batch in self.model_in.modelin_maker(
                simplified_batch,
                reuse_speaker=True,
                reuse_uuid=True,
                batch_size=3
            ):
                # 3. 再生成 token（复用 uuid）
                updated_batch = await self.tts_token_gener.tts_token_maker(batch, reuse_uuid=True)
                
            return True
        except Exception as e:
            self.logger.error(f"_retry_sentences_batch 出错: {e}")
            return False
