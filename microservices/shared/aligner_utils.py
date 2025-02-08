import logging
from typing import List
from .concurrency import run_sync

logger = logging.getLogger(__name__)

class DurationAligner:
    """
    负责句子时长对齐以及针对语速过快句子进行简化、重新生成 TTS token 的处理，
    参考 backend/core/timeadjust/duration_aligner.py 实现。
    """
    def __init__(self, model_in=None, simplifier=None, tts_token_gener=None, max_speed: float = 1.1, sample_rate: int = 24000):
        self.model_in = model_in
        self.simplifier = simplifier
        self.tts_token_gener = tts_token_gener
        self.max_speed = max_speed
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

    async def align_durations(self, sentences: List) -> None:
        if not sentences:
            return
        # 初次对齐
        self._align_batch(sentences)
        # 检查语速过快的句子
        retry_sentences = [s for s in sentences if s.speed > self.max_speed]
        if retry_sentences:
            self.logger.info(f"{len(retry_sentences)} sentences exceed speed threshold; retrying simplification...")
            success = await self._retry_sentences_batch(retry_sentences)
            if success:
                self._align_batch(sentences)
            else:
                self.logger.warning("Simplification failed, keeping original durations.")

    def _align_batch(self, sentences: List) -> None:
        if not sentences:
            return
        total_diff = 0
        for s in sentences:
            s.diff = s.duration - (s.target_duration if s.target_duration is not None else s.duration)
            total_diff += s.diff
        pos_diff = sum(s.diff for s in sentences if s.diff > 0)
        neg_diff = sum(abs(s.diff) for s in sentences if s.diff < 0)
        current_time = sentences[0].start if sentences else 0
        for s in sentences:
            s.adjusted_start = current_time
            s.adjusted_duration = s.duration
            s.speed = 1.0
            s.silence_duration = 0.0
            if total_diff != 0:
                if total_diff > 0 and s.diff > 0 and pos_diff > 0:
                    proportion = s.diff / pos_diff
                    adjustment = total_diff * proportion
                    s.adjusted_duration = s.duration - adjustment
                    s.speed = s.duration / max(s.adjusted_duration, 0.001)
                elif total_diff < 0 and s.diff < 0 and neg_diff > 0:
                    proportion = abs(s.diff) / neg_diff
                    needed = abs(total_diff) * proportion
                    max_slow = s.duration * 0.07
                    slowdown = min(needed, max_slow)
                    s.adjusted_duration = s.duration + slowdown
                    s.speed = s.duration / max(s.adjusted_duration, 0.001)
                    s.silence_duration = needed - slowdown
                    if s.silence_duration > 0:
                        s.adjusted_duration += s.silence_duration
            s.diff = s.duration - s.adjusted_duration
            current_time += s.adjusted_duration

    async def _retry_sentences_batch(self, sentences: List) -> bool:
        try:
            # 调用简化接口（例如 Translator.simplify_sentences），此处假定简化接口为异步生成器
            if self.simplifier:
                async for _ in self.simplifier.simplify_sentences(sentences, batch_size=4, target_speed=self.max_speed):
                    pass
            # 更新文本特征并生成新的 TTS token
            if self.model_in:
                async for batch in self.model_in.modelin_maker(sentences, reuse_speaker=True, reuse_uuid=True, batch_size=3):
                    if self.tts_token_gener:
                        await self.tts_token_gener.tts_token_maker(batch, reuse_uuid=True)
            return True
        except Exception as e:
            self.logger.error(f"Retry (simplification/TTS update) failed: {e}")
            return False

class TimestampAdjuster:
    """
    负责根据生成的音频更新句子的时间戳信息，并验证时间戳连续性，
    参考 backend/core/timeadjust/timestamp_adjuster.py 实现。
    """
    def __init__(self, sample_rate: int):
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate

    def update_timestamps(self, sentences: List, start_time: float = None) -> float:
        if not sentences:
            return start_time if start_time is not None else 0
        current_time = start_time if start_time is not None else sentences[0].start
        for s in sentences:
            if s.generated_audio is not None:
                actual_duration = (len(s.generated_audio) / self.sample_rate) * 1000
            else:
                actual_duration = 0
                self.logger.warning(f"Sentence {getattr(s, 'sentence_id', 'unknown')} missing generated audio.")
            s.adjusted_start = current_time
            s.adjusted_duration = actual_duration
            s.diff = s.duration - actual_duration
            current_time += actual_duration
        return current_time

    def validate_timestamps(self, sentences: List) -> bool:
        if not sentences:
            return True
        for i in range(len(sentences) - 1):
            curr = sentences[i]
            nxt = sentences[i + 1]
            expected = curr.adjusted_start + curr.adjusted_duration
            if abs(nxt.adjusted_start - expected) > 1:
                self.logger.error(f"Timestamps discontinuity: sentence {curr.sentence_id} expected {expected}, got {nxt.adjusted_start}")
                return False
            if curr.adjusted_duration <= 0:
                self.logger.error(f"Invalid adjusted duration: {curr.adjusted_duration} for sentence {curr.sentence_id}")
                return False
        return True
