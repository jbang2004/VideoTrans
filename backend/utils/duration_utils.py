import logging
import numpy as np
import librosa
from typing import List
from core.sentence_tools import Sentence
from utils.ffmpeg_utils import change_speed_ffmpeg
import asyncio

logger = logging.getLogger(__name__)

async def apply_speed_and_silence(sentences: List[Sentence], sample_rate: int = 24000) -> None:
    """异步应用速度调整和添加静音到句子的音频数据中
    
    Args:
        sentences: 句子列表
        sample_rate: 音频采样率，默认24kHz
    """
    if not sentences:
        logger.warning("apply_speed_and_silence: 收到空句子列表")
        return
    
    task_id = sentences[0].task_id if sentences else "unknown"
    
    for i, sentence in enumerate(sentences):
        try:
            if sentence.generated_audio is None or len(sentence.generated_audio) == 0:
                logger.warning(f"[{task_id}] 句子 {sentence.sentence_id}: 没有可调整的音频数据")
                continue
            
            original_duration = (len(sentence.generated_audio) / sample_rate) * 1000  # 原始时长（毫秒）
            
            # --- 1. 为第一个句子在音频前添加静音 ---
            if hasattr(sentence, 'is_first') and sentence.is_first and hasattr(sentence, 'start') and sentence.start > 0:
                silence_samples = int(sentence.start * sample_rate / 1000)
                if silence_samples > 0:
                    logger.info(f"[{task_id}] 句子 {sentence.sentence_id}: 在开头添加 {sentence.start:.2f}毫秒静音 ({silence_samples} 个采样点)")
                    leading_silence = np.zeros(silence_samples, dtype=np.float32)
                    sentence.generated_audio = np.concatenate([leading_silence, sentence.generated_audio])
                    # 记录操作结果
                    current_duration = (len(sentence.generated_audio) / sample_rate) * 1000
                    logger.warning(f"[{task_id}] 句子 {sentence.sentence_id}: 开头静音已添加，原始时长: {original_duration:.2f}毫秒，新时长: {current_duration:.2f}毫秒")
            
            # --- 2. 应用速度调整 ---
            if hasattr(sentence, 'speed') and sentence.speed != 1.0 and sentence.speed > 0:
                try:
                    logger.warning(f"[{task_id}] 句子 {sentence.sentence_id}: 调整速度至 {sentence.speed}")
                    
                    # 使用 librosa 时间伸缩调整速度
                    audio_np = sentence.generated_audio.astype(np.float32)
                    
                    # 确保音频是单通道
                    if audio_np.ndim > 1:
                        audio_np = audio_np.mean(axis=0)
                    
                    # 使用 FFmpeg atempo 滤镜进行高质量变速
                    sentence.generated_audio = await change_speed_ffmpeg(audio_np, sentence.speed, sample_rate)
                    
                    # 记录操作结果
                    new_duration = (len(sentence.generated_audio) / sample_rate) * 1000
                    logger.warning(f"[{task_id}] 句子 {sentence.sentence_id}: 音频速度已调整，原始时长: {original_duration:.2f}毫秒，新时长: {new_duration:.2f}毫秒")
                
                except Exception as e:
                    logger.error(f"[{task_id}] 句子 {sentence.sentence_id}: 调整速度失败: {e}")
            
            # --- 3. 添加静音 ---
            if hasattr(sentence, 'silence_duration') and sentence.silence_duration > 0:
                try:
                    silence_samples = int(sentence.silence_duration * sample_rate / 1000)
                    logger.info(f"[{task_id}] 句子 {sentence.sentence_id}: 在结尾添加 {sentence.silence_duration:.2f}毫秒静音 ({silence_samples} 个采样点)")
                    
                    # 创建静音数据并拼接到音频末尾
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    sentence.generated_audio = np.concatenate([sentence.generated_audio, silence])
                    
                    # 记录操作结果
                    new_duration = (len(sentence.generated_audio) / sample_rate) * 1000
                    logger.warning(f"[{task_id}] 句子 {sentence.sentence_id}: 结尾静音已添加，原始时长: {original_duration:.2f}毫秒，新时长: {new_duration:.2f}毫秒")
                
                except Exception as e:
                    logger.error(f"[{task_id}] 句子 {sentence.sentence_id}: 添加静音失败: {e}")
            
            # --- 4. 为最后一个句子添加视频结尾静音 ---
            if hasattr(sentence, 'is_last') and sentence.is_last and hasattr(sentence, 'ending_silence') and sentence.ending_silence > 0:
                try:
                    ending_silence_samples = int(sentence.ending_silence * sample_rate / 1000)
                    logger.info(f"[{task_id}] 句子 {sentence.sentence_id}: 为视频结尾添加 {sentence.ending_silence:.2f}毫秒静音 ({ending_silence_samples} 个采样点)")
                    
                    # 创建静音数据并拼接到音频末尾
                    ending_silence = np.zeros(ending_silence_samples, dtype=np.float32)
                    sentence.generated_audio = np.concatenate([sentence.generated_audio, ending_silence])
                    
                    # 记录操作结果
                    new_duration = (len(sentence.generated_audio) / sample_rate) * 1000
                    logger.warning(f"[{task_id}] 句子 {sentence.sentence_id}: 视频结尾静音已添加，原始时长: {original_duration:.2f}毫秒，新时长: {new_duration:.2f}毫秒")
                
                except Exception as e:
                    logger.error(f"[{task_id}] 句子 {sentence.sentence_id}: 添加视频结尾静音失败: {e}")
            
            # --- 5. 更新句子的duration属性以反映最终音频长度 ---
            final_duration = (len(sentence.generated_audio) / sample_rate) * 1000
            sentence.duration = final_duration
            sentence.adjusted_duration = final_duration  # 确保 adjusted_duration 也更新
            
            logger.debug(f"[{task_id}] 句子 {sentence.sentence_id}: 音频调整完成，最终时长: {final_duration:.2f}毫秒")
        
        except Exception as e:
            logger.error(f"[{task_id}] 处理句子 {sentence.sentence_id} 时出错: {e}")

def align_batch(sentences: List[Sentence]) -> List[Sentence]:
    """对句子批次进行时长对齐
    
    Args:
        sentences: 需要对齐的句子列表
        
    Returns:
        对齐后的句子列表
    """
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
                    max_slowdown = s.duration * 0.12
                    slowdown = min(total_needed, max_slowdown)
                    s.adjusted_duration = s.duration + slowdown
                    s.speed = s.duration / max(s.adjusted_duration, 0.001)
                    s.silence_duration = total_needed - slowdown
                    if s.silence_duration > 0:
                        s.adjusted_duration += s.silence_duration

        s.diff = s.duration - s.adjusted_duration
        current_time += s.adjusted_duration
    
    return aligned_sentences 