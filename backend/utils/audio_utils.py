import numpy as np
import soundfile as sf
import logging
from typing import Optional
import asyncio

logger = logging.getLogger(__name__)

def apply_fade_effect(audio_data: np.ndarray, full_audio_buffer: np.ndarray, overlap: int) -> np.ndarray:
    """
    在语音片段衔接处做 overlap 长度的淡入淡出衔接。
    
    Args:
        audio_data: 当前音频数据
        full_audio_buffer: 已累积的音频缓冲区
        overlap: 重叠区域长度（采样点数）
        
    Returns:
        处理后的音频数据
    """
    if audio_data is None or len(audio_data) == 0:
        return np.array([], dtype=np.float32)

    cross_len = min(overlap, len(full_audio_buffer), len(audio_data))
    if cross_len <= 0:
        return audio_data

    fade_out = np.sqrt(np.linspace(1.0, 0.0, cross_len, dtype=np.float32))
    fade_in  = np.sqrt(np.linspace(0.0, 1.0, cross_len, dtype=np.float32))

    audio_data = audio_data.copy()
    overlap_region = full_audio_buffer[-cross_len:]

    audio_data[:cross_len] = overlap_region * fade_out + audio_data[:cross_len] * fade_in
    return audio_data

async def mix_with_background(
    bg_path: str,
    start_time: float,
    duration: float,
    audio_data: np.ndarray,
    sample_rate: int,
    vocals_volume: float,
    background_volume: float
) -> np.ndarray:
    """
    从 bg_path 读取背景音乐，在 [start_time, start_time+duration] 区间截取，
    与 audio_data (人声) 混合。
    
    Args:
        bg_path: 背景音乐文件路径
        start_time: 开始时间（秒）
        duration: 持续时间（秒）
        audio_data: 人声音频数据
        sample_rate: 采样率
        vocals_volume: 人声音量系数
        background_volume: 背景音乐音量系数
        
    Returns:
        混合后的音频数据
    """
    # 异步读取背景音乐
    background_audio, sr = await asyncio.to_thread(sf.read, bg_path)
    background_audio = np.asarray(background_audio, dtype=np.float32)
    if sr != sample_rate:
        logger.warning(
            f"背景音采样率={sr} 与目标={sample_rate}不匹配, 未做重采样, 可能有问题."
        )

    target_length = int(duration * sample_rate)
    start_sample = int(start_time * sample_rate)
    end_sample   = start_sample + target_length

    if end_sample <= len(background_audio):
        bg_segment = background_audio[start_sample:end_sample]
    else:
        bg_segment = background_audio[start_sample:]

    result = np.zeros(target_length, dtype=np.float32)
    audio_len = min(len(audio_data), target_length)
    bg_len    = min(len(bg_segment), target_length)

    # 混合人声 & 背景
    if audio_len > 0:
        result[:audio_len] = audio_data[:audio_len] * vocals_volume
    if bg_len > 0:
        result[:bg_len] += bg_segment[:bg_len] * background_volume

    return result

def normalize_audio(audio_data: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """
    对音频做简单归一化
    
    Args:
        audio_data: 音频数据
        max_val: 最大音量值
        
    Returns:
        归一化后的音频数据
    """
    if len(audio_data) == 0:
        return audio_data
    current_max = np.max(np.abs(audio_data))
    if current_max > max_val:
        audio_data = audio_data * (max_val / current_max)
    return audio_data 