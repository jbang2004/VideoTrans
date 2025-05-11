import os
import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from config import Config
import math
import json

Token = int
Timestamp = Tuple[float, float]
SpeakerSegment = Tuple[float, float, int]

@dataclass
class Sentence:
    raw_text: str
    start: float
    end: float
    speaker_id: int
    trans_text: str = field(default="")
    sentence_id: int = field(default=-1)
    audio: str = field(default="")  # 音频文件路径
    target_duration: float = field(default=None)
    duration: float = field(default=0.0)
    speech_duration: float = field(default=0.0)
    diff: float = field(default=0.0)
    silence_duration: float = field(default=0.0)
    speed: float = field(default=1.0)
    is_first: bool = field(default=False)
    is_last: bool = field(default=False)
    model_input: Dict = field(default_factory=dict)
    generated_audio: np.ndarray = field(default=None)
    adjusted_start: float = field(default=0.0)
    adjusted_duration: float = field(default=0.0)
    task_id: str = field(default="")
    ending_silence: float = field(default=0.0)

def tokens_timestamp_sentence(tokens: List[Token], timestamps: List[Timestamp], speaker_segments: List[SpeakerSegment], tokenizer, config: Config) -> List[Tuple[List[Token], List[Timestamp], int]]:
    sentences = []
    current_tokens = []
    current_timestamps = []
    token_index = 0

    for segment in speaker_segments:
        seg_start_ms = int(segment[0] * 1000)
        seg_end_ms = int(segment[1] * 1000)
        speaker_id = segment[2]

        while token_index < len(tokens):
            token = tokens[token_index]
            token_start, token_end = timestamps[token_index]

            if token_start >= seg_end_ms:
                break
            if token_end <= seg_start_ms:
                token_index += 1
                continue

            current_tokens.append(token)
            current_timestamps.append(timestamps[token_index])
            token_index += 1

            if token in config.STRONG_END_TOKENS and len(current_tokens) <= config.MIN_SENTENCE_LENGTH:
                if sentences:
                    previous_end_time = sentences[-1][1][-1][1]
                    current_start_time = current_timestamps[0][0]
                    time_gap = current_start_time - previous_end_time

                    if time_gap > config.SHORT_SENTENCE_MERGE_THRESHOLD_MS:
                        continue

                    sentences[-1] = (
                        sentences[-1][0] + current_tokens[:],
                        sentences[-1][1] + current_timestamps[:],
                        sentences[-1][2]
                    )
                    current_tokens.clear()
                    current_timestamps.clear()
                continue

            if (token in config.SENTENCE_END_TOKENS or len(current_tokens) > config.MAX_TOKENS_PER_SENTENCE):
                sentences.append((current_tokens[:], current_timestamps[:], speaker_id))
                current_tokens.clear()
                current_timestamps.clear()

        if current_tokens:
            if len(current_tokens) >= config.MIN_SENTENCE_LENGTH or not sentences:
                sentences.append((current_tokens[:], current_timestamps[:], speaker_id))
                current_tokens.clear()
                current_timestamps.clear()
            else:
                continue

    if current_tokens:
        if len(current_tokens) >= config.MIN_SENTENCE_LENGTH or not sentences:
            sentences.append((current_tokens[:], current_timestamps[:], speaker_id))
            current_tokens.clear()
            current_timestamps.clear()
        else:
            sentences[-1] = (
                sentences[-1][0] + current_tokens[:],
                sentences[-1][1] + current_timestamps[:],
                sentences[-1][2]
            )
            current_tokens.clear()
            current_timestamps.clear()

    return sentences

def merge_sentences(raw_sentences: List[Tuple[List[Token], List[Timestamp], int]], 
                   tokenizer,
                   input_duration: float,
                   config: Config) -> List[Sentence]:
    merged_sentences = []
    current = None
    current_tokens_count = 0

    for tokens, timestamps, speaker_id in raw_sentences:
        time_gap = timestamps[0][0] - current.end if current else float('inf')
        
        if (current and 
            current.speaker_id == speaker_id and 
            current_tokens_count + len(tokens) <= config.MAX_TOKENS_PER_SENTENCE and
            time_gap <= config.MAX_GAP_MS):
            current.raw_text += tokenizer.decode(tokens)
            current.end = timestamps[-1][1]
            current_tokens_count += len(tokens)
        else:
            if current:
                current.target_duration = timestamps[0][0] - current.start
                merged_sentences.append(current)
            
            text = tokenizer.decode(tokens)
            current = Sentence(
                raw_text=text, 
                start=timestamps[0][0], 
                end=timestamps[-1][1], 
                speaker_id=speaker_id,
            )
            current_tokens_count = len(tokens)

    if current:
        current.target_duration = current.end - current.start
        current.ending_silence = input_duration - current.end
        merged_sentences.append(current)

    if merged_sentences:
        merged_sentences[0].is_first = True
        merged_sentences[-1].is_last = True

    return merged_sentences

def _extract_segment(speech: torch.Tensor, start: int, end: int, target_samples: int, ignore_samples: int) -> Optional[torch.Tensor]:
    """Helper to extract audio segment based on rules."""
    # Ensure non-negative duration and valid indices
    start = max(0, start)
    end = max(start, end) # Duration can be 0
    duration = end - start

    # Try extracting after ignoring samples
    adj_start = start + ignore_samples
    adj_start = max(0, adj_start) # Ensure non-negative start
    avail_len_adj = end - adj_start

    # Priority 1: Ignore start, extract target_samples if long enough
    if avail_len_adj >= target_samples:
        adj_end = adj_start + target_samples
        # Handle boundary case where adj_end might exceed speech length
        if adj_end <= speech.shape[-1]:
            return speech[:, adj_start : adj_end]
        elif adj_start < speech.shape[-1]: # adj_end exceeds, but adj_start is valid
             return speech[:, adj_start:] # Extract till the end
        else: # adj_start is already out of bounds
             return None

    # Priority 2: Ignore start, extract remaining if > 0 but < target_samples
    elif avail_len_adj > 0:
        # Extract from adj_start to the original end.
        # end is guaranteed to be <= speech.shape[-1] if start was valid initially and duration calc works.
        # Check adj_start boundary just in case.
        if adj_start < speech.shape[-1]:
             return speech[:, adj_start:end]
        else:
             return None # Cannot extract anything valid

    # Priority 3: If ignoring start yields nothing useful, try from original start
    elif duration > 0:
        extract_len = min(target_samples, duration)
        adj_end = start + extract_len
        # Handle boundary case where adj_end might exceed speech length
        if adj_end <= speech.shape[-1]:
            return speech[:, start : adj_end]
        elif start < speech.shape[-1]: # adj_end exceeds, but start is valid
             return speech[:, start:] # Extract till the end
        else: # start is already out of bounds
             return None

    # Final fallback: Cannot extract anything (e.g., duration is 0)
    else:
        return None

def extract_audio(sentences: List[Sentence], speech: torch.Tensor, sr: int, config: Config, 
                 task_id: str = None, task_paths = None) -> List[Sentence]:
    """
    提取每个句子的音频并保存为文件，设置句子的audio属性为文件路径。

    Args:
        sentences: 句子对象列表
        speech: 完整音频波形张量
        sr: 采样率
        config: 配置对象
        task_id: 任务ID (可选)
        task_paths: 任务路径对象 (可选)

    Returns:
        更新了audio字段(文件路径)的句子列表
    """
    target_samples = int(config.SPEAKER_AUDIO_TARGET_DURATION * sr)
    min_samples = int(config.SPEAKER_AUDIO_MIN_DURATION * sr)
    ignore_samples = int(0.5 * sr)  # Consider moving 0.5 to config if variable
    speech = speech.unsqueeze(0) if speech.dim() == 1 else speech # Ensure batch dim

    # 获取音频保存目录 (如果 task_paths 有效且包含该属性)
    audio_prompts_dir = None
    if task_paths is not None and hasattr(task_paths, 'audio_prompts_dir'):
        audio_prompts_dir = Path(task_paths.audio_prompts_dir)
        # 确保目录存在 (TaskPaths.create_directories 应该已经创建)
        audio_prompts_dir.mkdir(parents=True, exist_ok=True)

    for i, s in enumerate(sentences):
        start_sample = int(s.start * sr / 1000)
        end_sample = int(s.end * sr / 1000)

        # Ensure non-negative duration and valid indices
        start_sample = max(0, start_sample)
        end_sample = max(start_sample, end_sample) # duration can be 0

        # Attempt to extract audio
        audio_segment = _extract_segment(speech, start_sample, end_sample, target_samples, ignore_samples)

        # 优化补全：如果片段长度不足，则累积前后同说话人片段直到满足最短长度，并使用 ignore_samples
        if audio_segment is not None and audio_segment.shape[-1] < min_samples:
            needed = min_samples - audio_segment.shape[-1]
            # 向前累积补齐
            for j in range(i-1, -1, -1):
                prev = sentences[j]
                if prev.speaker_id != s.speaker_id:
                    continue
                prev_start = int(prev.start * sr / 1000)
                prev_end   = int(prev.end   * sr / 1000)
                prev_seg = _extract_segment(speech, prev_start, prev_end, needed, ignore_samples)
                if prev_seg is None or prev_seg.shape[-1] == 0:
                    continue
                prev_len = prev_seg.shape[-1]
                if prev_len >= needed:
                    # 截取最后 needed
                    audio_segment = torch.cat([prev_seg[:, -needed:], audio_segment], dim=-1)
                    needed = 0
                    break
                else:
                    # 累积整个片段
                    audio_segment = torch.cat([prev_seg, audio_segment], dim=-1)
                    needed = min_samples - audio_segment.shape[-1]
                    if needed <= 0:
                        break
            # 向后累积补齐
            if needed > 0:
                for j in range(i+1, len(sentences)):
                    nxt = sentences[j]
                    if nxt.speaker_id != s.speaker_id:
                        continue
                    next_start = int(nxt.start * sr / 1000)
                    next_end   = int(nxt.end   * sr / 1000)
                    next_seg = _extract_segment(speech, next_start, next_end, needed, ignore_samples)
                    if next_seg is None or next_seg.shape[-1] == 0:
                        continue
                    next_len = next_seg.shape[-1]
                    if next_len >= needed:
                        # 截取前 needed
                        audio_segment = torch.cat([audio_segment, next_seg[:, :needed]], dim=-1)
                        needed = 0
                        break
                    else:
                        audio_segment = torch.cat([audio_segment, next_seg], dim=-1)
                        needed = min_samples - audio_segment.shape[-1]
                        if needed <= 0:
                            break
            # 最终截断到 min_samples 样本
            if audio_segment.shape[-1] > min_samples:
                audio_segment = audio_segment[:, -min_samples:]

        # 如果有任务ID，设置到句子对象
        if task_id:
            s.task_id = task_id
            
        # 只有当能够提取音频和有保存目录时才保存
        if audio_segment is not None and audio_prompts_dir is not None:
            # 使用任务ID和句子索引创建唯一文件名
            audio_filename = f"{task_id}_s{i}.wav"
            audio_path = audio_prompts_dir / audio_filename
            
            try:
                # 保存音频文件
                torchaudio.save(
                    str(audio_path),
                    audio_segment,
                    sr
                )
                # 设置音频路径到句子对象
                s.audio = str(audio_path)
            except Exception as e:
                print(f"保存音频文件时出错: {e}")
                s.audio = ""  # 保存失败则设置为空字符串
        else:
            s.audio = ""  # 无法提取音频或无保存目录时设置为空字符串

    return sentences

def get_sentences(tokens: List[Token],
                  timestamps: List[Timestamp],
                  speech: torch.Tensor,
                  tokenizer,
                  sd_time_list: List[SpeakerSegment],
                  sample_rate: int = 16000,
                  config: Config = None,
                  task_id: str = None,
                  task_paths = None) -> List[Sentence]:
    """
    获取句子列表，包括音频提取和可选的文件保存，并在最后导出句子信息。

    Args:
        tokens: 文本标记
        timestamps: 时间戳
        speech: 语音音频
        tokenizer: 分词器
        sd_time_list: 说话人分段列表
        sample_rate: 采样率
        config: 配置
        task_id: 任务ID (可选)
        task_paths: 任务路径对象 (可选)

    Returns:
        句子列表
    """
    if config is None:
        config = Config()

    input_duration = (speech.shape[-1] / sample_rate) * 1000

    raw_sentences = tokens_timestamp_sentence(tokens, timestamps, sd_time_list, tokenizer, config)
    merged_sentences = merge_sentences(raw_sentences, tokenizer, input_duration, config)
    sentences_with_audio = extract_audio(merged_sentences, speech, sample_rate, config, 
                                         task_id, task_paths)

    return sentences_with_audio
