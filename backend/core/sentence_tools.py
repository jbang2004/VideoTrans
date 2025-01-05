# speech_processing.py
import os
import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from config import Config

# 定义数据结构
Token = int
Timestamp = Tuple[float, float]
SpeakerSegment = Tuple[float, float, int]

@dataclass
class Sentence:
    """
    句子数据结构
    
    Attributes:
        raw_text: 原始句子文本
        trans_text: 翻译后的文本
        start: 开始时间（毫秒）
        end: 结束时间（毫秒）
        speaker_id: 说话人ID
        sentence_id: 句子ID（全局唯一递增）
        audio: 音频数据
        target_duration: 目标音频长度（毫秒）
        duration: 实际音频长度（毫秒）
        diff: 时长差异（毫秒）
        silence_duration: 静音时长（毫秒）
        speed: 播放速度
        speaker_changed: 说话人是否改变
        is_first: 是否为第一个句子
        is_last: 是否为最后一个句子
        model_input: 用于TTS生成的输入数据
        generated_audio: 存储生成的音频数据
        adjusted_start: 调整后的开始时间（毫秒）
        adjusted_duration: 调整后的时长（毫秒）
        segment_index: 所属分段的索引
        segment_start: 所属分段的起始时间（秒）
    """
    raw_text: str
    start: float  # 单位：毫秒
    end: float  # 单位：毫秒
    speaker_id: int
    trans_text: str = field(default="")
    sentence_id: int = field(default=-1)
    audio: torch.Tensor = field(default=None)
    target_duration: float = field(default=None)  # 单位：毫秒
    duration: float = field(default=0.0)
    diff: float = field(default=0.0)
    adjusted_duration: float = field(default=0.0)  # 单位：毫秒
    silence_duration: float = field(default=0.0)
    speed: float = field(default=1.0)
    is_first: bool = field(default=False)
    is_last: bool = field(default=False)
    model_input: Dict = field(default_factory=dict)  # 用于存储与TTS相关的数据
    generated_audio: np.ndarray = field(default=None)  # 存储生成的音频数据
    adjusted_start: float = field(default=0.0)  # 单位：毫秒
    segment_index: int = field(default=-1)  # 所属分段的索引
    segment_start: float = field(default=0.0)  # 所属分段的起始时间（秒）

def tokens_timestamp_sentence(tokens: List[Token], timestamps: List[Timestamp], speaker_segments: List[SpeakerSegment], tokenizer: Any, config: Config) -> List[Tuple[List[Token], List[Timestamp], int]]:
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
                    # 计算当前句子与前一个句子的时间差
                    previous_end_time = sentences[-1][1][-1][1]
                    current_start_time = current_timestamps[0][0]
                    time_gap = current_start_time - previous_end_time

                    if time_gap > config.SHORT_SENTENCE_MERGE_THRESHOLD_MS:
                        continue

                    # 将当前短句子与前一个句子合并
                    sentences[-1] = (
                        sentences[-1][0] + current_tokens[:],
                        sentences[-1][1] + current_timestamps[:],
                        sentences[-1][2]
                    )
                    current_tokens.clear()
                    current_timestamps.clear()
                continue

            if (token in config.STRONG_END_TOKENS or len(current_tokens) > config.MAX_TOKENS_PER_SENTENCE):
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
                   tokenizer: Any,
                   input_duration: float,  # 输入音频总长度(ms)
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
        current.target_duration = input_duration - current.start  # 最后一个句子的目标时长直到输入结束
        merged_sentences.append(current)

    # 标记第一个和最后一个句子
    if merged_sentences:
        merged_sentences[0].is_first = True
        merged_sentences[-1].is_last = True

    return merged_sentences

def extract_audio(sentences: List[Sentence], speech: torch.Tensor, sr: int, config: Config) -> List[Sentence]:
    target_samples = int(config.SPEAKER_AUDIO_TARGET_DURATION * sr)
    speech = speech.unsqueeze(0) if speech.dim() == 1 else speech

    # 按说话人ID分组并计算每段音频长度
    speaker_segments: Dict[int, List[Tuple[int, int, int]]] = {}  # speaker_id -> List[(start_sample, end_sample, sentence_idx)]
    for idx, s in enumerate(sentences):
        start_sample = int(s.start * sr / 1000)
        end_sample = int(s.end * sr / 1000)
        speaker_segments.setdefault(s.speaker_id, []).append((start_sample, end_sample, idx))

    # 为每个说话人选择最长的音频片段
    speaker_audio_cache: Dict[int, torch.Tensor] = {}  # 缓存每个说话人的音频片段

    for speaker_id, segments in speaker_segments.items():
        # 按音频长度降序排序
        segments.sort(key=lambda x: x[1] - x[0], reverse=True)

        # 获取最长的音频片段
        longest_start, longest_end, _ = segments[0]

        # 计算需要忽略的起始样本数
        ignore_samples = int(0.5 * sr)

        # 计算调整后的起始位置
        adjusted_start = longest_start + ignore_samples

        # 计算从调整后的起始位置开始的可用长度
        available_length_adjusted = longest_end - adjusted_start

        if available_length_adjusted > 0:
            # 如果忽略前 0.5 秒后还有足够的音频
            audio_length = min(target_samples, available_length_adjusted)
            speaker_audio = speech[:, adjusted_start:adjusted_start + audio_length]
        else:
            # 如果忽略前 0.5 秒后没有足够的音频，则使用原始的音频片段长度
            available_length_original = longest_end - longest_start
            audio_length = min(target_samples, available_length_original)
            speaker_audio = speech[:, longest_start:longest_start + audio_length]

        # 缓存该说话人的音频片段
        speaker_audio_cache[speaker_id] = speaker_audio

    # 为所有句子音频，使用引用而不是复制
    for sentence in sentences:
        sentence.audio = speaker_audio_cache.get(sentence.speaker_id)

    # 修改输出目录到 temp/spk_cuts
    # output_dir = Path(config.BASE_TEMP_DIR) / 'spk_cuts'
    # output_dir.mkdir(parents=True, exist_ok=True)

    # # 只保存每个说话人的一个音频文件
    # for speaker_id, audio in speaker_audio_cache.items():
    #     if audio is not None:
    #         output_path = output_dir / f'spk_{speaker_id}.wav'
    #         torchaudio.save(str(output_path), audio, sr)

    return sentences

def get_sentences(tokens: List[Token],
                  timestamps: List[Timestamp],
                  speech: torch.Tensor,
                  tokenizer: Any,
                  sd_time_list: List[SpeakerSegment],
                  sample_rate: int = 16000,
                  config: Config = None) -> List[Sentence]:
    """
    主处理函数，整合所有步骤。
    """
    if config is None:
        config = Config()  # 使用默认配置

    # 计算输入音频的总长度(ms)
    input_duration = (speech.shape[-1] / sample_rate) * 1000

    raw_sentences = tokens_timestamp_sentence(tokens, timestamps, sd_time_list, tokenizer, config)
    merged_sentences = merge_sentences(raw_sentences, tokenizer, input_duration, config)
    sentences_with_audio = extract_audio(merged_sentences, speech, sample_rate, config)

    return sentences_with_audio