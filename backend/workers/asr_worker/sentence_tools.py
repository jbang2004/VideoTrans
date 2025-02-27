import os
import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path
from config import Config

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
    audio: torch.Tensor = field(default=None)
    target_duration: float = field(default=None)
    duration: float = field(default=0.0)
    diff: float = field(default=0.0)
    silence_duration: float = field(default=0.0)
    speed: float = field(default=1.0)
    is_first: bool = field(default=False)
    is_last: bool = field(default=False)
    model_input: Dict = field(default_factory=dict)
    generated_audio: np.ndarray = field(default=None)
    adjusted_start: float = field(default=0.0)
    adjusted_duration: float = field(default=0.0)
    segment_index: int = field(default=-1)
    segment_start: float = field(default=0.0)
    task_id: str = field(default="")

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
        current.target_duration = input_duration - current.start
        merged_sentences.append(current)

    if merged_sentences:
        merged_sentences[0].is_first = True
        merged_sentences[-1].is_last = True

    return merged_sentences

def extract_audio(sentences: List[Sentence], speech: torch.Tensor, sr: int, config: Config) -> List[Sentence]:
    target_samples = int(config.SPEAKER_AUDIO_TARGET_DURATION * sr)
    speech = speech.unsqueeze(0) if speech.dim() == 1 else speech

    speaker_segments: Dict[int, List[Tuple[int, int, int]]] = {}
    for idx, s in enumerate(sentences):
        start_sample = int(s.start * sr / 1000)
        end_sample = int(s.end * sr / 1000)
        speaker_segments.setdefault(s.speaker_id, []).append((start_sample, end_sample, idx))

    speaker_audio_cache: Dict[int, torch.Tensor] = {}

    for speaker_id, segments in speaker_segments.items():
        segments.sort(key=lambda x: x[1] - x[0], reverse=True)
        longest_start, longest_end, _ = segments[0]

        ignore_samples = int(0.5 * sr)
        adjusted_start = longest_start + ignore_samples
        available_length_adjusted = longest_end - adjusted_start

        if available_length_adjusted > 0:
            audio_length = min(target_samples, available_length_adjusted)
            speaker_audio = speech[:, adjusted_start:adjusted_start + audio_length]
        else:
            available_length_original = longest_end - longest_start
            audio_length = min(target_samples, available_length_original)
            speaker_audio = speech[:, longest_start:longest_start + audio_length]

        speaker_audio_cache[speaker_id] = speaker_audio

    for sentence in sentences:
        sentence.audio = speaker_audio_cache.get(sentence.speaker_id)

    output_dir = Path(config.TASKS_DIR) / sentences[0].task_id / 'speakers'
    output_dir.mkdir(parents=True, exist_ok=True)

    for speaker_id, audio in speaker_audio_cache.items():
        if audio is not None:
            output_path = output_dir / f'speaker_{speaker_id}.wav'
            torchaudio.save(str(output_path), audio, sr)

    return sentences

def get_sentences(tokens: List[Token],
                  timestamps: List[Timestamp],
                  speech: torch.Tensor,
                  tokenizer,
                  sd_time_list: List[SpeakerSegment],
                  sample_rate: int = 16000,
                  config: Config = None) -> List[Sentence]:
    if config is None:
        config = Config()

    input_duration = (speech.shape[-1] / sample_rate) * 1000

    raw_sentences = tokens_timestamp_sentence(tokens, timestamps, sd_time_list, tokenizer, config)
    merged_sentences = merge_sentences(raw_sentences, tokenizer, input_duration, config)
    sentences_with_audio = extract_audio(merged_sentences, speech, sample_rate, config)

    return sentences_with_audio 