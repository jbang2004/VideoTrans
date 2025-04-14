import os
import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from pathlib import Path
from config import Config
import math

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

def extract_audio(sentences: List[Sentence], speech: torch.Tensor, sr: int, config: Config) -> List[Sentence]:
    """
    Extracts audio for EACH sentence based ONLY on the current sentence's audio.
    No fallback to neighbors or longest sentence.
    No files are saved.

    Args:
        sentences: List of Sentence objects.
        speech: The full audio waveform tensor.
        sr: Sample rate.
        config: Configuration object.

    Returns:
        The list of sentences with the .audio field populated (or None if extraction failed).
    """
    target_samples = int(config.SPEAKER_AUDIO_TARGET_DURATION * sr)
    ignore_samples = int(0.5 * sr)  # Consider moving 0.5 to config if variable
    speech = speech.unsqueeze(0) if speech.dim() == 1 else speech # Ensure batch dim

    for s in sentences:
        start_sample = int(s.start * sr / 1000)
        end_sample = int(s.end * sr / 1000)

        # Ensure non-negative duration and valid indices
        start_sample = max(0, start_sample)
        end_sample = max(start_sample, end_sample) # duration can be 0

        # Attempt to extract audio only from the current sentence
        audio_segment = _extract_segment(speech, start_sample, end_sample, target_samples, ignore_samples)

        # Assign the determined audio segment (can be None if extraction failed)
        s.audio = audio_segment

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
