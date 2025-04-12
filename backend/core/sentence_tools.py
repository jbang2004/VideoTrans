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
    if duration < 0: return None # Should not happen with max() guards, but safety first

    # Try extracting after ignoring samples
    adj_start = start + ignore_samples
    adj_start = max(0, adj_start) # Ensure non-negative start
    avail_len_adj = end - adj_start

    if avail_len_adj >= target_samples:
        adj_end = adj_start + target_samples
        if adj_end <= speech.shape[-1]:
            return speech[:, adj_start : adj_end]
        else:
             # If adjusted end goes beyond speech length, but start was valid
             # take what's available from adjusted start
             if adj_start < speech.shape[-1]:
                  return speech[:, adj_start:]
             else:
                  return None # Cannot extract anything valid

    # If ignoring makes it too short or invalid, try from the original start
    elif duration > 0:
        extract_len = min(target_samples, duration)
        adj_end = start + extract_len
        if adj_end <= speech.shape[-1]:
            return speech[:, start : adj_end]
        else:
            # If end goes beyond speech length, take what's available
            if start < speech.shape[-1]:
                 return speech[:, start:]
            else:
                 return None # Cannot extract anything valid
    else:
        # Duration is 0 or negative (after guards, should be 0)
        return None

def extract_audio(sentences: List[Sentence], speech: torch.Tensor, sr: int, config: Config) -> List[Sentence]:
    """
    Extracts audio for EACH sentence based on simplified priority:
    1. Current sentence (if long enough after ignore).
    2. Nearest neighbor (before/after, same speaker, long enough).
    3. Absolute longest sentence (same speaker).
    No files are saved.

    Args:
        sentences: List of Sentence objects.
        speech: The full audio waveform tensor.
        sr: Sample rate.
        config: Configuration object.

    Returns:
        The list of sentences with the .audio field populated for each sentence.
    """
    target_samples = int(config.SPEAKER_AUDIO_TARGET_DURATION * sr)
    ignore_samples = int(0.5 * sr)
    speech = speech.unsqueeze(0) if speech.dim() == 1 else speech # Ensure batch dim

    # --- Pre-computation ---
    num_sentences = len(sentences)
    sentence_details = []
    long_enough_indices: Dict[int, List[int]] = {} # speaker_id -> list of indices of long-enough sentences
    speaker_longest_idx: Dict[int, int] = {} # speaker_id -> index of the longest sentence

    for idx, s in enumerate(sentences):
        start_sample = int(s.start * sr / 1000)
        end_sample = int(s.end * sr / 1000)
        speaker_id = s.speaker_id

        # Ensure non-negative duration and valid indices
        start_sample = max(0, start_sample)
        end_sample = max(start_sample, end_sample) # duration can be 0
        duration = end_sample - start_sample

        adjusted_start = start_sample + ignore_samples
        available_length_adjusted = end_sample - adjusted_start
        is_long_enough = available_length_adjusted >= target_samples

        details = {
            "index": idx,
            "speaker_id": speaker_id,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "duration": duration,
            "is_long_enough": is_long_enough
        }
        sentence_details.append(details)

        if is_long_enough:
            long_enough_indices.setdefault(speaker_id, []).append(idx)

        # Track longest sentence index per speaker
        if speaker_id not in speaker_longest_idx or duration > sentence_details[speaker_longest_idx[speaker_id]]["duration"]:
            speaker_longest_idx[speaker_id] = idx

    # --- Assign Audio per Sentence ---
    for i, s in enumerate(sentences):
        details = sentence_details[i]
        speaker_id = details["speaker_id"]
        audio_segment = None

        # Rule 1: Try current sentence
        if details["is_long_enough"]:
            audio_segment = _extract_segment(speech, details["start_sample"], details["end_sample"], target_samples, ignore_samples)

        # Rule 2: Find nearest long-enough neighbor (same speaker)
        if audio_segment is None and speaker_id in long_enough_indices:
            possible_indices = long_enough_indices[speaker_id]
            nearest_idx = -1
            min_dist = float('inf')

            for idx in possible_indices:
                if idx == i: continue # Skip self
                dist = abs(idx - i)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
                # If distance is equal, prefer the one that comes earlier? Or later? Let's stick with the first closest found.

            if nearest_idx != -1:
                neighbor_details = sentence_details[nearest_idx]
                audio_segment = _extract_segment(speech, neighbor_details["start_sample"], neighbor_details["end_sample"], target_samples, ignore_samples)

        # Rule 3: Use the absolute longest sentence (same speaker)
        if audio_segment is None and speaker_id in speaker_longest_idx:
            longest_idx = speaker_longest_idx[speaker_id]
            longest_details = sentence_details[longest_idx]
            # Only extract from longest if it has positive duration
            if longest_details["duration"] > 0:
                audio_segment = _extract_segment(speech, longest_details["start_sample"], longest_details["end_sample"], target_samples, ignore_samples)

        # Assign the determined audio segment (can be None if all fails)
        s.audio = audio_segment

    # Clean up precomputed dictionaries (optional)
    del sentence_details
    del long_enough_indices
    del speaker_longest_idx

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
