import os
import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Sentence:
    raw_text: str
    start: float
    end: float
    speaker_id: int
    trans_text: str = ""
    sentence_id: int = -1
    audio: torch.Tensor = None
    target_duration: float = None
    duration: float = 0.0
    diff: float = 0.0
    silence_duration: float = 0.0
    speed: float = 1.0
    is_first: bool = False
    is_last: bool = False
    model_input: Dict = field(default_factory=dict)
    generated_audio: any = None
    adjusted_start: float = 0.0
    adjusted_duration: float = 0.0
    segment_index: int = -1
    segment_start: float = 0.0
    task_id: str = ""

    def to_dict(self) -> Dict[str, any]:
        return {
            "raw_text": self.raw_text,
            "trans_text": self.trans_text,
            "start": self.start,
            "end": self.end,
            "speaker_id": self.speaker_id,
            "sentence_id": self.sentence_id,
            "segment_index": self.segment_index,
            "segment_start": self.segment_start,
            "task_id": self.task_id,
            "duration": self.duration,
            "speed": self.speed,
            "silence_duration": self.silence_duration,
            "adjusted_duration": self.adjusted_duration,
            "adjusted_start": self.adjusted_start,
            "diff": self.diff,
            "target_duration": self.target_duration,
            "model_input": self.model_input,
            "generated_audio": self.generated_audio,
            "is_first": self.is_first,
            "is_last": self.is_last
        }

    @classmethod
    def from_dict(cls, d: Dict[str, any]):
        return cls(
            raw_text=d.get("raw_text", ""),
            trans_text=d.get("trans_text", ""),
            start=d.get("start", 0),
            end=d.get("end", 0),
            speaker_id=d.get("speaker_id", 0),
            sentence_id=d.get("sentence_id", -1),
            segment_index=d.get("segment_index", -1),
            segment_start=d.get("segment_start", 0),
            task_id=d.get("task_id", ""),
            duration=d.get("duration", 0),
            speed=d.get("speed", 1.0),
            silence_duration=d.get("silence_duration", 0),
            adjusted_duration=d.get("adjusted_duration", 0),
            adjusted_start=d.get("adjusted_start", 0),
            diff=d.get("diff", 0),
            target_duration=d.get("target_duration", 0),
            model_input=d.get("model_input", {}),
            generated_audio=d.get("generated_audio", None),
            is_first=d.get("is_first", False),
            is_last=d.get("is_last", False)
        )

def sentences_to_dict_list(sents: List[Sentence]) -> List[Dict[str, any]]:
    return [s.to_dict() for s in sents]

def dict_list_to_sentences(dicts: List[Dict[str, any]]) -> List[Sentence]:
    return [Sentence.from_dict(d) for d in dicts]

def tokens_timestamp_sentence(tokens: List[int], timestamps: List[Tuple[float, float]], speaker_segments: List[Tuple[float, float, int]], tokenizer, config) -> List[Tuple[List[int], List[Tuple[float, float]], int]]:
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
                    previous_end = sentences[-1][1][-1][1]
                    current_start = current_timestamps[0][0]
                    if current_start - previous_end > config.SHORT_SENTENCE_MERGE_THRESHOLD_MS:
                        continue
                    sentences[-1] = (sentences[-1][0] + current_tokens, sentences[-1][1] + current_timestamps, sentences[-1][2])
                    current_tokens = []
                    current_timestamps = []
                continue
            if token in config.STRONG_END_TOKENS or len(current_tokens) > config.MAX_TOKENS_PER_SENTENCE:
                sentences.append((current_tokens.copy(), current_timestamps.copy(), speaker_id))
                current_tokens = []
                current_timestamps = []
        if current_tokens:
            if len(current_tokens) >= config.MIN_SENTENCE_LENGTH or not sentences:
                sentences.append((current_tokens.copy(), current_timestamps.copy(), speaker_id))
                current_tokens = []
                current_timestamps = []
            else:
                if sentences:
                    sentences[-1] = (sentences[-1][0] + current_tokens, sentences[-1][1] + current_timestamps, sentences[-1][2])
                current_tokens = []
                current_timestamps = []
    if current_tokens:
        if len(current_tokens) >= config.MIN_SENTENCE_LENGTH or not sentences:
            sentences.append((current_tokens.copy(), current_timestamps.copy(), speaker_id))
        else:
            if sentences:
                sentences[-1] = (sentences[-1][0] + current_tokens, sentences[-1][1] + current_timestamps, sentences[-1][2])
    return sentences

def merge_sentences(raw_sentences: List[Tuple[List[int], List[Tuple[float, float]], int]], tokenizer, input_duration: float, config) -> List[Sentence]:
    merged_sentences = []
    current = None
    current_tokens_count = 0
    for tokens, timestamps, speaker_id in raw_sentences:
        time_gap = timestamps[0][0] - (current.end if current else float('inf'))
        if current and current.speaker_id == speaker_id and current_tokens_count + len(tokens) <= config.MAX_TOKENS_PER_SENTENCE and time_gap <= config.MAX_GAP_MS:
            current.raw_text += tokenizer.decode(tokens)
            current.end = timestamps[-1][1]
            current_tokens_count += len(tokens)
        else:
            if current:
                current.target_duration = timestamps[0][0] - current.start
                merged_sentences.append(current)
            text = tokenizer.decode(tokens)
            current = Sentence(raw_text=text, start=timestamps[0][0], end=timestamps[-1][1], speaker_id=speaker_id)
            current_tokens_count = len(tokens)
    if current:
        current.target_duration = input_duration - current.start
        merged_sentences.append(current)
    if merged_sentences:
        merged_sentences[0].is_first = True
        merged_sentences[-1].is_last = True
    return merged_sentences

def extract_audio(sentences: List[Sentence], speech: torch.Tensor, sr: int, config) -> List[Sentence]:
    target_samples = int(config.SPEAKER_AUDIO_TARGET_DURATION * sr)
    if speech.dim() == 1:
        speech = speech.unsqueeze(0)
    speaker_segments = {}
    for idx, s in enumerate(sentences):
        start_sample = int(s.start * sr / 1000)
        end_sample = int(s.end * sr / 1000)
        speaker_segments.setdefault(s.speaker_id, []).append((start_sample, end_sample, idx))
    speaker_audio_cache = {}
    for speaker_id, segments in speaker_segments.items():
        segments.sort(key=lambda x: x[1]-x[0], reverse=True)
        longest = segments[0]
        ignore_samples = int(0.5 * sr)
        adjusted_start = longest[0] + ignore_samples
        available = longest[1] - adjusted_start
        if available > 0:
            audio_length = min(target_samples, available)
            speaker_audio = speech[:, adjusted_start:adjusted_start+audio_length]
        else:
            audio_length = min(target_samples, longest[1]-longest[0])
            speaker_audio = speech[:, longest[0]:longest[0]+audio_length]
        speaker_audio_cache[speaker_id] = speaker_audio
    for s in sentences:
        s.audio = speaker_audio_cache.get(s.speaker_id)
    output_dir = Path(config.TASKS_DIR) / sentences[0].task_id / 'speakers'
    output_dir.mkdir(parents=True, exist_ok=True)
    import torchaudio
    for speaker_id, audio in speaker_audio_cache.items():
        if audio is not None:
            output_path = output_dir / f'speaker_{speaker_id}.wav'
            torchaudio.save(str(output_path), audio, sr)
    return sentences

def get_sentences(tokens: List[int], timestamps: List[Tuple[float, float]], speech: torch.Tensor, tokenizer, sd_time_list: List[Tuple[float, float, int]], sample_rate: int = 16000, config=None) -> List[Sentence]:
    if config is None:
        from shared.config import Config
        config = Config()
    input_duration = (speech.shape[-1] / sample_rate) * 1000
    raw_sentences = tokens_timestamp_sentence(tokens, timestamps, sd_time_list, tokenizer, config)
    merged = merge_sentences(raw_sentences, tokenizer, input_duration, config)
    sentences_with_audio = extract_audio(merged, speech, sample_rate, config)
    return sentences_with_audio
