import numpy as np
import logging
import soundfile as sf
import os
import asyncio
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import List
import pysubs2

from .decorators import handle_errors
from .ffmpeg_utils import FFmpegTool
from shared.sentence_tools import Sentence

logger = logging.getLogger(__name__)

class MediaMixer:
    """
    用于将多句合成音频与原视频混合，并生成带字幕的视频。
    """
    def __init__(self, config, sample_rate: int):
        self.config = config
        self.sample_rate = sample_rate
        self.max_val = 1.0
        self.overlap = config.AUDIO_OVERLAP
        self.vocals_volume = config.VOCALS_VOLUME
        self.background_volume = config.BACKGROUND_VOLUME
        self.full_audio_buffer = np.array([], dtype=np.float32)
        self.ffmpeg_tool = FFmpegTool()

    @handle_errors(logger)
    async def mix_sentences(self, sentences: List[Sentence], task_state, output_path: str) -> bool:
        """
        将若干句子对应的音频合并为一段完整的音轨，与video合并，并可写字幕.
        """
        if not sentences:
            logger.warning("mix_sentences: Empty sentence list")
            return False
        segment_index = sentences[0].segment_index
        segment_files = task_state.segment_media_files.get(segment_index)
        if not segment_files:
            logger.error(f"No media files for segment {segment_index}")
            return False

        full_audio = np.array([], dtype=np.float32)
        for sentence in sentences:
            if sentence.generated_audio is not None:
                audio_data = np.asarray(sentence.generated_audio, dtype=np.float32)
                if len(full_audio) > 0:
                    audio_data = self._apply_fade_effect(audio_data)
                full_audio = np.concatenate((full_audio, audio_data))
            else:
                logger.warning(f"Missing audio for sentence: {sentence.raw_text}")

        if len(full_audio) == 0:
            logger.error("mix_sentences: No valid audio data")
            return False

        start_time = 0.0
        if not sentences[0].is_first:
            start_time = (sentences[0].adjusted_start - sentences[0].segment_start * 1000) / 1000.0
        duration = sum(s.adjusted_duration for s in sentences) / 1000.0

        background_audio_path = segment_files.get('background')
        if background_audio_path is not None:
            full_audio = self._mix_with_background(background_audio_path, start_time, duration, full_audio)
            full_audio = self._normalize_audio(full_audio)

        self.full_audio_buffer = np.concatenate((self.full_audio_buffer, full_audio))
        video_path = segment_files.get('video')
        if video_path:
            await self._add_video_segment(video_path, start_time, duration, full_audio, output_path, sentences, task_state)
            return True

        logger.warning("mix_sentences: No video path available")
        return False

    def _apply_fade_effect(self, audio_data: np.ndarray) -> np.ndarray:
        if audio_data is None or len(audio_data) == 0:
            return np.array([], dtype=np.float32)
        cross_len = min(self.overlap, len(self.full_audio_buffer), len(audio_data))
        if cross_len <= 0:
            return audio_data
        fade_out = np.sqrt(np.linspace(1.0, 0.0, cross_len, dtype=np.float32))
        fade_in = np.sqrt(np.linspace(0.0, 1.0, cross_len, dtype=np.float32))
        audio_data = audio_data.copy()
        overlap_region = self.full_audio_buffer[-cross_len:]
        audio_data[:cross_len] = overlap_region * fade_out + audio_data[:cross_len] * fade_in
        return audio_data

    def _mix_with_background(self, bg_path: str, start_time: float, duration: float, audio_data: np.ndarray) -> np.ndarray:
        background_audio, sr = sf.read(bg_path)
        background_audio = np.asarray(background_audio, dtype=np.float32)
        if sr != self.sample_rate:
            logger.warning(f"Background sample rate {sr} != target {self.sample_rate}")
        target_length = int(duration * self.sample_rate)
        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + target_length
        if end_sample <= len(background_audio):
            bg_segment = background_audio[start_sample:end_sample]
        else:
            bg_segment = background_audio[start_sample:]
        result = np.zeros(target_length, dtype=np.float32)
        audio_len = min(len(audio_data), target_length)
        bg_len = min(len(bg_segment), target_length)
        if audio_len > 0:
            result[:audio_len] = audio_data[:audio_len] * self.vocals_volume
        if bg_len > 0:
            result[:bg_len] += bg_segment[:bg_len] * self.background_volume
        return result

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        if len(audio_data) == 0:
            return audio_data
        max_val = np.max(np.abs(audio_data))
        if max_val > self.max_val:
            audio_data = audio_data * (self.max_val / max_val)
        return audio_data

    @handle_errors(logger)
    async def _add_video_segment(self, video_path: str, start_time: float, duration: float, audio_data: np.ndarray, output_path: str, sentences: List[Sentence], task_state) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if len(audio_data) == 0 or duration <= 0:
            raise ValueError("Invalid audio data or duration")
        with ExitStack() as stack:
            temp_video = stack.enter_context(NamedTemporaryFile(suffix='.mp4'))
            temp_audio = stack.enter_context(NamedTemporaryFile(suffix='.wav'))
            end_time = start_time + duration
            await self.ffmpeg_tool.cut_video_track(video_path, temp_video.name, start_time, end_time)
            await asyncio.to_thread(sf.write, temp_audio.name, audio_data, self.sample_rate)

            if task_state.generate_subtitle:
                temp_ass = stack.enter_context(NamedTemporaryFile(suffix='.ass'))
                await asyncio.to_thread(self._generate_subtitles_for_segment, sentences, start_time * 1000, temp_ass.name, task_state.target_language)
                await self.ffmpeg_tool.cut_video_with_subtitles_and_audio(temp_video.name, temp_audio.name, temp_ass.name, output_path)
            else:
                await self.ffmpeg_tool.cut_video_with_audio(temp_video.name, temp_audio.name, output_path)
            logger.debug(f"Video segment added to {output_path}")

    def _generate_subtitles_for_segment(self, sentences: List[Sentence], segment_start_ms: float, output_sub_path: str, target_language: str = "en"):
        subs = pysubs2.SSAFile()
        for s in sentences:
            start_local = s.adjusted_start - segment_start_ms - s.segment_start * 1000
            duration_ms = s.adjusted_duration
            if duration_ms <= 0:
                continue
            blocks = self._split_long_text_to_sub_blocks(s.trans_text, start_local, duration_ms, target_language)
            for block in blocks:
                evt = pysubs2.SSAEvent(start=int(block["start"]), end=int(block["end"]), text=block["text"])
                subs.append(evt)

        style = subs.styles.get("Default", pysubs2.SSAStyle())
        style.fontname = "Arial"
        style.fontsize = 22
        style.bold = True
        style.primarycolor = pysubs2.Color(255, 255, 255, 0)
        style.outlinecolor = pysubs2.Color(0, 0, 0, 100)
        style.borderstyle = 3
        style.shadow = 0
        style.alignment = pysubs2.Alignment.BOTTOM_CENTER
        style.marginv = 20
        subs.styles["Default"] = style
        subs.save(output_sub_path, format="ass")
        logger.debug(f"Subtitles generated at {output_sub_path}")

    def _split_long_text_to_sub_blocks(self, text: str, start_ms: float, duration_ms: float, lang: str = "en") -> List[dict]:
        recommended_max_chars = {"zh": 20, "ja": 20, "ko": 20, "en": 40}
        max_chars = recommended_max_chars.get(lang, 40)
        if len(text) <= max_chars:
            return [{"start": start_ms, "end": start_ms + duration_ms, "text": text}]
        chunks = []
        if lang == "en":
            chunks = self._chunk_english_text(text, max_chars)
        else:
            chunks = self._chunk_cjk_text(text, max_chars)
        sub_blocks = []
        total_chars = sum(len(c) for c in chunks)
        current_start = start_ms
        for c in chunks:
            chunk_dur = duration_ms * (len(c) / total_chars) if total_chars > 0 else 0
            sub_blocks.append({"start": current_start, "end": current_start + chunk_dur, "text": c})
            current_start += chunk_dur
        if sub_blocks:
            sub_blocks[-1]["end"] = start_ms + duration_ms
        else:
            sub_blocks.append({"start": start_ms, "end": start_ms + duration_ms, "text": text})
        return sub_blocks

    def _chunk_english_text(self, text: str, max_chars: int) -> List[str]:
        words = text.split()
        chunks = []
        current_line = []
        for w in words:
            line_len = sum(len(x) for x in current_line) + len(current_line)
            if line_len + len(w) > max_chars:
                if current_line:
                    chunks.append(" ".join(current_line))
                    current_line = []
            current_line.append(w)
        if current_line:
            chunks.append(" ".join(current_line))
        return chunks

    def _chunk_cjk_text(self, text: str, max_chars: int) -> List[str]:
        chunks = []
        total_length = len(text)
        start_idx = 0
        cjk_puncts = set("，,。.!！？?；;：:、…~— ")
        while start_idx < total_length:
            end_idx = start_idx + max_chars
            if end_idx < total_length and text[end_idx] in cjk_puncts:
                end_idx += 1
            end_idx = min(end_idx, total_length)
            chunks.append(text[start_idx:end_idx])
            start_idx = end_idx
        return chunks
