import logging
import asyncio
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from .decorators import handle_errors
from .concurrency import run_sync
from .ffmpeg_utils import FFmpegTool

logger = logging.getLogger(__name__)

class MediaUtils:
    def __init__(self, config, audio_separator, target_sr: int = 24000):
        self.config = config
        self.target_sr = target_sr
        self.logger = logging.getLogger(__name__)
        self.audio_separator = audio_separator
        self.ffmpeg_tool = FFmpegTool()

    def normalize_and_resample(self, audio_input: Union[Tuple[int, np.ndarray], np.ndarray], target_sr: int = None) -> np.ndarray:
        if isinstance(audio_input, tuple):
            fs, audio_input = audio_input
        else:
            fs = target_sr
        audio_input = audio_input.astype(np.float32)
        max_val = np.abs(audio_input).max()
        if max_val > 0:
            audio_input = audio_input / max_val
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=-1)
        if fs != target_sr:
            audio_input = np.ascontiguousarray(audio_input)
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_sr, dtype=torch.float32)
            audio_input = resampler(torch.from_numpy(audio_input)[None, :])[0].numpy()
        return audio_input

    @handle_errors(logger)
    async def get_video_duration(self, video_path: str) -> float:
        return await self.ffmpeg_tool.get_duration(video_path)

    @handle_errors(logger)
    async def get_audio_segments(self, duration: float) -> List[Tuple[float, float]]:
        segment_length = self.config.SEGMENT_MINUTES * 60
        min_length = self.config.MIN_SEGMENT_MINUTES * 60
        if duration <= min_length:
            return [(0, duration)]
        segments = []
        current_pos = 0.0
        while current_pos < duration:
            remaining = duration - current_pos
            if remaining <= segment_length:
                if remaining < min_length and segments:
                    start = segments[-1][0]
                    segments[-1] = (start, duration - start)
                else:
                    segments.append((current_pos, remaining))
                break
            segments.append((current_pos, segment_length))
            current_pos += segment_length
        return segments

    @handle_errors(logger)
    async def extract_segment(self, video_path: str, start: float, duration: float, output_dir: Path, segment_index: int) -> Dict[str, Union[str, float]]:
        temp_files = {}
        try:
            silent_video = str(output_dir / f"video_silent_{segment_index}.mp4")
            full_audio = str(output_dir / f"audio_full_{segment_index}.wav")
            vocals_audio = str(output_dir / f"vocals_{segment_index}.wav")
            background_audio = str(output_dir / f"background_{segment_index}.wav")

            await asyncio.gather(
                self.ffmpeg_tool.extract_audio(video_path, full_audio, start, duration),
                self.ffmpeg_tool.extract_video(video_path, silent_video, start, duration)
            )

            def separate():
                return self.audio_separator.separate_audio(full_audio)
            vocals, background, sr = await run_sync(separate)

            def resample_bg():
                return self.normalize_and_resample((sr, background), self.target_sr)
            background = await run_sync(resample_bg)

            def write_vocals():
                sf.write(vocals_audio, vocals, sr, subtype='FLOAT')
            def write_bg():
                sf.write(background_audio, background, self.target_sr, subtype='FLOAT')
            await run_sync(write_vocals)
            await run_sync(write_bg)

            segment_duration = len(vocals) / sr
            Path(full_audio).unlink(missing_ok=True)
            temp_files = {
                'video': silent_video,
                'vocals': vocals_audio,
                'background': background_audio,
                'duration': segment_duration
            }
            return temp_files
        except Exception as e:
            for f in temp_files.values():
                if isinstance(f, str) and Path(f).exists():
                    Path(f).unlink()
            raise
