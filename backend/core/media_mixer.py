import numpy as np
import logging
import soundfile as sf
import os
import shutil
import asyncio
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Optional, List
from utils.decorators import handle_errors

logger = logging.getLogger(__name__)

class MediaMixer:
    def __init__(self, config, sample_rate: int):
        self.config = config
        self.sample_rate = sample_rate
        self.max_val = 1.0
        self.overlap = self.config.AUDIO_OVERLAP
        self.vocals_volume = self.config.VOCALS_VOLUME
        self.background_volume = self.config.BACKGROUND_VOLUME
        self.full_audio_buffer = np.array([], dtype=np.float32)

    @handle_errors(logger)
    async def mixed_media_maker(self, sentences, task_state=None, output_path=None):
        """
        处理一批句子的音频和视频
        """
        if not sentences:
            logger.warning("接收到空的句子列表")
            return False

        full_audio = np.array([], dtype=np.float32)

        segment_index = sentences[0].segment_index
        segment_files = task_state.segment_media_files.get(segment_index)
        if not segment_files:
            logger.error(f"找不到分段 {segment_index} 的媒体文件")
            return False

        for sentence in sentences:
            if sentence.generated_audio is not None:
                audio_data = np.asarray(sentence.generated_audio, dtype=np.float32)
                if len(full_audio) > 0:
                    audio_data = self._apply_fade_effect(audio_data)
                full_audio = np.concatenate((full_audio, audio_data))
            else:
                logger.warning(
                    f"句子音频生成失败: '{sentence.raw_text[:30]}...', "
                    f"UUID: {sentence.model_input.get('uuid', 'unknown')}"
                )

        if len(full_audio) == 0:
            logger.error("没有有效的音频数据")
            return False

        start_time = 0.0 if sentences[0].is_first else (sentences[0].adjusted_start - sentences[0].segment_start * 1000) / 1000.0
        duration = sum(s.adjusted_duration for s in sentences) / 1000.0

        background_audio_path = segment_files['background']
        if background_audio_path is not None:
            full_audio = self._mix_with_background(background_audio_path, start_time, duration, full_audio)
            full_audio = self._normalize_audio(full_audio)

        self.full_audio_buffer = np.concatenate((self.full_audio_buffer, full_audio))

        video_path = segment_files['video']
        if video_path:
            await self._add_video_segment(video_path, start_time, duration, full_audio, output_path)
            return True

        return False

    def _apply_fade_effect(self, audio_data: np.ndarray) -> np.ndarray:
        """应用淡入淡出效果，自然处理重叠"""
        if audio_data is None or len(audio_data) == 0:
            return np.array([], dtype=np.float32)
        
        if len(audio_data) > self.overlap * 2:
            audio_data = audio_data.copy()
            fade_in = np.linspace(0, 1, self.overlap)
            fade_out = np.linspace(1, 0, self.overlap)
            
            audio_data[:self.overlap] *= fade_in
            audio_data[-self.overlap:] *= fade_out
            
            if len(self.full_audio_buffer) > 0:
                overlap_region = self.full_audio_buffer[-self.overlap:]
                audio_data[:self.overlap] = np.add(
                    overlap_region,
                    audio_data[:self.overlap],
                    dtype=np.float32
                )
        return audio_data

    def _mix_with_background(self, background_audio_path: str, start_time: float, duration: float, audio_data: np.ndarray) -> np.ndarray:
        background_audio, _ = sf.read(background_audio_path)
        background_audio = np.asarray(background_audio, dtype=np.float32)
        
        target_length = int(duration * self.sample_rate)
        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + target_length
        background_segment = background_audio[start_sample:end_sample] if end_sample <= len(background_audio) else background_audio[start_sample:]
        
        result = np.zeros(target_length, dtype=np.float32)
        audio_length = min(len(audio_data), target_length)
        background_length = min(len(background_segment), target_length)
        
        if audio_length > 0:
            result[:audio_length] = audio_data[:audio_length] * self.vocals_volume
        if background_length > 0:
            result[:background_length] += background_segment[:background_length] * self.background_volume
        
        return result

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        if audio_data is None or len(audio_data) == 0:
            return np.array([], dtype=np.float32)
        
        max_val = np.abs(audio_data).max()
        if max_val > self.max_val:
            audio_data = audio_data * (self.max_val / max_val)
        return audio_data

    @handle_errors(logger)
    async def _add_video_segment(self, video_path: str, start_time: float, duration: float, audio_data: np.ndarray, output_path: str) -> None:
        """添加视频片段"""
        if not os.path.exists(video_path):
            logger.error("视频文件不存在")
            raise FileNotFoundError("视频文件不存在")
        
        if audio_data is None or len(audio_data) == 0:
            logger.error("无有效音频数据")
            raise ValueError("无有效音频数据")
        
        if duration <= 0:
            logger.error("无效的持续时间")
            raise ValueError("无效的持续时间")

        with ExitStack() as stack:
            temp_video = stack.enter_context(NamedTemporaryFile(suffix='.mp4'))
            temp_audio = stack.enter_context(NamedTemporaryFile(suffix='.wav'))

            end_time = start_time + duration
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c:v', 'libx264',
                '-preset', 'superfast',
                '-an',
                '-vsync', 'vfr',
                temp_video.name
            ]
            await self._run_ffmpeg_command(cmd)

            await asyncio.to_thread(sf.write, temp_audio.name, audio_data, self.sample_rate)

            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video.name,
                '-i', temp_audio.name,
                '-c:v', 'copy',
                '-c:a', 'aac',
                output_path
            ]
            await self._run_ffmpeg_command(cmd)

    @handle_errors(logger)
    async def _run_ffmpeg_command(self, command: List[str]) -> None:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg 命令执行失败: {stderr.decode()}")

    async def reset(self):
        self.full_audio_buffer = np.array([], dtype=np.float32)
        logger.debug("已重置 mixer 状态")