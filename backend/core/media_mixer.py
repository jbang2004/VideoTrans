# ---------------------------------------------------
# backend/core/media_mixer.py (完整可复制版本)
# ---------------------------------------------------
import numpy as np
import logging
import soundfile as sf
import os
import asyncio
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import Optional, List
from utils.decorators import handle_errors

from utils.ffmpeg_utils import FFmpegTool
from config import Config
from core.sentence_tools import Sentence
from utils.task_state import TaskState

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

        self.ffmpeg_tool = FFmpegTool()

    @handle_errors(logger)
    async def mixed_media_maker(
        self,
        sentences: List[Sentence],
        task_state: TaskState = None,
        output_path=None,
        # =========== (新增) ===========
        generate_subtitle: bool = False
    ):
        """
        处理一批句子的音频和视频，并最终生成一个带音频的分段 MP4。
        如果 generate_subtitle=True, 则在输出视频中烧制字幕。
        """
        if not sentences:
            logger.warning("mixed_media_maker: 收到空的句子列表")
            return False

        full_audio = np.array([], dtype=np.float32)

        segment_index = sentences[0].segment_index
        segment_files = task_state.segment_media_files.get(segment_index)
        if not segment_files:
            logger.error(f"找不到分段 {segment_index} 的媒体文件")
            return False

        # 拼接所有句子的合成音频
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

        # 计算分段内的起始时间 & 本批时长
        # 如果是首段，则从0开始，否则相对于已计算的 (adjusted_start - segment_start * 1000)
        start_time = 0.0 if sentences[0].is_first else (sentences[0].adjusted_start - sentences[0].segment_start * 1000) / 1000.0
        duration = sum(s.adjusted_duration for s in sentences) / 1000.0

        background_audio_path = segment_files['background']
        if background_audio_path is not None:
            full_audio = self._mix_with_background(background_audio_path, start_time, duration, full_audio)
            full_audio = self._normalize_audio(full_audio)

        self.full_audio_buffer = np.concatenate((self.full_audio_buffer, full_audio))

        video_path = segment_files['video']
        if video_path:
            await self._add_video_segment(
                video_path=video_path,
                start_time=start_time,
                duration=duration,
                audio_data=full_audio,
                output_path=output_path,
                # =========== (新增) ===========
                sentences=sentences,
                generate_subtitle=generate_subtitle
            )
            return True

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

    def _mix_with_background(self, background_audio_path: str, start_time: float, duration: float, audio_data: np.ndarray) -> np.ndarray:
        import soundfile as sf
        background_audio, sr = sf.read(background_audio_path)
        background_audio = np.asarray(background_audio, dtype=np.float32)

        target_length = int(duration * self.sample_rate)
        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + target_length
        if end_sample <= len(background_audio):
            background_segment = background_audio[start_sample:end_sample]
        else:
            background_segment = background_audio[start_sample:]

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
    async def _add_video_segment(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        audio_data: np.ndarray,
        output_path: str,
        # =========== (新增) ===========
        sentences: List[Sentence],
        generate_subtitle: bool = False
    ) -> None:
        """
        切割出指定时段的视频片段，并与合成音频合并输出。
        若 generate_subtitle=True，则额外烧制字幕。
        """
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

            # 1) 截取无声视频 (仅取指定时段)
            await self.ffmpeg_tool.cut_video_track(
                input_path=video_path,
                output_path=temp_video.name,
                start=start_time,
                end=end_time
            )

            # 2) 将合成音频写入临时 wav
            await asyncio.to_thread(sf.write, temp_audio.name, audio_data, self.sample_rate)

            # =========== (新增) ===========
            if generate_subtitle:
                # a) 生成与本片段匹配的 SRT 文本
                srt_content = self._generate_srt_for_segment(sentences, segment_start_ms=start_time * 1000.0)
                temp_srt = stack.enter_context(NamedTemporaryFile(suffix='.srt'))
                await asyncio.to_thread(lambda: open(temp_srt.name, 'w', encoding='utf-8').write(srt_content))

                # b) 用新的 FFmpeg 命令把字幕烧进画面
                await self.ffmpeg_tool.cut_video_with_subtitles_and_audio(
                    input_video_path=temp_video.name,
                    input_audio_path=temp_audio.name,
                    subtitles_path=temp_srt.name,
                    output_path=output_path
                )
            else:
                # 原逻辑：快速合并 (copy video + aac audio)
                await self.ffmpeg_tool.cut_video_with_audio(
                    input_video_path=temp_video.name,
                    input_audio_path=temp_audio.name,
                    output_path=output_path
                )

    # =========== (新增) ===========
    def _generate_srt_for_segment(self, sentences: List[Sentence], segment_start_ms: float) -> str:
        """
        针对本段 sentences，生成局部SRT文本。
        segment_start_ms: 该段在整部视频中的起始毫秒数，用于将 sentence.adjusted_start
                          转成相对本段的时间(从0开始)
        """
        def ms_to_srt_time(ms: float) -> str:
            # SRT格式: HH:MM:SS,mmm
            hours = int(ms // 3600000)
            minutes = int((ms % 3600000) // 60000)
            seconds = int((ms % 60000) // 1000)
            millis = int(ms % 1000)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

        lines = []
        index = 1
        for s in sentences:
            start_ms_global = s.adjusted_start
            end_ms_global = s.adjusted_start + s.adjusted_duration

            start_local = start_ms_global - segment_start_ms
            end_local = end_ms_global - segment_start_ms

            if start_local < 0:
                # 说明这句的开始在当前片段之前
                continue
            if start_local > (end_local):
                # 不合理
                continue

            start_str = ms_to_srt_time(start_local)
            end_str = ms_to_srt_time(end_local)

            text_line = s.trans_text or s.raw_text
            if not text_line:
                continue

            lines.append(f"{index}")
            lines.append(f"{start_str} --> {end_str}")
            lines.append(text_line)
            lines.append("")  # 空行
            index += 1

        return "\n".join(lines)

    async def reset(self):
        self.full_audio_buffer = np.array([], dtype=np.float32)
        logger.debug("已重置 mixer 状态")
