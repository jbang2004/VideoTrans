# ---------------------------------------------------
# backend/core/media_mixer.py (精简示例改动)
# ---------------------------------------------------
import numpy as np
import logging
import soundfile as sf
import os
import asyncio
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import List

import pysubs2  # 用于简化字幕处理

from utils.decorators import handle_errors
from utils.ffmpeg_utils import FFmpegTool
from config import Config
from core.sentence_tools import Sentence
from utils.task_state import TaskState

logger = logging.getLogger(__name__)

class MediaMixer:
    def __init__(self, config: Config, sample_rate: int):
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
        task_state: TaskState,
        output_path: str,
        generate_subtitle: bool = False
    ):
        """
        合成并输出带音频的视频片段，可选是否烧制字幕。
        """
        if not sentences:
            logger.warning("mixed_media_maker: 收到空的句子列表")
            return False

        segment_index = sentences[0].segment_index
        segment_files = task_state.segment_media_files.get(segment_index)
        if not segment_files:
            logger.error(f"找不到分段 {segment_index} 的媒体文件")
            return False

        # (1) 拼接全部句子的音频
        full_audio = np.array([], dtype=np.float32)
        for sentence in sentences:
            if sentence.generated_audio is not None:
                audio_data = np.asarray(sentence.generated_audio, dtype=np.float32)
                if len(full_audio) > 0:
                    audio_data = self._apply_fade_effect(audio_data)
                full_audio = np.concatenate((full_audio, audio_data))
            else:
                logger.warning("某句子音频生成失败...")

        if len(full_audio) == 0:
            logger.error("没有有效的音频数据")
            return False

        # 计算当前片段的起始时间和时长(秒)
        start_time = 0.0 if sentences[0].is_first \
                     else (sentences[0].adjusted_start - sentences[0].segment_start * 1000) / 1000.0
        duration = sum(s.adjusted_duration for s in sentences) / 1000.0

        # (2) 混合背景音乐
        background_audio_path = segment_files['background']
        if background_audio_path is not None:
            full_audio = self._mix_with_background(
                background_audio_path, start_time, duration, full_audio
            )
            full_audio = self._normalize_audio(full_audio)

        # (3) 与视频合成
        video_path = segment_files['video']
        if video_path:
            await self._add_video_segment(
                video_path=video_path,
                start_time=start_time,
                duration=duration,
                audio_data=full_audio,
                output_path=output_path,
                sentences=sentences,
                generate_subtitle=generate_subtitle
            )
            return True

        return False

    # ------------------------------
    # 下面方法和你原先类似，只做小幅改动
    # ------------------------------
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
            logger.warning("背景音乐采样率不匹配, 未做重采样, 可能有问题.")

        target_length = int(duration * self.sample_rate)
        start_sample = int(start_time * self.sample_rate)
        end_sample = start_sample + target_length

        if end_sample <= len(background_audio):
            bg_segment = background_audio[start_sample:end_sample]
        else:
            bg_segment = background_audio[start_sample:]

        result = np.zeros(target_length, dtype=np.float32)
        audio_length = min(len(audio_data), target_length)
        bg_length = min(len(bg_segment), target_length)

        if audio_length > 0:
            result[:audio_length] = audio_data[:audio_length] * self.vocals_volume
        if bg_length > 0:
            result[:bg_length] += bg_segment[:bg_length] * self.background_volume

        return result

    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        if len(audio_data) == 0:
            return audio_data
        max_val = np.abs(audio_data).max()
        if max_val > self.max_val:
            audio_data *= (self.max_val / max_val)
        return audio_data

    @handle_errors(logger)
    async def _add_video_segment(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        audio_data: np.ndarray,
        output_path: str,
        sentences: List[Sentence],
        generate_subtitle: bool = False
    ):
        """
        截取 [start_time, start_time+duration] 的视频段，与音频合并。
        若 generate_subtitle=True，则生成并“烧制” .ass 字幕(使用 ffmpeg_utils)。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        if len(audio_data) == 0 or duration <= 0:
            raise ValueError("音频或持续时间无效")

        with ExitStack() as stack:
            temp_video = stack.enter_context(NamedTemporaryFile(suffix='.mp4'))
            temp_audio = stack.enter_context(NamedTemporaryFile(suffix='.wav'))

            end_time = start_time + duration

            # (1) 截取无声视频
            await self.ffmpeg_tool.cut_video_track(
                input_path=video_path,
                output_path=temp_video.name,
                start=start_time,
                end=end_time
            )

            # (2) 将合成音频写入 wav
            await asyncio.to_thread(sf.write, temp_audio.name, audio_data, self.sample_rate)

            # (3) 是否要烧制字幕
            if generate_subtitle:
                temp_ass = stack.enter_context(NamedTemporaryFile(suffix='.ass'))
                # 生成带“美观”样式的 .ass
                await asyncio.to_thread(self._generate_subtitles_for_segment,
                    sentences,
                    segment_start_ms=start_time * 1000.0,
                    output_sub_path=temp_ass.name
                )

                # 调用 ffmpeg 工具合并(不再传任何 force_style, 直接用 .ass 里的样式)
                await self.ffmpeg_tool.cut_video_with_subtitles_and_audio(
                    input_video_path=temp_video.name,
                    input_audio_path=temp_audio.name,
                    subtitles_path=temp_ass.name,
                    output_path=output_path
                )
            else:
                # 仅合并音视频
                await self.ffmpeg_tool.cut_video_with_audio(
                    input_video_path=temp_video.name,
                    input_audio_path=temp_audio.name,
                    output_path=output_path
                )

    # --------------------------------------
    # 这里重点给出 "YouTube风格" 的示例
    # --------------------------------------
    def _generate_subtitles_for_segment(
        self,
        sentences: List[Sentence],
        segment_start_ms: float,
        output_sub_path: str
    ):
        """
        使用 pysubs2 生成 .ass 字幕, 并写入 YouTube-like 样式(不被 FFmpeg 覆盖).
        包含长句拆分、底部居中、白字黑边等效果.
        """
        subs = pysubs2.SSAFile()

        # 遍历句子 -> 生成 event
        for s in sentences:
            start_local = s.adjusted_start - segment_start_ms
            end_local = (s.adjusted_start + s.adjusted_duration) - segment_start_ms
            if end_local <= 0:
                continue

            raw_text = s.trans_text or s.raw_text
            if not raw_text.strip():
                continue

            duration_ms = end_local - start_local
            if duration_ms <= 0:
                continue

            # 若需要: 拆分长句子
            blocks = self._split_long_text_to_sub_blocks(
                text=raw_text.strip(),
                start_ms=start_local,
                duration_ms=duration_ms,
                max_chars=30  # 例: 一行30字, 过长再切分(可调)
            )

            for block in blocks:
                evt = pysubs2.SSAEvent(
                    start=int(block["start"]),
                    end=int(block["end"]),
                    text=block["text"]
                )
                subs.append(evt)

        # 让 "Default" 样式更像 YouTube
        # pysubs2 默认会有一个 styles["Default"]，
        # 如果没有，也可以 new 一个 style 再 subs.styles["Default"] = style
        style = subs.styles.get("Default", pysubs2.SSAStyle())

        # 下面是一些常见设置:
        style.fontname = "Arial"                  # YouTube字幕常见无衬线字体
        style.fontsize = 28                       # 字号稍大些
        style.primarycolor = pysubs2.Color(255, 255, 255, 0)  # (r,g,b,a=0) => 白色无透明
        style.outlinecolor = pysubs2.Color(0, 0, 0, 0)        # 黑色描边
        style.bold = False
        style.italic = False
        style.underline = False

        style.borderstyle = 1  # 1: 普通描边&阴影  3: 有背景框
        style.outline = 2      # 描边宽度
        style.shadow = 0       # 阴影 (0=无)
        style.alignment = pysubs2.Alignment.BOTTOM_CENTER  # 底部居中
        style.margin_v = 40    # 离底部距离(像素)

        subs.styles["Default"] = style

        # 保存为 .ass
        subs.save(output_sub_path, format="ass")
        logger.debug(f"已写入美观字幕: {output_sub_path}")

    def _split_long_text_to_sub_blocks(
        self,
        text: str,
        start_ms: float,
        duration_ms: float,
        max_chars: int = 30
    ):
        """
        将过长文本在[ start_ms, start_ms+duration_ms ]区间内拆分，
        按字符数比例分配每段的展示时长。
        """
        length = len(text)
        if length <= max_chars:
            return [{
                "start": start_ms,
                "end":   start_ms + duration_ms,
                "text":  text
            }]

        # 拆分
        chunks = []
        idx = 0
        while idx < length:
            chunk_text = text[idx : idx + max_chars]
            chunks.append(chunk_text)
            idx += max_chars

        sub_blocks = []
        total_chars = length
        current_start = start_ms
        for chunk_text in chunks:
            c_len = len(chunk_text)
            chunk_dur = duration_ms * (c_len / total_chars)
            block_start = current_start
            block_end = current_start + chunk_dur

            sub_blocks.append({
                "start": block_start,
                "end":   block_end,
                "text":  chunk_text
            })
            current_start += chunk_dur

        # 修正最后一块结束时间
        sub_blocks[-1]["end"] = start_ms + duration_ms
        return sub_blocks

    async def reset(self):
        self.full_audio_buffer = np.array([], dtype=np.float32)
        logger.debug("mixer 状态已重置")
