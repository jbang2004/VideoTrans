# ---------------------------------------------------
# backend/core/media_mixer.py  (改进版，完整可复制)
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

import pysubs2  # 新增: 用于简化字幕处理

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
        task_state: TaskState = None,
        output_path: str = None,
        # 新增: 是否要生成（烧制）字幕
        generate_subtitle: bool = False
    ):
        """
        处理一批句子的音频和视频，并最终生成一个带音频的分段 MP4。
        如果 generate_subtitle=True, 则在输出视频中"烧制"字幕。
        """
        if not sentences:
            logger.warning("mixed_media_maker: 收到空的句子列表")
            return False

        segment_index = sentences[0].segment_index
        segment_files = task_state.segment_media_files.get(segment_index)
        if not segment_files:
            logger.error(f"找不到分段 {segment_index} 的媒体文件")
            return False

        # 拼接所有句子的合成音频
        full_audio = np.array([], dtype=np.float32)
        for sentence in sentences:
            if sentence.generated_audio is not None:
                audio_data = np.asarray(sentence.generated_audio, dtype=np.float32)
                if len(full_audio) > 0:
                    # 和上一片段做淡入淡出衔接
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

        # 计算本段的起始时间和总时长(秒)
        # 如果是首段，则从0开始，否则相对于已计算的 (adjusted_start - segment_start * 1000)
        start_time = 0.0 if sentences[0].is_first else (sentences[0].adjusted_start - sentences[0].segment_start * 1000) / 1000.0
        duration = sum(s.adjusted_duration for s in sentences) / 1000.0

        # 背景音混合
        background_audio_path = segment_files['background']
        if background_audio_path is not None:
            full_audio = self._mix_with_background(
                background_audio_path=background_audio_path,
                start_time=start_time,
                duration=duration,
                audio_data=full_audio
            )
            full_audio = self._normalize_audio(full_audio)

        # 保存到 mixer 缓存(可选)
        self.full_audio_buffer = np.concatenate((self.full_audio_buffer, full_audio))

        video_path = segment_files['video']
        if video_path:
            # 与视频合成
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

    def _apply_fade_effect(self, audio_data: np.ndarray) -> np.ndarray:
        """
        在语音衔接处做一定长度(overlap)的淡入淡出。
        """
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
        """
        将合成好的人声与背景音乐混合。
        注意只截取背景音乐指定区间 [start_time, start_time+duration]。
        """
        background_audio, sr = sf.read(background_audio_path)
        background_audio = np.asarray(background_audio, dtype=np.float32)
        if sr != self.sample_rate:
            logger.warning(f"背景音乐采样率({sr})与目标采样率({self.sample_rate})不一致，"
                           "此处示例未做重采样，需要可自行添加重采样操作。")

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
        """
        简单归一化，确保音量不超过 self.max_val
        """
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
        sentences: List[Sentence],
        generate_subtitle: bool = False
    ) -> None:
        """
        从整段视频中切割出 [start_time, start_time+duration] 的片段，
        然后与合成好的音频混合，并可选将字幕"烧制"进去。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        if audio_data is None or len(audio_data) == 0:
            raise ValueError("无有效音频数据")

        if duration <= 0:
            raise ValueError("无效的持续时间: duration <= 0")

        with ExitStack() as stack:
            temp_video = stack.enter_context(NamedTemporaryFile(suffix='.mp4'))
            temp_audio = stack.enter_context(NamedTemporaryFile(suffix='.wav'))

            end_time = start_time + duration

            # (1) 先用 ffmpeg_tool 把 [start_time, end_time] 这段无声视频截出来
            await self.ffmpeg_tool.cut_video_track(
                input_path=video_path,
                output_path=temp_video.name,
                start=start_time,
                end=end_time
            )

            # (2) 将合成音频写入临时 wav
            await asyncio.to_thread(sf.write, temp_audio.name, audio_data, self.sample_rate)

            # (3) 是否需要烧制字幕？
            if generate_subtitle:
                # 生成当前片段对应的字幕文件(ASS 或 SRT)。这里我们生成ASS来获得更美观的效果
                temp_sub = stack.enter_context(NamedTemporaryFile(suffix='.ass'))
                await asyncio.to_thread(
                    self._generate_subtitles_for_segment,
                    sentences,
                    segment_start_ms=start_time * 1000.0,
                    output_sub_path=temp_sub.name,
                    subtitle_format="ass",   # 也可 "srt" 看需求
                    max_chars_per_line=25    # 每行最多多少字符，再拆分
                )

                # 用 FFmpegTool 带上 subtitles 的 filter 烧制到视频
                await self.ffmpeg_tool.cut_video_with_subtitles_and_audio(
                    input_video_path=temp_video.name,
                    input_audio_path=temp_audio.name,
                    subtitles_path=temp_sub.name,
                    output_path=output_path,
                    font_size=28,
                    font_color="white",
                    font_outline=2,
                    position="bottom"  # 你可改成 "center" / "top"
                )
            else:
                # 仅合并音视频，不加字幕
                await self.ffmpeg_tool.cut_video_with_audio(
                    input_video_path=temp_video.name,
                    input_audio_path=temp_audio.name,
                    output_path=output_path
                )

    def _generate_subtitles_for_segment(
        self,
        sentences: List[Sentence],
        segment_start_ms: float,
        output_sub_path: str,
        subtitle_format: str = "ass",
        max_chars_per_line: int = 25
    ):
        """
        使用 pysubs2 生成/写出字幕文件 (SRT/ASS)。
        在这里做"长句拆分"、"多行换行"等操作，提升美观度。

        Args:
            sentences: 本视频片段相关的 Sentence 列表
            segment_start_ms: 该片段在全局视频中的开始毫秒 (用来计算相对时间)
            output_sub_path: 要输出的字幕文件路径
            subtitle_format: "srt" 或 "ass"
            max_chars_per_line: 当一句文本超过此字符数时会自动拆分(分时段 sequential)
        """
        # 新建一个字幕容器
        subs = pysubs2.SSAFile()

        for s in sentences:
            # 计算在当前段内的相对时间(毫秒)
            start_local = s.adjusted_start - segment_start_ms
            end_local = (s.adjusted_start + s.adjusted_duration) - segment_start_ms

            # 如果这整句都在 segment 开始之前结束了，则跳过
            if end_local <= 0:
                continue

            # 如果时长无效，跳过
            duration_local = end_local - start_local
            if duration_local <= 0:
                continue

            raw_text = s.trans_text or s.raw_text
            if not raw_text.strip():
                continue

            # 拆分：将过长句子分段 (顺序出现)
            blocks = self._split_long_text_to_sub_blocks(
                raw_text.strip(),
                start_ms=start_local,
                duration_ms=duration_local,
                max_chars=max_chars_per_line
            )

            # 逐块生成 pysubs2 的字幕事件
            for block in blocks:
                evt = pysubs2.SSAEvent(
                    start=int(block["start"]),   # pysubs2 用毫秒
                    end=int(block["end"]),
                    text=block["text"]           # 最终要显示的行
                )
                subs.append(evt)

        # 给 ASS 做默认样式(若你选择存成 .srt，则这里可以忽略样式)
        if subtitle_format.lower() in ("ass", "ssa"):
            if "Default" in subs.styles:
                # 自定义一些更美观的默认样式
                subs.styles["Default"].fontsize = 28
                subs.styles["Default"].outline = 2
                subs.styles["Default"].shadow = 0
                subs.styles["Default"].alignment = pysubs2.Alignment.BOTTOM_CENTER
            else:
                # 如果没有默认样式，创建一个
                style = pysubs2.SSAStyle()
                style.fontsize = 28
                style.outline = 2
                style.shadow = 0
                style.alignment = pysubs2.Alignment.BOTTOM_CENTER
                subs.styles["Default"] = style

        # 按需输出
        subs.save(output_sub_path, format=subtitle_format)
        logger.debug(f"字幕已生成: {output_sub_path}")

    def _split_long_text_to_sub_blocks(
        self,
        text: str,
        start_ms: float,
        duration_ms: float,
        max_chars: int
    ):
        """
        将长文本在 [start_ms, start_ms+duration_ms] 时间范围内，按照 max_chars 拆分成多段。
        每段依次出现，按字符长度平均分配时间。

        返回结构:
        [
           {"start": float, "end": float, "text": str},
           ...
        ]
        """
        text_length = len(text)
        if text_length <= max_chars:
            # 不需要拆分
            return [{
                "start": start_ms,
                "end":   start_ms + duration_ms,
                "text":  text
            }]

        # 1) 字符分段
        chunks = []
        idx = 0
        while idx < text_length:
            chunk_text = text[idx : idx + max_chars]
            chunks.append(chunk_text)
            idx += max_chars

        # 2) 依字符数分配时间
        sub_blocks = []
        total_chars = text_length
        current_start = start_ms
        for i, chunk_text in enumerate(chunks):
            c_len = len(chunk_text)
            chunk_dur = duration_ms * (c_len / total_chars)  # 按比例
            block_start = current_start
            block_end   = current_start + chunk_dur

            sub_blocks.append({
                "start": block_start,
                "end":   block_end,
                "text":  chunk_text
            })
            current_start += chunk_dur

        # 最后一块兜底
        sub_blocks[-1]["end"] = start_ms + duration_ms
        return sub_blocks

    async def reset(self):
        self.full_audio_buffer = np.array([], dtype=np.float32)
        logger.debug("已重置 mixer 状态")
