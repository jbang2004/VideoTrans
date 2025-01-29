# ---------------------------------------------------
# backend/core/media_mixer.py (最终改进版)
# ---------------------------------------------------
import numpy as np
import logging
import soundfile as sf
import os
import asyncio
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import List
import shutil

import pysubs2  # 用于简化字幕处理

from utils.decorators import handle_errors
from utils.ffmpeg_utils import FFmpegTool
from config import Config
from core.sentence_tools import Sentence
from utils.task_state import TaskState

logger = logging.getLogger(__name__)

class MediaMixer:
    """
    用于将多段句子的合成音频与原视频片段混合，并可生成带字幕的视频。
    支持:
      - 音频淡入淡出
      - 背景音乐混合
      - 基于 pysubs2 生成 .ass 字幕(“YouTube风格”)
      - 按语言自动决定单行最大长度
    """
    def __init__(self, config: Config, sample_rate: int):
        self.config = config
        self.sample_rate = sample_rate

        # 音量相关
        self.max_val = 1.0
        self.overlap = self.config.AUDIO_OVERLAP
        self.vocals_volume = self.config.VOCALS_VOLUME
        self.background_volume = self.config.BACKGROUND_VOLUME

        # 全局缓存, 可按需使用
        self.full_audio_buffer = np.array([], dtype=np.float32)

        self.ffmpeg_tool = FFmpegTool()

    @handle_errors(logger)
    async def mixed_media_maker(
        self,
        sentences: List[Sentence],
        task_state: TaskState,
        output_path: str,
        generate_subtitle: bool = False
    ) -> bool:
        """
        主入口: 处理一批句子的音频与视频，输出一段带音频的 MP4。
        根据 generate_subtitle 决定是否烧制字幕。

        Args:
            sentences: 本片段内的所有句子对象
            task_state: 任务状态，内部包含 target_language 等
            output_path: 生成的 MP4 文件路径
            generate_subtitle: 是否在最终视频里烧制字幕

        Returns:
            True / False 表示成功或失败
        """
        if not sentences:
            logger.warning("mixed_media_maker: 收到空的句子列表")
            return False

        segment_index = sentences[0].segment_index
        segment_files = task_state.segment_media_files.get(segment_index)
        if not segment_files:
            logger.error(f"找不到分段 {segment_index} 对应的媒体文件信息")
            return False

        # =========== (1) 拼接所有句子的合成音频 =============
        full_audio = np.array([], dtype=np.float32)
        for sentence in sentences:
            if sentence.generated_audio is not None:
                audio_data = np.asarray(sentence.generated_audio, dtype=np.float32)
                # 如果已经有前面累积的音频，做淡入淡出衔接
                if len(full_audio) > 0:
                    audio_data = self._apply_fade_effect(audio_data)
                full_audio = np.concatenate((full_audio, audio_data))
            else:
                logger.warning(
                    "句子音频生成失败: text=%r, UUID=%s",
                    sentence.raw_text,  # 或 sentence.trans_text
                    sentence.model_input.get("uuid", "unknown")
                )

        if len(full_audio) == 0:
            logger.error("mixed_media_maker: 没有有效的合成音频数据")
            return False

        # 计算当前片段的起始时间和时长(秒)
        start_time = 0.0
        if not sentences[0].is_first:
            start_time = (sentences[0].adjusted_start - sentences[0].segment_start * 1000) / 1000.0

        duration = sum(s.adjusted_duration for s in sentences) / 1000.0

        # =========== (2) 背景音乐混合 (可选) =============
        background_audio_path = segment_files['background']
        if background_audio_path is not None:
            full_audio = self._mix_with_background(
                bg_path=background_audio_path,
                start_time=start_time,
                duration=duration,
                audio_data=full_audio
            )
            full_audio = self._normalize_audio(full_audio)

        # (可按需储存到全局 mixer 缓存)
        self.full_audio_buffer = np.concatenate((self.full_audio_buffer, full_audio))

        # =========== (3) 如果有视频，就把音频合并到视频里 ============
        video_path = segment_files['video']
        if video_path:
            await self._add_video_segment(
                video_path=video_path,
                start_time=start_time,
                duration=duration,
                audio_data=full_audio,
                output_path=output_path,
                sentences=sentences,
                generate_subtitle=generate_subtitle,
                task_state=task_state  # 传入以获取 target_language
            )
            return True

        logger.warning("mixed_media_maker: 本片段无video_path可用")
        return False

    # -------------------------------------------------------------------------
    # 辅助函数: 做音频淡入淡出
    # -------------------------------------------------------------------------
    def _apply_fade_effect(self, audio_data: np.ndarray) -> np.ndarray:
        """
        在语音片段衔接处做 overlap 长度的淡入淡出衔接。
        """
        if audio_data is None or len(audio_data) == 0:
            return np.array([], dtype=np.float32)

        cross_len = min(self.overlap, len(self.full_audio_buffer), len(audio_data))
        if cross_len <= 0:
            return audio_data

        fade_out = np.sqrt(np.linspace(1.0, 0.0, cross_len, dtype=np.float32))
        fade_in  = np.sqrt(np.linspace(0.0, 1.0, cross_len, dtype=np.float32))

        audio_data = audio_data.copy()
        overlap_region = self.full_audio_buffer[-cross_len:]

        audio_data[:cross_len] = overlap_region * fade_out + audio_data[:cross_len] * fade_in
        return audio_data

    # -------------------------------------------------------------------------
    # 辅助函数: 将合成音频与背景音乐混合
    # -------------------------------------------------------------------------
    def _mix_with_background(
        self,
        bg_path: str,
        start_time: float,
        duration: float,
        audio_data: np.ndarray
    ) -> np.ndarray:
        """
        从 bg_path 读取背景音乐，在 [start_time, start_time+duration] 区间截取，
        与 audio_data (人声) 混合。
        """
        background_audio, sr = sf.read(bg_path)
        background_audio = np.asarray(background_audio, dtype=np.float32)
        if sr != self.sample_rate:
            logger.warning(
                f"背景音采样率={sr} 与目标={self.sample_rate}不匹配, 未做重采样, 可能有问题."
            )

        target_length = int(duration * self.sample_rate)
        start_sample = int(start_time * self.sample_rate)
        end_sample   = start_sample + target_length

        if end_sample <= len(background_audio):
            bg_segment = background_audio[start_sample:end_sample]
        else:
            bg_segment = background_audio[start_sample:]

        result = np.zeros(target_length, dtype=np.float32)
        audio_len = min(len(audio_data), target_length)
        bg_len    = min(len(bg_segment), target_length)

        # 混合人声 & 背景
        if audio_len > 0:
            result[:audio_len] = audio_data[:audio_len] * self.vocals_volume
        if bg_len > 0:
            result[:bg_len] += bg_segment[:bg_len] * self.background_volume

        return result

    # -------------------------------------------------------------------------
    # 辅助函数: 对音频做简单归一化
    # -------------------------------------------------------------------------
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        if len(audio_data) == 0:
            return audio_data
        max_val = np.max(np.abs(audio_data))
        if max_val > self.max_val:
            audio_data = audio_data * (self.max_val / max_val)
        return audio_data

    # -------------------------------------------------------------------------
    # 核心函数: 将 audio_data 与截取后的视频段合并
    # -------------------------------------------------------------------------
    @handle_errors(logger)
    async def _add_video_segment(
        self,
        video_path: str,
        start_time: float,
        duration: float,
        audio_data: np.ndarray,
        output_path: str,
        sentences: List[Sentence],
        generate_subtitle: bool,
        task_state: TaskState
    ):
        """
        从原视频里截取 [start_time, start_time + duration] 的视频段(无声)，
        与合成音频合并。
        若 generate_subtitle=True, 则生成 .ass 字幕并在 ffmpeg 工具中进行"烧制"。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"_add_video_segment: 视频文件不存在: {video_path}")
        if len(audio_data) == 0:
            raise ValueError("_add_video_segment: 无音频数据")
        if duration <= 0:
            raise ValueError("_add_video_segment: 无效时长 <=0")

        with ExitStack() as stack:
            temp_video = stack.enter_context(NamedTemporaryFile(suffix='.mp4'))
            temp_audio = stack.enter_context(NamedTemporaryFile(suffix='.wav'))

            end_time = start_time + duration

            # 1) 截取视频 (无音轨)
            await self.ffmpeg_tool.cut_video_track(
                input_path=video_path,
                output_path=temp_video.name,
                start=start_time,
                end=end_time
            )

            # 2) 写合成音频到临时文件
            await asyncio.to_thread(sf.write, temp_audio.name, audio_data, self.sample_rate)

            # 3) 如果需要字幕，则构建 .ass 并用 ffmpeg "烧"进去
            if generate_subtitle:
                temp_ass = stack.enter_context(NamedTemporaryFile(suffix='.ass'))
                # 调用生成字幕的函数
                # (这里把 task_state.target_language 传给它)
                await asyncio.to_thread(
                    self._generate_subtitles_for_segment,
                    sentences,
                    start_time * 1000,   # segment_start_ms
                    temp_ass.name,
                    task_state.target_language  # 新增：把目标语言传进去
                )

                # 保存字幕文件的副本到本地
                subtitle_output_path = output_path.rsplit('.', 1)[0] + '.ass'
                shutil.copy2(temp_ass.name, subtitle_output_path)
                logger.info(f"字幕文件已保存到: {subtitle_output_path}")

                # 生成带字幕的视频
                await self.ffmpeg_tool.cut_video_with_subtitles_and_audio(
                    input_video_path=temp_video.name,
                    input_audio_path=temp_audio.name,
                    subtitles_path=temp_ass.name,
                    output_path=output_path
                )
            else:
                # 不加字幕，仅合并音频
                await self.ffmpeg_tool.cut_video_with_audio(
                    input_video_path=temp_video.name,
                    input_audio_path=temp_audio.name,
                    output_path=output_path
                )

    # -------------------------------------------------------------------------
    # 核心字幕生成函数: 生成 .ass, 带"YouTube风格" + 多语言拆分
    # -------------------------------------------------------------------------
    def _generate_subtitles_for_segment(
        self,
        sentences: List[Sentence],
        segment_start_ms: float,
        output_sub_path: str,
        target_language: str = "en"
    ):
        """
        使用 pysubs2 生成 ASS 字幕文件.
        1. 遍历每条 Sentence, 将其文本拆分成若干块(避免过长字幕).
        2. 向 pysubs2 中写入事件, 并设置"类YouTube"的默认样式.
        3. 最后保存为 .ass 文件.

        Args:
            sentences: 本段的句子列表
            segment_start_ms: 当前段在全局中起始毫秒 (用来计算相对时间)
            output_sub_path: 存放字幕的 .ass 路径
            target_language: 用来确定拆分逻辑(中文/英文/日文/韩文)
        """
        subs = pysubs2.SSAFile()

        for s in sentences:
            # 计算相对时间
            start_local = s.adjusted_start - segment_start_ms - s.segment_start * 1000

            sub_text = (s.trans_text or s.raw_text or "").strip()
            if not sub_text:
                continue

            # 直接使用adjusted_duration作为duration_ms
            duration_ms = s.duration / s.speed
            if duration_ms <= 0:
                continue

            # 如果 Sentence 本身带 lang, 就优先使用 s.lang, 否则用 target_language
            lang = target_language or "en"

            # 拆分长句子 -> 多段 sequential
            blocks = self._split_long_text_to_sub_blocks(
                text=sub_text,
                start_ms=start_local,
                duration_ms=duration_ms,
                lang=lang
            )

            for block in blocks:
                evt = pysubs2.SSAEvent(
                    start=int(block["start"]),
                    end=int(block["end"]),
                    text=block["text"]
                )
                subs.append(evt)

        # 设置"类YouTube"的默认样式
        # 若 "Default" 不存在则创建
        style = subs.styles.get("Default", pysubs2.SSAStyle())

        style.fontname = "Arial"             # 常见无衬线
        style.fontsize = 22
        style.bold = True
        style.italic = False
        style.underline = False

        # 颜色 (R, G, B, A=0 => 不透明)
        # 文字: 白色,  描边/背景: 黑色
        style.primarycolor = pysubs2.Color(255, 255, 255, 0)
        style.outlinecolor = pysubs2.Color(0, 0, 0, 100)  # 半透明黑
        style.borderstyle = 3  # 3 => 有背景块
        # style.outline = 4      # 背景矩形厚度
        style.shadow = 0
        style.alignment = pysubs2.Alignment.BOTTOM_CENTER
        style.marginv = 20    # 离底部像素
        # style.marginl = 30      # 左边距，根据需要调整
        # style.marginr = 30      # 右边距，根据需要调整

        # 更新回 default
        subs.styles["Default"] = style

        # 写入文件
        subs.save(output_sub_path, format="ass")
        logger.debug(f"_generate_subtitles_for_segment: 已写入字幕 => {output_sub_path}")

    # -------------------------------------------------------------------------
    # 字幕拆分逻辑: 根据语言不同设定 max_chars, 尝试在标点/空格/单词边界处断行
    # -------------------------------------------------------------------------
    def _split_long_text_to_sub_blocks(
        self,
        text: str,
        start_ms: float,
        duration_ms: float,
        lang: str = "en"
    ) -> List[dict]:
        """
        将文本在 [start_ms, start_ms+duration_ms] 区间内拆分成多块 sequential 字幕，
        并按照每块的字符数在总时长内做比例分配。

        lang: "zh"/"ja"/"ko"/"en", 若无匹配则默认英文
        """
        # 1) 确定每行最大字符数
        recommended_max_chars = {
            "zh": 20,
            "ja": 20,
            "ko": 20,
            "en": 40
        }
        if lang not in recommended_max_chars:
            lang = "en"
        max_chars = recommended_max_chars[lang]

        if len(text) <= max_chars:
            # 不需要拆分
            return [{
                "start": start_ms,
                "end":   start_ms + duration_ms,
                "text":  text
            }]

        # 2) 根据语言拆分成若干行
        chunks = self._chunk_text_by_language(text, lang, max_chars)

        # 3) 根据各行的字符数在总时长内进行时间分配
        sub_blocks = []
        total_chars = sum(len(c) for c in chunks)
        current_start = start_ms

        for c in chunks:
            chunk_len = len(c)
            chunk_dur = duration_ms * (chunk_len / total_chars) if total_chars > 0 else 0
            block_start = current_start
            block_end   = current_start + chunk_dur

            sub_blocks.append({
                "start": block_start,
                "end":   block_end,
                "text":  c
            })
            current_start += chunk_dur

        # 修正最后一块结束
        if sub_blocks:
            sub_blocks[-1]["end"] = start_ms + duration_ms
        else:
            # 理论上不会发生
            sub_blocks.append({
                "start": start_ms,
                "end":   start_ms + duration_ms,
                "text":  text
            })

        return sub_blocks

    def _chunk_text_by_language(self, text: str, lang: str, max_chars: int) -> List[str]:
        """
        根据语言做拆分:
         - 英文: 按单词拆, 避免截断单词
         - 中/日/韩: 按字符拆, 尝试在标点附近断行
        """
        cjk_puncts = set("，,。.!！？?；;：:、…~— ")
        eng_puncts = set(".,!?;: ")

        if lang == "en":
            return self._chunk_english_text(text, max_chars, eng_puncts)
        else:
            return self._chunk_cjk_text(text, max_chars, cjk_puncts)

    def _chunk_english_text(self, text: str, max_chars: int, puncts: set) -> List[str]:
        words = text.split()
        chunks = []
        current_line = []

        for w in words:
            # 计算本行加上下一个单词后长多少
            line_len = sum(len(x) for x in current_line) + len(current_line)  # 单词总长 + 空格数
            if line_len + len(w) > max_chars:
                if current_line:
                    chunks.append(" ".join(current_line))
                    current_line = []
            current_line.append(w)

        # 收尾
        if current_line:
            chunks.append(" ".join(current_line))

        return chunks

    def _chunk_cjk_text(self, text: str, max_chars: int, puncts: set) -> List[str]:
        chunks = []
        total_length = len(text)
        start_idx = 0

        while start_idx < total_length:
            # 基础结束位置
            end_idx = start_idx + max_chars
            
            # 如果下一个字符是标点，则包含到当前块
            if end_idx < total_length and text[end_idx] in puncts:
                end_idx += 1  # 包含标点符号

            # 确保不越界
            end_idx = min(end_idx, total_length)
            
            # 截取当前块
            chunk = text[start_idx:end_idx]
            chunks.append(chunk)
            
            # 移动起始位置
            start_idx = end_idx

        return chunks

    # -------------------------------------------------------------------------
    # 重置mixer状态(可选调用)
    # -------------------------------------------------------------------------
    async def reset(self):
        """
        重置 full_audio_buffer, 适合在一次任务结束后做清理。
        """
        self.full_audio_buffer = np.array([], dtype=np.float32)
        logger.debug("MediaMixer 已重置 full_audio_buffer")
