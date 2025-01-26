import numpy as np
import logging
import soundfile as sf
import os
import asyncio
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import Optional, List
from utils.decorators import handle_errors

# 引入统一的 FFmpegTool
from utils.ffmpeg_utils import FFmpegTool

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

        # 新增 ffmpeg 工具类
        self.ffmpeg_tool = FFmpegTool()

    @handle_errors(logger)
    async def mixed_media_maker(self, sentences, task_state=None, output_path=None):
        """
        处理一批句子的音频和视频，并最终生成一个带音频的分段 MP4。
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

        # 拼接所有句子的合成音频
        for sentence in sentences:
            if sentence.generated_audio is not None:
                audio_data = np.asarray(sentence.generated_audio, dtype=np.float32)
                if len(full_audio) > 0:
                    # 调用更高级的音频衔接方式
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
        """
        使用“等功率（equal power）”交叉淡入淡出进行更高级的音频平滑衔接。
        当存在已累积的 self.full_audio_buffer 时，取其中末尾 overlap 样本
        与新 audio_data 的开头 overlap 样本作交叉融合。

        Returns:
            np.ndarray: 更新后的 new audio_data
        """
        if audio_data is None or len(audio_data) == 0:
            return np.array([], dtype=np.float32)

        # 交叉区域长度取 self.overlap 与现有 buffer、新音频长度的最小值
        cross_len = min(self.overlap, len(self.full_audio_buffer), len(audio_data))
        if cross_len <= 0:
            # 如果没有可交叉的长度，则直接返回原音频
            return audio_data

        # 等功率淡入淡出系数
        fade_out = np.sqrt(np.linspace(1.0, 0.0, cross_len, dtype=np.float32))
        fade_in = np.sqrt(np.linspace(0.0, 1.0, cross_len, dtype=np.float32))

        audio_data = audio_data.copy()

        # 取出 self.full_audio_buffer 末尾 cross_len 片段进行淡出
        overlap_region = self.full_audio_buffer[-cross_len:]

        # 在交叉区对新音频做淡入，对旧音频做淡出，再叠加
        audio_data[:cross_len] = overlap_region * fade_out + audio_data[:cross_len] * fade_in

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
        """切割出指定时段的视频片段，并与音频合并输出。"""
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

            # 1) 截取无声视频
            await self.ffmpeg_tool.cut_video_track(
                input_path=video_path,
                output_path=temp_video.name,
                start=start_time,
                end=end_time
            )

            # 2) 将合成音频写入临时 wav
            await asyncio.to_thread(sf.write, temp_audio.name, audio_data, self.sample_rate)

            # 3) 将无声视频和合成音轨合并
            await self.ffmpeg_tool.cut_video_with_audio(
                input_video_path=temp_video.name,
                input_audio_path=temp_audio.name,
                output_path=output_path
            )

    async def reset(self):
        self.full_audio_buffer = np.array([], dtype=np.float32)
        logger.debug("已重置 mixer 状态")
