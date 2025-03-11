import os
import asyncio
import soundfile as sf
import numpy as np
import logging
from contextlib import ExitStack
from tempfile import NamedTemporaryFile
from typing import List, Any
from pathlib import Path

from utils.ffmpeg_utils import FFmpegTool
from utils.subtitle_utils import generate_subtitles_for_segment

logger = logging.getLogger(__name__)

async def add_video_segment(
    video_path: str,
    start_time: float,
    duration: float,
    audio_data: np.ndarray,
    output_path: str,
    sentences: List[Any],
    generate_subtitle: bool,
    task_state: Any,
    sample_rate: int,
    ffmpeg_tool: FFmpegTool
):
    """
    从原视频里截取 [start_time, start_time + duration] 的视频段(无声)，
    与合成音频合并。
    若 generate_subtitle=True, 则生成 .ass 字幕并在 ffmpeg 工具中进行"烧制"。
    
    Args:
        video_path: 视频文件路径
        start_time: 开始时间（秒）
        duration: 持续时间（秒）
        audio_data: 音频数据
        output_path: 输出文件路径
        sentences: 句子列表
        generate_subtitle: 是否生成字幕
        task_state: 任务状态对象
        sample_rate: 采样率
        ffmpeg_tool: FFmpeg工具实例
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"add_video_segment: 视频文件不存在: {video_path}")
    if len(audio_data) == 0:
        raise ValueError("add_video_segment: 无音频数据")
    if duration <= 0:
        raise ValueError("add_video_segment: 无效时长 <=0")

    with ExitStack() as stack:
        temp_video = stack.enter_context(NamedTemporaryFile(suffix='.mp4'))
        temp_audio = stack.enter_context(NamedTemporaryFile(suffix='.wav'))

        end_time = start_time + duration

        # 1) 截取视频 (无音轨)
        await ffmpeg_tool.cut_video_track(
            input_path=video_path,
            output_path=temp_video.name,
            start=start_time,
            end=end_time
        )

        # 2) 写合成音频到临时文件
        await asyncio.to_thread(sf.write, temp_audio.name, audio_data, sample_rate)

        # 3) 如果需要字幕，则构建 .ass 并用 ffmpeg "烧"进去
        if generate_subtitle:
            temp_ass = stack.enter_context(NamedTemporaryFile(suffix='.ass'))
            # 调用生成字幕的函数
            await asyncio.to_thread(
                generate_subtitles_for_segment,
                sentences,
                start_time * 1000,   # segment_start_ms
                temp_ass.name,
                task_state.target_language
            )

            # 生成带字幕的视频
            await ffmpeg_tool.cut_video_with_subtitles_and_audio(
                input_video_path=temp_video.name,
                input_audio_path=temp_audio.name,
                subtitles_path=temp_ass.name,
                output_path=output_path
            )
        else:
            # 不加字幕，仅合并音频
            await ffmpeg_tool.cut_video_with_audio(
                input_video_path=temp_video.name,
                input_audio_path=temp_audio.name,
                output_path=output_path
            )

async def concat_video_segments(
    task_state: Any,
    output_path: Path,
    ffmpeg_tool: FFmpegTool,
    hls_manager=None
) -> Path:
    """
    合并所有处理后的视频分段，生成最终视频文件。
    
    Args:
        task_state: 任务状态对象
        output_path: 输出文件路径
        ffmpeg_tool: FFmpeg工具实例
        hls_manager: HLS管理器实例（可选）
        
    Returns:
        最终视频文件的路径，如果失败则返回None
    """
    if not task_state.merged_segments:
        logger.warning(f"[VideoUtils] 无可合并的视频分段, TaskID={task_state.task_id}")
        return None
        
    try:
        logger.info(f"[VideoUtils] 开始合并 {len(task_state.merged_segments)} 个视频分段, TaskID={task_state.task_id}")
        
        # 如果有HLS管理器，确保播放列表已完成
        if hls_manager and hls_manager.has_segments:
            await hls_manager.finalize_playlist()
            logger.info(f"[VideoUtils] HLS播放列表已完成, TaskID={task_state.task_id}")
        
        # 创建合并列表文件
        list_txt = output_path.parent / f"concat_{task_state.task_id}.txt"
        with open(list_txt, 'w', encoding='utf-8') as f:
            for seg_mp4 in task_state.merged_segments:
                abs_path = Path(seg_mp4).resolve()
                f.write(f"file '{abs_path}'\n")
        
        # 执行合并命令
        start_time = asyncio.get_event_loop().time()
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_txt),
            "-c", "copy",
            str(output_path)
        ]
        
        # 使用ffmpeg_tool执行命令
        success = await ffmpeg_tool.run_command(cmd)
        processing_time = asyncio.get_event_loop().time() - start_time
        
        if success and output_path.exists():
            logger.info(
                f"[VideoUtils] 视频分段合并成功, 耗时: {processing_time:.2f}s, "
                f"输出文件: {output_path}, TaskID={task_state.task_id}"
            )
            
            # 可选：清理临时文件
            try:
                list_txt.unlink(missing_ok=True)
                logger.debug(f"[VideoUtils] 已删除临时合并列表文件: {list_txt}")
            except Exception as e:
                logger.warning(f"[VideoUtils] 清理临时文件失败: {e}")
            
            return output_path
        else:
            logger.error(f"[VideoUtils] 视频分段合并失败, TaskID={task_state.task_id}")
            return None
            
    except Exception as e:
        logger.exception(f"[VideoUtils] 合并视频分段时发生异常: {str(e)}, TaskID={task_state.task_id}")
        return None 