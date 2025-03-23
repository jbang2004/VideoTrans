from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator
import logging
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import time

from config import Config
from core.translation.translator import Translator
from core.model_in_maker import ModelInMaker
from core.media_mixer import MediaMixer
from core.tts_token_generator import TtsTokenGenerator
from core.audio_generator import AudioGenerator
from core.video_separator import VideoSeparator
from core.asr_model import ASRModel
from core.timeadjust.duration_aligner import DurationAligner
from core.timeadjust.timestamp_adjuster import TimestampAdjuster
from core.video_segmenter import VideoSegmenter
from utils.task_state import TaskState
from utils.ffmpeg_utils import FFmpegTool
from core.state_manager import StateManager

logger = logging.getLogger(__name__)

# 创建各服务的部署句柄
translator_handle = Translator.bind()
model_in_handle = ModelInMaker.bind()
tts_token_gen_handle = TtsTokenGenerator.bind()
audio_gen_handle = AudioGenerator.bind()
simplifier_handle = Translator.bind()
media_mixer_handle = MediaMixer.bind()
video_separator_handle = VideoSeparator.bind()
asr_handle = ASRModel.bind()
duration_aligner_handle = DurationAligner.bind(simplifier_handle, model_in_handle, tts_token_gen_handle)
timestamp_adjuster_handle = TimestampAdjuster.bind()
video_segmenter_handle = VideoSegmenter.bind()
# 不在模块级别获取StateManager的句柄
# state_manager_handle = serve.get_deployment_handle("StateManager", app_name="StateManager")

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1.0, "num_gpus": 0.1}  # 降低资源请求以适应当前环境
)
class VideoTransPipe:
    """
    视频翻译流水线的主协调器
    
    负责:
    1. 协调各个处理组件
    2. 管理视频分段处理
    3. 合并最终视频
    """
    def __init__(self,
                 state_manager_handle: DeploymentHandle = None,
                 video_segmenter_handle: DeploymentHandle = None,
                 video_separator_handle: DeploymentHandle = None,
                 asr_handle: DeploymentHandle = None,
                 translator_handle: DeploymentHandle = None,
                 model_in_handle: DeploymentHandle = None,
                 tts_token_gen_handle: DeploymentHandle = None,
                 duration_aligner_handle: DeploymentHandle = None,
                 audio_gen_handle: DeploymentHandle = None,
                 timestamp_adjuster_handle: DeploymentHandle = None,
                 media_mixer_handle: DeploymentHandle = None):
        """初始化VideoTransPipe，注入所有依赖的服务handles"""
        # 获取状态管理器 - 在实际初始化时获取
        self.state_manager = state_manager_handle or serve.get_deployment_handle("StateManager", app_name="StateManager")
        
        # 获取各组件handles
        self.video_segmenter = video_segmenter_handle or video_segmenter_handle
        self.video_separator = video_separator_handle or video_separator_handle
        self.asr = asr_handle or asr_handle
        self.translator = translator_handle.options(stream=True) if translator_handle else translator_handle.options(stream=True)
        self.model_in = model_in_handle.options(stream=True) if model_in_handle else model_in_handle.options(stream=True)
        self.tts_token_gen = tts_token_gen_handle or tts_token_gen_handle
        self.duration_aligner = duration_aligner_handle or duration_aligner_handle
        self.audio_gen = audio_gen_handle or audio_gen_handle
        self.timestamp_adjuster = timestamp_adjuster_handle or timestamp_adjuster_handle
        self.media_mixer = media_mixer_handle or media_mixer_handle
        
        # 基础配置
        self.config = Config()
        self.sample_rate = self.config.TARGET_SR
        self.logger = logger
        self.ffmpeg_tool = FFmpegTool()
        
        self.logger.info("VideoTransPipe初始化完成")

    async def __call__(
        self, 
        task_id: str
    ) -> Dict[str, Any]:
        """
        翻译视频任务：对视频进行分段，翻译处理所有分段，最后合并成完整视频
        
        Args:
            task_id: 任务ID
            
        Returns:
            处理结果信息，包含最终视频路径
        """
        try:
            # 立即设置HLS为就绪状态，让前端显示"正在获取视频流..."
            await self.state_manager.update_task_progress.remote(
                task_id, 
                hls_ready=True
            )
            self.logger.info(f"HLS状态已设置为就绪 -> TaskID={task_id}")
            
            # 1. 从StateManager获取任务状态
            try:
                self.logger.info(f"正在从StateManager获取任务状态: {task_id}")
                task_state = await self.state_manager.get_task_state.remote(task_id)
                if not task_state:
                    self.logger.error(f"任务不存在: {task_id}")
                    return {"status": "error", "message": "任务不存在"}
                self.logger.info(f"成功获取任务状态: {task_id}")
            except Exception as e:
                self.logger.error(f"获取任务状态失败: {str(e)}")
                return {"status": "error", "message": f"获取任务状态失败: {str(e)}"}
                
            # 2. 从StateManager获取HLS管理器
            try:
                self.logger.info(f"正在从StateManager获取HLS管理器: {task_id}")
                hls_manager = await self.state_manager.get_hls_manager.remote(task_id)
                self.logger.info(f"成功获取HLS管理器: {task_id}")
            except Exception as e:
                self.logger.error(f"获取HLS管理器失败: {str(e)}")
                return {"status": "error", "message": f"获取HLS管理器失败: {str(e)}"}
            
            # 记录开始时间
            start_time = time.time()
            self.logger.info(f"开始翻译视频: {task_state.video_path}, TaskID={task_id}")
                
            # 3. 调用VideoSegmenter对视频进行分段
            segmenter_result = await self.video_segmenter.segment_video.remote(task_state.video_path)
            
            if segmenter_result["status"] != "success":
                self.logger.error(f"视频分段失败: {segmenter_result['message']}")
                await self.state_manager.complete_task.remote(task_id, success=False, message=segmenter_result["message"])
                return {"status": "error", "message": segmenter_result["message"]}
            
            # 4. 获取分段信息
            segments = segmenter_result["segments"]
            self.logger.info(f"视频分段完成，共 {len(segments)} 个分段")
            
            # 保存分段信息到task_state
            task_state.segments = segments
            
            # 5. 处理所有分段
            for seg_idx, (seg_start, seg_duration) in enumerate(segments):
                self.logger.info(f"开始翻译分段 {seg_idx+1}/{len(segments)}: 开始时间={seg_start:.2f}s, 持续时间={seg_duration:.2f}s")
                
                try:
                    # 5.1 提取并分离人声/背景
                    media_files = await self.video_separator.separate_video.remote(
                        video_path=task_state.video_path,
                        start=seg_start,
                        output_dir=str(task_state.task_paths.processing_dir),
                        segment_index=seg_idx,
                        target_sr=self.sample_rate,
                        duration=seg_duration
                    )
                    task_state.segment_media_files[seg_idx] = media_files
                    
                    # 获取该分段的媒体文件信息
                    media_files = task_state.segment_media_files.get(seg_idx)
                    if not media_files or 'vocals' not in media_files:
                        self.logger.error(f"找不到分段 {seg_idx} 的vocals文件, TaskID={task_id}")
                        continue
                    
                    # 5.2 执行ASR识别
                    sentences = await self.asr.generate_async.remote(
                        input=media_files['vocals'],
                        cache={},
                        language="auto",
                        use_itn=True,
                        batch_size_s=60,
                        merge_vad=False
                    )
                    
                    self.logger.info(f"ASR识别完成: {len(sentences)} 条句子, seg={seg_idx}, TaskID={task_id}")
                    
                    if not sentences:
                        self.logger.warning(f"ASR结果为空, seg={seg_idx}, TaskID={task_id}")
                        continue
                    
                    # 5.3 为句子添加元数据
                    for s in sentences:
                        s.segment_index = seg_idx
                        s.segment_start = seg_start
                        s.task_id = task_id
                        s.sentence_id = task_state.sentence_counter
                        task_state.sentence_counter += 1
                        
                    self.logger.debug(f"处理 {len(sentences)} 个句子, TaskID={task_id}")
                    
                    # 5.4 翻译处理流程
                    async for translated_sentences in self.translator.translate_sentences.remote(
                        sentences,
                        target_language=task_state.target_language,
                        batch_size=self.config.TRANSLATION_BATCH_SIZE
                    ):
                        # 5.5 模型输入处理
                        async for modelin_sentences in self.model_in.modelin_maker.remote(
                            translated_sentences,
                            reuse_speaker=False,
                            batch_size=self.config.MODELIN_BATCH_SIZE
                        ):
                            # 5.6 TTS标记生成
                            self.logger.info(f"TTS token生成开始")
                            tts_token_sentences = await self.tts_token_gen.generate_tts_tokens.remote(
                                modelin_sentences
                            )
                            
                            # 5.7 时长对齐
                            self.logger.info(f"创建时长对齐任务")
                            aligned_sentences = await self.duration_aligner.remote(
                                tts_token_sentences,
                                1.1  # 使用默认值1.1，而不是self.config.MAX_SPEED
                            )
                            
                            # 5.8 音频生成
                            self.logger.info(f"创建音频生成任务")
                            audio_gen_sentences = await self.audio_gen.generate_audio.remote(
                                aligned_sentences
                            )
                            
                            # 5.9 时间戳调整
                            self.logger.info(f"创建时间戳调整任务")
                            sentences_with_timestamps = await self.timestamp_adjuster.remote(
                                audio_gen_sentences,
                                self.sample_rate,
                                task_state.current_time
                            )
                            
                            # 更新当前时间（使用最后一个句子的结束时间）
                            if sentences_with_timestamps:
                                last_sentence = sentences_with_timestamps[-1]
                                task_state.current_time = last_sentence.adjusted_start + last_sentence.adjusted_duration
                            
                            self.logger.info(f"时间戳调整任务完成: {len(sentences_with_timestamps) if sentences_with_timestamps else 0}个句子")
                            
                            # 5.10 媒体混合
                            self.logger.info(f"调用MediaMixer进行媒体混合")
                            output_path = await self.media_mixer.mix_media.remote(
                                sentences_with_timestamps,
                                task_state
                            )
                            
                            if output_path:
                                # 记录已处理的片段
                                task_state.merged_segments.append(output_path)
                                
                                # 如果处理成功且有HLS管理器，添加到HLS流
                                if hls_manager:
                                    try:
                                        # 使用常规方法调用而非Ray远程调用
                                        success = hls_manager.add_segment(
                                            output_path, 
                                            task_state.batch_counter
                                        )
                                        if success:
                                            self.logger.info(f"分段 {task_state.batch_counter} 已加入 HLS -> TaskID={task_id}")
                                            # 成功添加第一个分段后，设置HLS就绪状态
                                            if not task_state.hls_ready and task_state.batch_counter == 0:
                                                task_state.hls_ready = True
                                                # 更新StateManager中的HLS状态
                                                await self.state_manager.update_task_progress.remote(
                                                    task_id, 
                                                    hls_ready=True
                                                )
                                                self.logger.info(f"HLS播放列表已就绪 -> TaskID={task_id}")
                                        else:
                                            self.logger.error(f"添加HLS片段失败 -> TaskID={task_id}")
                                    except Exception as e:
                                        self.logger.error(f"添加HLS片段失败: {e} -> TaskID={task_id}")
                                
                                # 更新批次计数器
                                task_state.batch_counter += 1
                                
                                # 更新StateManager中的进度
                                await self.state_manager.update_task_progress.remote(
                                    task_id, 
                                    batch_counter=task_state.batch_counter
                                )
                
                except Exception as e:
                    self.logger.error(f"翻译分段 {seg_idx} 失败: {e}, TaskID={task_id}")
                    # 继续处理下一个分段
                    continue
                
                self.logger.info(f"完成翻译分段 {seg_idx+1}/{len(segments)}")
            
            # 6. 合并所有视频分段，生成最终视频
            final_video_path = None
            if task_state.merged_segments:
                try:
                    # 创建最终输出路径
                    final_path = task_state.task_paths.output_dir / f"final_{task_id}.mp4"
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 合并所有视频分段
                    self.logger.info(f"开始合并 {len(task_state.merged_segments)} 个视频分段，TaskID={task_id}")
                    
                    # 创建合并列表文件
                    list_txt = final_path.parent / f"concat_{task_id}.txt"
                    with open(list_txt, 'w', encoding='utf-8') as f:
                        for seg_mp4 in task_state.merged_segments:
                            abs_path = Path(seg_mp4).resolve()
                            f.write(f"file '{abs_path}'\n")
                    
                    # 执行合并命令
                    start_time = time.time()
                    final_video_path = self.ffmpeg_tool.concat_videos(
                        input_list=str(list_txt),
                        output_path=str(final_path)
                    )
                    duration = time.time() - start_time
                    
                    # 清理合并列表文件
                    if list_txt.exists():
                        list_txt.unlink()
                        
                    self.logger.info(f"视频合并完成，耗时={duration:.2f}s，TaskID={task_id}")
                    
                    # 7. 如果有HLS管理器，标记播放列表为完成状态
                    if hls_manager and final_video_path and final_video_path.exists():
                        success = hls_manager.finalize_playlist()
                        if success:
                            self.logger.info(f"HLS播放列表已标记为完成，TaskID={task_id}")
                        else:
                            self.logger.warning(f"HLS播放列表标记完成失败，TaskID={task_id}")
                    
                except Exception as e:
                    self.logger.error(f"视频合并失败: {e}, TaskID={task_id}")
                    if final_path.exists():
                        final_path.unlink()
                    
                    # 标记任务失败
                    await self.state_manager.complete_task.remote(
                        task_id, 
                        success=False, 
                        message=f"视频合并失败: {e}"
                    )
                    
                    return {"status": "error", "message": f"视频合并失败: {e}"}
            else:
                self.logger.warning(f"没有可合并的视频分段，TaskID={task_id}")
                
                # 标记任务失败
                await self.state_manager.complete_task.remote(
                    task_id, 
                    success=False, 
                    message="没有可合并的视频分段"
                )
                
                return {"status": "error", "message": "没有可合并的视频分段"}
            
            # 8. 清理GPU显存
            try:
                import torch
                torch.cuda.empty_cache()
                self.logger.info(f"已释放未使用的GPU显存, TaskID={task_id}")
            except Exception as e:
                self.logger.warning(f"释放GPU显存失败: {e}")
            
            # 9. 标记任务完成
            total_duration = time.time() - start_time
            success_message = f"视频翻译完成，共处理 {len(segments)} 个分段，总耗时: {total_duration:.2f}s"
            
            await self.state_manager.complete_task.remote(
                task_id, 
                success=True, 
                message=success_message,
                final_video_path=final_video_path
            )
            
            # 返回处理结果
            if final_video_path and final_video_path.exists():
                return {
                    "status": "success", 
                    "message": success_message,
                    "final_video_path": str(final_video_path)
                }
            else:
                return {
                    "status": "error", 
                    "message": "翻译完成，但无法生成最终视频文件"
                }
                
        except Exception as e:
            self.logger.exception(f"视频翻译失败: {e}")
            
            # 标记任务失败
            await self.state_manager.complete_task.remote(
                task_id, 
                success=False, 
                message=f"视频翻译失败: {e}"
            )
            
            return {"status": "error", "message": str(e)}

# 创建应用部署
app = VideoTransPipe.bind(
    None,  # state_manager_handle在VideoTransPipe初始化时会自动获取
    video_segmenter_handle,
    video_separator_handle,
    asr_handle,
    translator_handle, 
    model_in_handle, 
    tts_token_gen_handle, 
    duration_aligner_handle,
    audio_gen_handle, 
    timestamp_adjuster_handle,
    media_mixer_handle
)
