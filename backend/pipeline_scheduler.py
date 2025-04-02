from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator
import logging
import asyncio
from typing import List, Optional, Dict, Any
from pathlib import Path
import time
import sys
import os

# 然后导入config
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
from core.hls_manager import HLSManager
from utils.task_state import TaskState
from utils.ffmpeg_utils import concat_videos
from core.state_manager import StateManager

logger = logging.getLogger(__name__)

# 创建各服务的部署句柄
translator_handle = Translator.options(
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus": 1}  # 翻译器CPU资源
).bind()

model_in_handle = ModelInMaker.options(
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.3}  # 模型输入CPU资源
).bind()

tts_token_gen_handle = TtsTokenGenerator.options(
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus":1, "num_gpus": 0.3}  # TTS标记生成器资源
).bind()

audio_gen_handle = AudioGenerator.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.1}  # 音频生成器资源
).bind()

simplifier_handle = Translator.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}  # 简化器CPU资源
).bind()

media_mixer_handle = MediaMixer.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1}  # 媒体混合器CPU资源
).bind()

video_separator_handle = VideoSeparator.options(
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.15}  # 视频分离器GPU资源
).bind()

asr_handle = ASRModel.options(
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.15}  # ASR模型GPU资源
).bind()

duration_aligner_handle = DurationAligner.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}  # 时长对齐器GPU资源
).bind(simplifier_handle, model_in_handle, tts_token_gen_handle)

timestamp_adjuster_handle = TimestampAdjuster.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}  # 时间戳调整器GPU资源
).bind()

video_segmenter_handle = VideoSegmenter.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}  # 视频分段器GPU资源
).bind()

hls_manager_handle = HLSManager.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}  # HLS管理器资源
).bind()

# 在模块级别创建StateManager句柄 - 与HLSManager保持一致
state_manager_handle = StateManager.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}  # 状态管理器资源
).bind(Config())

@serve.deployment(
    num_replicas=3,
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus": 0.5},  # 降低资源请求以适应当前环境
    logging_config={"log_level": "INFO"}
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
                 media_mixer_handle: DeploymentHandle = None,
                 hls_manager_handle: DeploymentHandle = None):
        """初始化VideoTransPipe，注入所有依赖的服务handles"""
        # 获取状态管理器 - 如果未提供则通过名称获取
        self.state_manager = state_manager_handle or serve.get_deployment_handle("StateManager", app_name="StateManager")
        
        # 获取HLS管理器 - 如果未提供则通过名称获取，需要指定app_name
        self.hls_manager = hls_manager_handle or serve.get_deployment_handle("hls_manager", app_name="hls_manager")
        
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
        
        # 添加性能指标记录
        self.performance_metrics = {}
        
        self.logger.info("VideoTransPipe初始化完成")

    async def __call__(
        self, 
        task_id: str,
        video_path: str = None,
        target_language: str = None,
        generate_subtitle: bool = False
    ) -> Dict[str, Any]:
        """
        翻译视频任务：对视频进行分段，翻译处理所有分段，最后合并成完整视频
        
        Args:
            task_id: 任务ID
            video_path: 视频文件路径（可选，如果提供则创建新任务）
            target_language: 目标语言（可选）
            generate_subtitle: 是否生成字幕（可选）
            
        Returns:
            处理结果信息，包含最终视频路径
        """
        # 初始化该任务的性能指标
        self.performance_metrics[task_id] = {
            'total': {'start': time.time(), 'end': None, 'duration': None},
            'segments_processing': []  # 只记录总时间和每个分段的处理时间
        }
        
        try:
            # 增加更多日志记录点，确保每个关键步骤都有日志
            self.logger.info(f"====== VideoTransPipe开始处理任务: {task_id} ======")
            
            # 1. 先创建任务或获取任务状态
            if video_path:
                self.logger.info(f"创建新任务: {task_id}, 视频: {video_path}, 语言: {target_language}")
                try:
                    task_data = await self.state_manager.create_task.remote(
                        task_id=task_id,
                        video_path=video_path,
                        target_language=target_language,
                        generate_subtitle=generate_subtitle
                    )
                    task_state = task_data["task_state"]
                    self.logger.info(f"成功创建任务状态: {task_id}")
                except Exception as e:
                    self.logger.error(f"创建任务状态失败: {str(e)}", exc_info=True)
                    raise
            else:
                # 从StateManager获取任务状态
                self.logger.info(f"正在从StateManager获取任务状态: {task_id}")
                try:
                    task_state = await self.state_manager.get_task_state.remote(task_id)
                    if not task_state:
                        self.logger.error(f"任务不存在: {task_id}")
                        return {"status": "error", "message": "任务不存在"}
                    self.logger.info(f"成功获取任务状态: {task_id}")
                except Exception as e:
                    self.logger.error(f"获取任务状态失败: {str(e)}", exc_info=True)
                    raise
                
            # 3. 创建HLS管理器
            try:
                result = await self.hls_manager.create_manager.remote(task_id, task_state.task_paths)
                if result.get("status") != "success":
                    self.logger.warning(f"创建HLS管理器返回非成功状态: {result}, TaskID={task_id}")
                else:
                    self.logger.info(f"成功创建HLS管理器: {task_id}")
            except Exception as e:
                self.logger.error(f"创建HLS管理器失败: {e}, TaskID={task_id}")
                # 继续执行，不因HLS管理器创建失败而中断整个流程
            
            # 3. 调用VideoSegmenter对视频进行分段
            self.logger.info(f"开始对视频进行分段: {task_state.video_path}, TaskID={task_id}")
            
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
            self.logger.info(f"开始处理 {len(segments)} 个视频分段, TaskID={task_id}")
            
            for seg_idx, (seg_start, seg_duration) in enumerate(segments):
                # 记录分段处理开始时间
                segment_start_time = time.time()
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
                    sentences = await self.asr.generate.remote(
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
                                
                                # 如果处理成功，添加到HLS流
                                try:
                                    result = await self.hls_manager.add_segment.remote(
                                        task_id,
                                        output_path, 
                                        task_state.batch_counter
                                    )
                                    if result.get("status") == "success":
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
                                        self.logger.warning(f"添加HLS片段返回非成功状态: {result}, TaskID={task_id}")
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
                
                # 记录分段处理结束时间和持续时间
                segment_end_time = time.time()
                segment_duration = segment_end_time - segment_start_time
                
                # 记录这个分段的处理时间到性能指标中
                self.performance_metrics[task_id]['segments_processing'].append({
                    'segment_index': seg_idx,
                    'start_time': segment_start_time,
                    'end_time': segment_end_time,
                    'duration': segment_duration
                })
                
                self.logger.info(f"完成翻译分段 {seg_idx+1}/{len(segments)}, 耗时: {segment_duration:.2f}s")
            
            # 6. 合并所有视频分段，生成最终视频
            final_video_path = None
            if task_state.merged_segments:
                try:
                    # 创建最终输出路径
                    final_path = task_state.task_paths.output_dir / f"final_{task_id}.mp4"
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.logger.info(f"开始合并 {len(task_state.merged_segments)} 个视频分段，TaskID={task_id}")
                    
                    # 创建合并列表文件
                    list_txt = final_path.parent / f"concat_{task_id}.txt"
                    with open(list_txt, 'w', encoding='utf-8') as f:
                        for seg_mp4 in task_state.merged_segments:
                            abs_path = Path(seg_mp4).resolve()
                            f.write(f"file '{abs_path}'\n")
                    
                    # 执行合并命令
                    final_video_path = await concat_videos(
                        input_list=str(list_txt),
                        output_path=str(final_path)
                    )
                    
                    # 清理合并列表文件
                    if list_txt.exists():
                        list_txt.unlink()
                        
                    self.logger.info(f"视频合并完成，TaskID={task_id}")
                    
                    # 7. 标记播放列表为完成状态
                    if final_video_path and final_video_path.exists():
                        result = await self.hls_manager.finalize_playlist.remote(task_id)
                        if result.get("status") == "success":
                            self.logger.info(f"HLS播放列表已标记为完成，TaskID={task_id}")
                        else:
                            self.logger.warning(f"HLS播放列表标记完成失败: {result}，TaskID={task_id}")
                    
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
            # 记录总执行时间
            self.performance_metrics[task_id]['total']['end'] = time.time()
            self.performance_metrics[task_id]['total']['duration'] = self.performance_metrics[task_id]['total']['end'] - self.performance_metrics[task_id]['total']['start']
            total_duration = self.performance_metrics[task_id]['total']['duration']
            
            success_message = f"视频翻译完成，共处理 {len(segments)} 个分段，总耗时: {total_duration:.2f}s"
            self.logger.info(f"任务 {task_id} 总执行完成，总耗时: {total_duration:.2f}s")
            
            await self.state_manager.complete_task.remote(
                task_id, 
                success=True, 
                message=success_message,
                final_video_path=final_video_path
            )
            
            # 返回处理结果，包含性能指标
            if final_video_path and final_video_path.exists():
                return {
                    "status": "success", 
                    "message": success_message,
                    "final_video_path": str(final_video_path),
                    "performance": self.performance_metrics[task_id]
                }
            else:
                return {
                    "status": "error", 
                    "message": "翻译完成，但无法生成最终视频文件",
                    "performance": self.performance_metrics[task_id]
                }
                
        except Exception as e:
            # 记录总执行时间(即使出错)
            if task_id in self.performance_metrics and 'total' in self.performance_metrics[task_id]:
                self.performance_metrics[task_id]['total']['end'] = time.time()
                self.performance_metrics[task_id]['total']['duration'] = self.performance_metrics[task_id]['total']['end'] - self.performance_metrics[task_id]['total']['start']
                total_duration = self.performance_metrics[task_id]['total']['duration']
                self.logger.info(f"任务 {task_id} 失败，耗时: {total_duration:.2f}s")
            
            self.logger.exception(f"视频翻译失败: {e}")
            
            # 标记任务失败
            await self.state_manager.complete_task.remote(
                task_id, 
                success=False, 
                message=f"视频翻译失败: {e}"
            )
            
            return {
                "status": "error", 
                "message": str(e),
                "performance": self.performance_metrics.get(task_id, {})
            }

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
    media_mixer_handle,
    None   # hls_manager_handle在VideoTransPipe初始化时会自动获取
)

# 添加Ray和Ray Serve的初始化和服务管理函数
def init_ray(address="auto", namespace="videotrans", log_to_driver=True):
    """
    初始化Ray运行时
    
    Args:
        address: Ray集群地址，默认为'auto'自动连接本地Ray实例
        namespace: Ray命名空间
        log_to_driver: 是否将日志输出到驱动程序
    
    Returns:
        是否初始化成功
    """
    import ray
    
    if not ray.is_initialized():
        ray.init(address=address, namespace=namespace, log_to_driver=log_to_driver)
        logger.info(f"Ray已初始化，地址: {address}, 命名空间: {namespace}")
    else:
        logger.info("Ray已经初始化，跳过初始化步骤")
    
    return ray.is_initialized()

def start_serve(detached=False):
    """
    启动Ray Serve服务
    
    Args:
        detached: 是否以分离模式运行
    """
    serve.start(detached=detached)
    logger.info(f"Ray Serve已启动，detached模式: {detached}")

def setup_pipeline_services():
    """
    设置和部署VideoTrans流水线所需的核心服务
    
    Args:
        pipeline_app_name: 流水线引擎应用名称
    
    Returns:
        部署状态信息
    """
    # 确保Ray已初始化
    init_ray()
    
    # 启动Ray Serve
    start_serve()
    
    # 1. 首先部署StateManager - 使用全局变量中的句柄
    serve.run(state_manager_handle, name="StateManager", route_prefix=None)
    logger.info(f"StateManager已部署，应用名: StateManager")
    
    # 2. 部署HLSManager - 使用全局变量中的句柄
    serve.run(hls_manager_handle, name="hls_manager", route_prefix=None)
    logger.info("HLSManager已部署，应用名: hls_manager")
    
    # 等待确保StateManager和HLSManager就绪
    time.sleep(2)
    
    # 3. 部署PipelineEngine
    serve.run(app, name="PipelineEngine", route_prefix=None)
    logger.info(f"PipelineEngine已部署，应用名: PipelineEngine")
    
    # 等待确保PipelineEngine就绪
    time.sleep(1)
    
    return {
        "state_manager": "StateManager",
        "hls_manager": "hls_manager",
        "pipeline": "PipelineEngine",
        "status": "deployed"
    }

# 用于直接运行pipeline服务的入口点
if __name__ == "__main__":
    setup_pipeline_services()
