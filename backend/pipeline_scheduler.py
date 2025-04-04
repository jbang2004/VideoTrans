from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponseGenerator
import logging
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator
from pathlib import Path
import time
import sys
import os
import aiofiles
import torch
import gc

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
    max_ongoing_requests=3,
    ray_actor_options={"num_cpus": 0.5}  # 翻译器CPU资源
).bind()

model_in_handle = ModelInMaker.options(
    num_replicas=1,
    max_ongoing_requests=3,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.1}  # 模型输入CPU资源
).bind()

tts_token_gen_handle = TtsTokenGenerator.options(
    num_replicas=3,
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus":0.9, "num_gpus": 0.2}  # TTS标记生成器资源
).bind()

audio_gen_handle = AudioGenerator.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5, "num_gpus": 0.1}  # 音频生成器资源
).bind()

simplifier_handle = Translator.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}  # 简化器CPU资源
).bind()

media_mixer_handle = MediaMixer.options(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}  # 媒体混合器CPU资源
).bind()

video_separator_handle = VideoSeparator.options(
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.1}  # 视频分离器GPU资源
).bind()

asr_handle = ASRModel.options(
    num_replicas=1,
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.1}  # ASR模型GPU资源
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
    num_replicas=3,  # 改回1个副本，避免多副本共享状态导致段错误
    max_ongoing_requests=1,
    ray_actor_options={"num_cpus": 0.5},
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
        # 获取状态管理器和HLS管理器
        self.state_manager = state_manager_handle or serve.get_deployment_handle("StateManager", app_name="StateManager")
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
        接收视频，开始翻译处理
        
        Args:
            task_id: 任务ID
            video_path: 视频文件路径
            target_language: 目标翻译语言 
            generate_subtitle: 是否生成字幕
            
        Returns:
            Dict[str, Any]: 处理结果状态信息
        """
        task_state = None  # 先声明局部变量，确保finally能清理
        try:
            self.logger.info(f"开始处理任务: {task_id}")
            
            # 1. 获取/创建任务状态
            try:
                if video_path:
                    task_data = await self.state_manager.create_task.remote(
                        task_id=task_id,
                        video_path=video_path,
                        target_language=target_language,
                        generate_subtitle=generate_subtitle
                    )
                    task_state = task_data["task_state"]
                else:
                    task_state = await self.state_manager.get_task_state.remote(task_id)
            except Exception as e:
                self.logger.exception(f"任务状态初始化失败: {str(e)}")
                return {"status": "error", "message": f"任务初始化失败: {str(e)}"}
            
            if not task_state:
                return {"status": "error", "message": "任务不存在"}

            # 2. 初始化HLS管理器
            try:
                await self.hls_manager.create_manager.remote(task_id, task_state.task_paths)
            except Exception as e:
                await self.state_manager.complete_task.remote(task_id, False, f"HLS管理器创建失败: {str(e)}")
                return {"status": "error", "message": f"HLS管理器创建失败: {str(e)}"}

            # 3. 视频分段
            segments = None
            try:
                video_segmenter_result = await self.video_segmenter.segment_video.remote(task_state.video_path)
                
                if video_segmenter_result["status"] != "success":
                    error_msg = video_segmenter_result.get("message", "未知错误")
                    await self.state_manager.complete_task.remote(task_id, False, f"视频分段失败: {error_msg}")
                    return {"status": "error", "message": f"视频分段失败: {error_msg}"}
                    
                segments = video_segmenter_result["segments"]
                self.logger.info(f"[{task_id}] 视频分段完成，共 {len(segments)} 个分段")
                task_state.segments = segments
            except Exception as e:
                await self.state_manager.complete_task.remote(task_id, False, f"视频分段失败: {str(e)}")
                return {"status": "error", "message": f"视频分段失败: {str(e)}"}

            # 4. 循环处理每个分段
            for seg_idx, (seg_start, seg_duration) in enumerate(segments):
                try:
                    self.logger.info(f"[{task_id}] 开始处理分段 {seg_idx+1}/{len(segments)}")
                    
                    # 4.1 视频/音频分离
                    media_files = await self.video_separator.separate_video.remote(
                        video_path=task_state.video_path,
                        start=seg_start,
                        output_dir=str(task_state.task_paths.processing_dir),
                        segment_index=seg_idx,
                        target_sr=self.config.TARGET_SR,
                        duration=seg_duration
                    )
                    if not media_files or "vocals" not in media_files:
                        raise ValueError("视频分离失败，或未返回vocals路径")
                    
                    task_state.segment_media_files[seg_idx] = media_files

                    # 4.2 ASR处理
                    sentences = await self.asr.generate.remote(
                        input=media_files["vocals"],
                        cache={},
                        language="auto",
                        use_itn=True,
                        batch_size_s=60,
                        merge_vad=False
                    )
                    if not sentences:
                        self.logger.warning(f"[{task_id}][分段{seg_idx}] ASR没有返回句子，可能是静音片段")
                        continue  # 跳过这个分段
                    
                    # 初始化句子元数据
                    for i, s in enumerate(sentences):
                        s.segment_index = seg_idx
                        s.segment_start = seg_start
                        s.sentence_id = task_state.sentence_counter + i
                        s.task_id = task_id
                    
                    task_state.sentence_counter += len(sentences)
                    self.logger.info(f"[{task_id}][分段{seg_idx}] ASR识别完成: {len(sentences)}个句子")

                    # 4.3 翻译处理流水线
                    batch_count = await self._run_translation_pipeline(sentences, task_state, seg_idx)
                    
                    self.logger.info(f"[{task_id}][分段{seg_idx}] 翻译流水线处理完成，共{batch_count}个批次")
                    
                    # 每处理完一个分段，主动触发GC
                    self._clean_memory()
                    
                    # 让出CPU，让系统有机会进行其他处理
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.exception(f"[{task_id}] 处理分段{seg_idx}失败: {str(e)}")
                    # 继续尝试处理下一个分段

            # 5. 合并所有分段视频
            try:
                self.logger.info(f"[{task_id}] 开始合并{len(task_state.merged_segments)}个视频片段")
                
                if not task_state.merged_segments:
                    await self.state_manager.complete_task.remote(task_id, False, "没有处理成功的视频片段可以合并")
                    return {"status": "error", "message": "没有处理成功的视频片段"}
                    
                # 创建合并列表
                list_txt_path = str(task_state.task_paths.processing_dir / "concat_list.txt")
                with open(list_txt_path, "w") as f:
                    for seg_mp4 in task_state.merged_segments:
                        f.write(f"file '{Path(seg_mp4).resolve()}'\n")
                
                # 最终输出路径
                final_output_path = str(task_state.task_paths.output_dir / f"{task_id}.mp4")
                
                # 执行合并
                final_video_path = await concat_videos(list_txt_path, final_output_path)
                
                if final_video_path and final_video_path.exists():
                    # 标记HLS播放列表完成和任务完成
                    await self.hls_manager.finalize_playlist.remote(task_id)
                    await self.state_manager.complete_task.remote(task_id, True, "视频处理成功", final_video_path)
                    
                    return {
                        "status": "success", 
                        "message": "视频处理成功", 
                        "output_path": str(final_video_path)
                    }
                else:
                    await self.state_manager.complete_task.remote(task_id, False, "视频合并失败")
                    return {"status": "error", "message": "视频合并失败"}
            except Exception as e:
                await self.state_manager.complete_task.remote(task_id, False, f"合并视频失败: {str(e)}")
                return {"status": "error", "message": f"合并视频失败: {str(e)}"}
                
        except Exception as e:
            self.logger.exception(f"[{task_id}] 处理过程中发生未捕获的异常: {str(e)}")
            await self.state_manager.complete_task.remote(task_id, False, f"处理失败: {str(e)}")
            return {"status": "error", "message": f"处理失败: {str(e)}"}
        finally:
            # 最终清理内存
            if task_state is not None:
                # 手动删除大型变量引用
                if hasattr(task_state, 'segment_media_files'):
                    task_state.segment_media_files.clear()
            self._clean_memory()

    def _clean_memory(self) -> None:
        """
        清理内存和GPU缓存
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def _run_translation_pipeline(self, sentences: List, task_state: TaskState, seg_idx: int) -> int:
        """
        运行翻译流水线，包括翻译、TTS生成和媒体混合
        
        Args:
            sentences: 要处理的句子列表
            task_state: 当前任务状态
            seg_idx: 当前处理的分段索引
            
        Returns:
            int: 处理的批次数量
        """
        batch_counter = 0
        
        try:
            # 使用异步生成器处理翻译
            async for translated_sentences in self.translator.translate_sentences.remote(
                sentences, 
                target_language=task_state.target_language,
                batch_size=self.config.TRANSLATION_BATCH_SIZE
            ):
                try:
                    batch_counter += 1
                    
                    # 生成模型输入，简化嵌套层级
                    async for modelin_sentences in self.model_in.modelin_maker.remote(
                        translated_sentences, 
                        reuse_speaker=False, 
                        batch_size=self.config.MODELIN_BATCH_SIZE
                    ):
                        try:
                            # 处理音频生成和混合
                            await self._generate_audio_and_mix(
                                modelin_sentences,
                                task_state,
                                seg_idx
                            )
                                
                            # 内存管理：每批次处理后强制释放无需保留的对象
                            del modelin_sentences
                            
                            # 每处理5个批次强制GC
                            if batch_counter % 5 == 0:
                                self._clean_memory()
                                await asyncio.sleep(0.1)  # 让事件循环运行
                                
                        except Exception as e:
                            self.logger.error(f"处理模型输入批次失败: {str(e)}")
                            # 继续尝试下一批
                    
                    # 每个翻译批次处理完成后删除引用
                    del translated_sentences
                    
                except Exception as e:
                    self.logger.error(f"处理翻译批次失败: {str(e)}")
                    # 继续尝试下一批
        except Exception as e:
            self.logger.error(f"翻译处理生成器错误: {str(e)}")
        finally:
            # 释放资源
            self._clean_memory()
            
        return batch_counter

    async def _generate_audio_and_mix(self, modelin_sentences: List, task_state: TaskState, seg_idx: int) -> None:
        """
        处理单个句子批次的音频生成和混合
        
        Args:
            modelin_sentences: 带有模型输入的句子列表
            task_state: 当前任务状态
            seg_idx: 当前处理的分段索引
        """
        task_id = task_state.task_id
        
        try:
            # 1. 生成TTS Token
            tts_token_sentences = await self.tts_token_gen.generate_tts_tokens.remote(modelin_sentences)
            
            # 2. 时长对齐
            aligned_sentences = await self.duration_aligner.remote(tts_token_sentences, 1.1)
            
            # 3. 音频生成
            audio_gen_sentences = await self.audio_gen.generate_audio.remote(aligned_sentences)
            
            # 4. 时间戳调整
            sentences_with_timestamps = await self.timestamp_adjuster.remote(
                audio_gen_sentences, 
                self.config.TARGET_SR,
                task_state.current_time
            )
            
            if not sentences_with_timestamps:
                self.logger.warning(f"[{task_id}][分段{seg_idx}] 时间戳调整返回空结果")
                return
            
            # 更新当前时间
            last_sentence = sentences_with_timestamps[-1]
            task_state.current_time = last_sentence.adjusted_start + last_sentence.adjusted_duration
            
            # 5. 媒体混合
            output_path = await self.media_mixer.mix_media.remote(
                sentences_with_timestamps,
                task_state
            )
            
            if not output_path:
                self.logger.warning(f"[{task_id}][分段{seg_idx}] 媒体混合未返回有效路径")
                return
            
            # 更新处理状态
            task_state.merged_segments.append(output_path)
            task_state.batch_counter += 1
            
            # 6. HLS处理
            hls_add_result = await self.hls_manager.add_segment.remote(
                task_id, 
                output_path, 
                task_state.batch_counter
            )
            
            if hls_add_result["status"] == "success":
                is_first_hls = not task_state.hls_ready
                task_state.hls_ready = True
                await self.state_manager.update_task_progress.remote(
                    task_id, 
                    batch_counter=task_state.batch_counter,
                    hls_ready=task_state.hls_ready
                )
                if is_first_hls:
                    self.logger.info(f"[{task_id}] HLS播放就绪")
            
            # 显式删除无需继续使用的大型变量
            del tts_token_sentences
            del aligned_sentences
            del audio_gen_sentences
            del sentences_with_timestamps
            
        except Exception as e:
            self.logger.error(f"[{task_id}][分段{seg_idx}] 音频生成和混合失败: {str(e)}")
        finally:
            # 确保清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
