# pipeline_scheduler.py (精简版)
import logging
import asyncio
import time
import sys
import os
backend_dir = os.path.dirname(os.path.abspath(__file__))
# IndexTTS 包所在的目录是 backend/models/IndexTTS
index_tts_dir = os.path.join(backend_dir, 'models', 'IndexTTS')

# 将包含 indextts 包的目录添加到 sys.path
if index_tts_dir not in sys.path:
    sys.path.insert(0, index_tts_dir) # 插入到最前面，优先搜索


import aiofiles
import torch
import gc
from typing import List, Optional, Dict, Any, AsyncGenerator
from pathlib import Path

# Ray 和 Serve
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

# 项目模块
from config import Config, init_logging
from core.state_manager import StateManager
from core.hls_manager import HLSManager
from core.video_separator import VideoSeparator
from core.asr_model import ASRModel
from core.translation.translator import Translator
from core.my_index_tts import MyIndexTTSDeployment
from core.timeadjust.duration_aligner import DurationAligner
from core.timeadjust.timestamp_adjuster import TimestampAdjuster
from core.media_mixer import MediaMixer
from utils.task_state import TaskState
from utils.ffmpeg_utils import concat_videos, get_duration

# 初始化全局日志配置
init_logging()

logger = logging.getLogger(__name__)

# --- 全局配置 ---
# 确保 Config 类被正确导入并实例化
try:
    global_config = Config()
    global_config.init_directories() # 初始化存储目录

except ImportError:
    logger.critical("无法导入 Config 类或初始化目录，请确保 config.py 文件存在且正确。")
    sys.exit(1)
except Exception as e:
    logger.critical(f"初始化 Config 或目录时出错: {e}", exc_info=True)
    sys.exit(1)


# --- Ray Serve 部署句柄创建 ---
# 确保为每个部署提供唯一的 name
# --- Ray Serve 部署句柄创建 ---
# 统一 handle 命名，确保后续可读性和一致性
try:
    state_manager_handle = StateManager.options(name="StateManager", num_replicas=1, ray_actor_options={"num_cpus": 0.25}).bind(global_config)
    hls_manager_handle = HLSManager.options(name="hls_manager", num_replicas=1, ray_actor_options={"num_cpus": 0.5}).bind()
    video_separator_handle = VideoSeparator.options(name="video_separator", num_replicas=1, max_ongoing_requests=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.2}).bind()
    asr_handle = ASRModel.options(name="asr_model", num_replicas=1, max_ongoing_requests=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.3}).bind()
    translator_handle = Translator.options(name="translator", num_replicas=1, max_ongoing_requests=3, ray_actor_options={"num_cpus": 0.5}).bind()
    simplifier_handle = Translator.options(name="simplifier", num_replicas=1, ray_actor_options={"num_cpus": 0.5}).bind()
    my_index_tts_handle = MyIndexTTSDeployment.options(name="my_index_tts", num_replicas=1, max_ongoing_requests=2, ray_actor_options={"num_cpus": 1, "num_gpus": 0.5}).bind(global_config)
    # DurationAligner 依赖 simplifier 和 TTS 句柄
    duration_aligner_handle = DurationAligner.options(name="duration_aligner", num_replicas=1, ray_actor_options={"num_cpus": 0.25}).bind(simplifier_handle, my_index_tts_handle)
    timestamp_adjuster_handle = TimestampAdjuster.options(name="timestamp_adjuster", num_replicas=1, ray_actor_options={"num_cpus": 0.5}).bind()
    media_mixer_handle = MediaMixer.options(name="media_mixer", num_replicas=1, ray_actor_options={"num_cpus": 0.5}).bind()
except Exception as e:
    logger.critical(f"创建 Ray Serve 部署句柄时出错: {e}", exc_info=True)
    sys.exit(1)

# --- 主管道部署 ---
@serve.deployment(
    name="VideoTransPipe",
    num_replicas=1, # 单副本以简化状态管理和调试
    max_ongoing_requests=global_config.MAX_PARALLEL_SEGMENTS, # 控制并发任务数
    ray_actor_options={"num_cpus": 0.5},
    logging_config={"log_level": "INFO"} # 控制 Serve 日志级别
)
class VideoTransPipe:
    """视频翻译流水线主协调器 (重构版)"""
    def __init__(self, **handles: DeploymentHandle):
        """初始化视频翻译流水线协调器"""
        self.config = global_config
        self.sample_rate = self.config.TARGET_SR
        self.logger = logger
        
        # 获取所有必要的服务句柄
        handle_mapping = {
            "StateManager_handle": "state_manager",
            "hls_manager_handle": "hls_manager", 
            "video_separator_handle": "video_separator",
            "asr_model_handle": "asr",
            "translator_handle": "translator",
            "simplifier_handle": "simplifier",
            "my_index_tts_handle": "my_index_tts",
            "duration_aligner_handle": "duration_aligner",
            "timestamp_adjuster_handle": "timestamp_adjuster",
            "media_mixer_handle": "media_mixer"
        }
        
        # 批量设置句柄属性
        for handle_key, attr_name in handle_mapping.items():
            handle = handles.get(handle_key)
            if not handle:
                raise ValueError(f"缺少必要的句柄: {handle_key}")
            setattr(self, attr_name, handle)
        
        # 设置流式属性
        self.translator = self.translator.options(stream=True)
        self.simplifier = self.simplifier.options(stream=True)
        self.my_index_tts = self.my_index_tts.options(stream=True)
        
        self.logger.info("VideoTransPipe 初始化完成")

    async def __call__(self, task_id: str, video_path: str = None, target_language: str = None, generate_subtitle: bool = False) -> Dict[str, Any]:
        """处理单个视频翻译任务"""
        task_state: Optional[TaskState] = None
        start_pipeline_time = time.time()
        self.logger.info(f"[{task_id}] 开始处理视频任务。语言: {target_language}, 字幕: {generate_subtitle}")
        
        try:
            # 初始化任务状态和HLS
            task_state = await self._init_task_state(task_id, video_path, target_language, generate_subtitle)
            if not task_state:
                return {"status": "error", "message": "任务状态初始化失败"}
                
            if not await self._init_hls(task_id, task_state):
                return {"status": "error", "message": "HLS管理器初始化失败"}
                
            # 处理视频并合并结果
            await self._run_sep_asr(task_id, task_state)
            return await self._merge_segments(task_id, task_state)
            
        except Exception as e:
            self.logger.exception(f"[{task_id}] 任务处理失败: {e}")
            if task_id and task_state:
                await self.state_manager.complete_task.remote(task_id, False, f"处理失败: {e}")
            return {"status": "error", "message": f"处理失败: {e}"}
            
        finally:
            # 清理并记录总耗时
            pipeline_end_time = time.time()
            self.logger.info(f"[{task_id}] 任务处理总耗时: {pipeline_end_time - start_pipeline_time:.2f}s")
            
            if task_state:
                if hasattr(task_state, 'media_files'): task_state.media_files = {}
                if hasattr(task_state, 'merged_segments'): task_state.merged_segments.clear()
            
            self._clean_memory()

    async def _init_task_state(self, task_id, video_path, target_language, generate_subtitle):
        """初始化或获取任务状态"""
        try:
            if video_path and target_language:
                # 创建新任务
                task_data = await self.state_manager.create_task.remote(task_id, video_path, target_language, generate_subtitle)
                task_state = task_data.get("task_state") if isinstance(task_data, dict) else None
            else:
                # 获取已有任务状态
                task_state = await self.state_manager.get_task_state.remote(task_id)
                
            if not task_state:
                raise ValueError("无法获取或创建任务状态")
                
            return task_state
            
        except Exception as e:
            self.logger.error(f"[{task_id}] 任务状态初始化失败: {e}")
            if task_id:
                await self.state_manager.complete_task.remote(task_id, False, f"任务状态初始化失败: {e}")
            return None

    async def _init_hls(self, task_id, task_state):
        """初始化HLS管理器"""
        try:
            result = await self.hls_manager.create_manager.remote(task_id, task_state.task_paths)
            if isinstance(result, dict) and result.get("status") == "error":
                raise RuntimeError(f"HLS管理器创建失败: {result.get('message')}")
            return True
        except Exception as e:
            self.logger.error(f"[{task_id}] HLS管理器初始化失败: {e}")
            await self.state_manager.complete_task.remote(task_id, False, f"HLS管理器初始化失败: {e}")
            return False

    async def _run_sep_asr(self, task_id, task_state):
        """分离视频并执行语音识别"""
        seg_start_time = time.time()
        self.logger.info(f"[{task_id}] 开始处理视频")
        
        try:
            # 1. 获取视频时长和分离音视频
            duration = await get_duration(task_state.video_path)
            self.logger.info(f"[{task_id}] 处理视频，时长={duration:.2f}s")
            
            media_files = await self.video_separator.separate_video.remote(
                task_state.video_path,
                str(task_state.task_paths.media_dir),
                self.config.TARGET_SR
            )
            
            if not media_files or "vocals" not in media_files or not Path(media_files["vocals"]).exists():
                self.logger.warning(f"[{task_id}] 视频分离失败或无有效人声，跳过处理")
                return
                
            # 记录分离结果
            task_state.media_files = media_files
            
            # 2. 执行ASR识别
            sentences = await self.asr.generate.remote(
                input=media_files["vocals"],
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
                task_id=task_id,
                task_paths=task_state.task_paths
            )
            
            if not sentences:
                self.logger.info(f"[{task_id}] ASR未检测到语音，跳过后续处理")
                return
                
            # 3. 更新句子计数并运行翻译流水线
            self.logger.info(f"[{task_id}] ASR完成: {len(sentences)}个句子")
            
            # 4. 运行翻译和TTS流水线
            pipeline_info = await self._run_translation_pipeline(sentences, task_state)
            self.logger.info(f"[{task_id}] 子流水线处理完成: {pipeline_info}")
            
        except Exception as e:
            self.logger.exception(f"[{task_id}] 处理视频时发生错误: {e}")
        finally:
            self._clean_memory()
            self.logger.info(f"[{task_id}] 视频处理耗时: {time.time() - seg_start_time:.2f}s")

    async def _merge_segments(self, task_id, task_state):
        """合并处理好的视频片段"""
        self.logger.info(f"[{task_id}] 开始合并 {len(task_state.merged_segments)} 个视频片段")
        
        # 检查是否有可合并的片段
        if not task_state.merged_segments:
            msg = "没有处理成功的视频片段可以合并"
            await self.state_manager.complete_task.remote(task_id, False, msg)
            return {"status": "error", "message": msg}
        
        # 准备合并列表文件
        list_txt_path = task_state.task_paths.processing_dir / "concat_list.txt"
        async with aiofiles.open(list_txt_path, "w", encoding='utf-8') as f:
            for seg_mp4 in task_state.merged_segments:
                formatted_path = str(Path(seg_mp4).resolve()).replace("\\", "/")
                await f.write(f"file '{formatted_path}'\n")
        
        # 执行视频合并
        final_output_path = task_state.task_paths.output_dir / f"final_{task_id}.mp4"
        merge_start_time = time.time()
        final_video_path = await concat_videos(str(list_txt_path), str(final_output_path))
        
        # 检查合并结果
        if final_video_path and final_video_path.exists():
            final_video_path_str = str(final_video_path)
            # 完成HLS列表和任务状态更新
            await self.hls_manager.finalize_playlist.remote(task_id)
            await self.state_manager.complete_task.remote(task_id, True, "视频处理成功", final_video_path_str)
            
            self.logger.info(f"[{task_id}] 视频合并成功，耗时: {time.time() - merge_start_time:.2f}s")
            return {"status": "success", "message": "视频处理成功", "output_path": final_video_path_str}
        else:
            msg = "视频合并失败，最终文件未生成"
            await self.state_manager.complete_task.remote(task_id, False, msg)
            return {"status": "error", "message": msg}

    def _clean_memory(self) -> None:
        """清理内存和GPU缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def _run_translation_pipeline(self, sentences: List, task_state: TaskState) -> Dict:
        """运行从翻译到媒体混合的子流水线"""
        task_id = task_state.task_id
        processed_tts_batches = 0
        added_hls_segments = 0
        start_time = time.time()

        try:
            # 1. 翻译 (流式)
            async for translated_batch in self.translator.translate_sentences.remote(
                sentences, 
                batch_size=int(self.config.TRANSLATION_BATCH_SIZE),
                target_language=task_state.target_language 
            ):
                if not translated_batch: 
                    continue
                    
                # 2. TTS (流式)
                async for tts_batch in self.my_index_tts.generate_audio_stream.remote(translated_batch):
                    if not tts_batch: 
                        continue
                        
                    processed_tts_batches += 1
                    
                    # 3. 时长对齐
                    aligned_batch = await self.duration_aligner.remote(tts_batch, max_speed=1.5)
                    if not aligned_batch: 
                        continue

                    # 4. 时间戳调整
                    adjusted_batch = await self.timestamp_adjuster.remote(
                        aligned_batch, 
                        self.config.TARGET_SR, 
                        task_state.current_time
                    )
                    if not adjusted_batch: 
                        continue
                        
                    # 更新当前时间戳
                    task_state.current_time = adjusted_batch[-1].adjusted_start + adjusted_batch[-1].adjusted_duration

                    # 5. 媒体混合
                    output_path = await self.media_mixer.mix_media.remote(adjusted_batch, task_state)
                    if not output_path:
                        self.logger.warning(f"[{task_id}] 媒体混合失败，跳过此批次")
                        continue

                    # 6. HLS处理
                    hls_result = await self.hls_manager.add_segment.remote(
                        task_id, 
                        output_path, 
                        task_state.batch_counter + 1
                    )
                    
                    if hls_result["status"] == "success":
                        added_hls_segments += 1
                        task_state.merged_segments.append(output_path)
                        task_state.batch_counter += 1
                        
                        # 处理HLS就绪状态
                        if not task_state.hls_ready:
                            task_state.hls_ready = True
                            await self.state_manager.update_task_progress.remote(
                                task_id, 
                                batch_counter=task_state.batch_counter, 
                                hls_ready=True
                            )
                            self.logger.info(f"[{task_id}] HLS播放就绪")
                        else:
                            await self.state_manager.update_task_progress.remote(
                                task_id, 
                                batch_counter=task_state.batch_counter
                            )
                    else:
                        self.logger.error(f"[{task_id}] 添加HLS片段失败: {hls_result.get('message')}")
                    
                    # 清理每次循环的变量
                    self._clean_memory()

        except Exception as e:
            self.logger.exception(f"[{task_id}] 子流水线异常: {e}")

        self.logger.info(f"[{task_id}] 子流水线完成，耗时: {time.time() - start_time:.2f}s, TTS批次: {processed_tts_batches}, HLS段: {added_hls_segments}")
        return {"processed_tts_batches": processed_tts_batches, "added_hls_segments": added_hls_segments}

# --- Ray 和 Serve 初始化与服务部署 ---
def init_ray(address="auto", namespace="videotrans", log_to_driver=True):
    """初始化或连接到Ray集群"""
    if ray.is_initialized():
        try:
            ray.cluster_resources()  # 检查连接是否有效
            logger.info("已连接到Ray集群")
            return True
        except Exception:
            logger.warning("Ray连接无效，重新初始化")
            ray.shutdown()

    try:
        ray.init(
            address=address, 
            namespace=namespace, 
            log_to_driver=log_to_driver, 
            ignore_reinit_error=True
        )
        logger.info(f"Ray初始化成功: {ray.get_runtime_context().gcs_address}")
        return True
    except Exception as e:
        logger.error(f"Ray初始化失败: {e}")
        return False

def start_serve(detached=False, http_host="0.0.0.0", http_port=8000):
    """启动Ray Serve服务"""
    try:
        serve.start(
            detached=detached, 
            http_options={"host": http_host, "port": http_port}
        )
        logger.info(f"Ray Serve已启动: http://{http_host}:{http_port}")
        return True
    except Exception as e:
        logger.error(f"Ray Serve启动失败: {e}")
        return False

def setup_pipeline_services():
    """设置和部署VideoTrans流水线服务"""
    try:
        # 初始化Ray和Serve
        if not init_ray():
            logger.critical("无法初始化Ray集群")
            return None
            
        if not start_serve():
            logger.critical("无法启动Ray Serve")
            return None

        logger.info("准备部署流水线应用...")
        
        # 收集所有服务句柄
        handles_to_deploy = {
            "StateManager": state_manager_handle,
            "hls_manager": hls_manager_handle,
            "video_separator": video_separator_handle,
            "asr_model": asr_handle,
            "translator": translator_handle,
            "simplifier": simplifier_handle,
            "my_index_tts": my_index_tts_handle,
            "duration_aligner": duration_aligner_handle,
            "timestamp_adjuster": timestamp_adjuster_handle,
            "media_mixer": media_mixer_handle,
        }

        # 部署主管道应用 (会自动部署依赖项)
        pipeline_app = VideoTransPipe.bind(**{name + "_handle": handle for name, handle in handles_to_deploy.items()})
        serve.run(pipeline_app, name="PipelineEngine", route_prefix=None)
        
        logger.info("主管道应用已部署，等待服务稳定...")
        time.sleep(10)  # 等待服务稳定

        # 检查部署状态
        apps_status = serve.status().applications
        if "PipelineEngine" in apps_status:
            app_status = apps_status["PipelineEngine"]
            logger.info(f"应用状态: PipelineEngine - {app_status.status}")
        else:
            logger.warning("PipelineEngine应用尚未就绪")

        logger.info("流水线服务设置完成")
        return {"status": "deployed", "application": "PipelineEngine"}

    except Exception as e:
        logger.critical(f"设置流水线服务失败: {e}", exc_info=True)
        return None

# --- 脚本入口 ---
if __name__ == "__main__":
    setup_pipeline_services()