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
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# Ray 和 Serve
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

# 项目模块
from config import Config, init_logging
from core.hls_manager import HLSManager
from core.video_separator import VideoSeparator
from core.asr_model import ASRModel
from core.translation.translator import Translator
from core.my_index_tts import MyIndexTTSDeployment
from core.timeadjust.duration_aligner import DurationAligner
from core.timeadjust.timestamp_adjuster import TimestampAdjuster
from core.media_mixer import MediaMixer
from utils.task_storage import TaskPaths
from utils.ffmpeg_utils import concat_videos, get_duration
from core.supabase_client import SupabaseClient
from core.sentence_tools import Sentence # 添加 Sentence 导入

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

# --- 分阶段流水线部署 ---
@serve.deployment(
    name="PreprocessingPipe",
    num_replicas=1,
    max_ongoing_requests=global_config.MAX_PARALLEL_SEGMENTS,
    ray_actor_options={"num_cpus": 0.5},
    logging_config={"log_level": "INFO"}
)
class PreprocessingPipe:
    """分阶段流水线：预处理阶段，负责视频分离和ASR"""
    def __init__(self, video_separator_handle: DeploymentHandle, asr_model_handle: DeploymentHandle):
        self.logger = logger
        self.supabase_client = SupabaseClient(config=global_config)
        self.video_separator = video_separator_handle
        self.asr = asr_model_handle
        self.config = global_config

    async def __call__(self, task_id: str, video_path: str, target_language: str, generate_subtitle: bool):
        self.logger.info(f"[{task_id}] Starting preprocessing stage.")
        # 初始化任务状态
        init_success = await self._init_task_state(task_id, video_path, target_language, generate_subtitle)
        if not init_success:
            self.logger.error(f"[{task_id}] Task state initialization failed.")
            return {"status": "error", "message": "Task state initialization failed."}

        # 视频分离与ASR
        media_files = await self._run_sep_asr(task_id, video_path)
        if media_files is None:
            final_task = await self.supabase_client.get_task(task_id) if self.supabase_client else {}
            status_msg = final_task.get('status', 'error')
            error_msg = final_task.get('error_message', 'Preprocessing failed')
            return {"status": status_msg, "message": error_msg}

        self.logger.info(f"[{task_id}] Preprocessing completed.")
        return {"status": "preprocessed", "message": "Preprocessing finished"}

    async def _init_task_state(self, task_id, video_path, target_language, generate_subtitle) -> bool:
        try:
            if not video_path or not target_language:
                raise ValueError("缺少 video_path 或 target_language")

            # 1. 创建任务路径对象和目录
            task_paths = TaskPaths(self.config, task_id)
            await asyncio.to_thread(task_paths.create_directories)
            self.logger.info(f"[{task_id}] 任务目录已创建: {task_paths.task_dir}")

            # 2. 在 Supabase 中创建或更新任务记录
            existing_task = await self.supabase_client.get_task(task_id)
            task_data = {
                'task_id': task_id,
                'status': 'preprocessing',
                'target_language': target_language,
                'generate_subtitle': generate_subtitle,
                'original_video_path': str(video_path),
            }
            if existing_task:
                await self.supabase_client.update_task(task_id, task_data)
                self.logger.info(f"[{task_id}] 更新已存在的任务记录，状态: {task_data['status']}")
            else:
                response = await self.supabase_client.store_task(task_data)
                if not response or not response.data:
                    raise Exception("存储初始任务到 Supabase 失败")
                self.logger.info(f"[{task_id}] 创建新任务记录，状态: {task_data['status']}")

            return True
        except Exception as e:
            self.logger.error(f"[{task_id}] 任务初始化失败: {e}")
            if task_id and self.supabase_client:
                await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f"任务初始化失败: {e}"})
            return False

    async def _run_sep_asr(self, task_id: str, video_path: str) -> Optional[Dict]:
        seg_start_time = time.time()
        self.logger.info(f"[{task_id}] 开始处理视频")
        try:
            # 创建任务路径对象
            task_paths = TaskPaths(self.config, task_id)
            # 跳过已预处理任务
            current_task = await self.supabase_client.get_task(task_id)
            if current_task and current_task.get('status') == 'preprocessed':
                self.logger.info(f"[{task_id}] 已预处理完成，跳过分离和ASR")
                return {
                    'silent_video_path': current_task.get('silent_video_path'),
                    'vocals_audio_path': current_task.get('vocals_audio_path'),
                    'background_audio_path': current_task.get('background_audio_path')
                }

            # 1. 视频分离
            duration = await get_duration(video_path)
            self.logger.info(f"[{task_id}] 视频时长={duration:.2f}s，开始分离")
            media_files = await self.video_separator.separate_video.remote(
                video_path,
                str(task_paths.media_dir),
                self.config.TARGET_SR
            )
            if not media_files or "vocals_audio_path" not in media_files or not Path(media_files["vocals_audio_path"]).exists():
                self.logger.warning(f"[{task_id}] 视频分离失败或无人声检测失败")
                await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': '视频分离失败或无人声'})
                return None
            # 更新分离结果路径
            await self.supabase_client.update_task(task_id, {**media_files, 'status': 'preprocessing'})
            self.logger.info(f"[{task_id}] 视频分离完成: {media_files}")

            # 2. ASR 识别
            sentences = await self.asr.generate.remote(
                input=media_files["vocals_audio_path"],
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
                task_id=task_id,
                task_paths=task_paths
            )
            if not sentences:
                self.logger.info(f"[{task_id}] ASR 未检测到语音")
                await self.supabase_client.update_task(task_id, {'status': 'preprocessed', 'error_message': 'ASR 未检测到语音'})
                return media_files

            # 3. 保存句子并更新状态
            response = await self.supabase_client.store_sentences(sentences, task_id)
            if not response or not response.data:
                raise Exception("存储句子到 Supabase 失败")
            await self.supabase_client.update_task(task_id, {'status': 'preprocessed'})
            self.logger.info(f"[{task_id}] 预处理完成，共 {len(sentences)} 个句子")
            return media_files
        except Exception as e:
            self.logger.exception(f"[{task_id}] 分离/ASR 发生错误: {e}")
            await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f"分离/ASR 错误: {e}"})
            return None
        finally:
            self._clean_memory()
            self.logger.info(f"[{task_id}] 分离和ASR耗时: {time.time() - seg_start_time:.2f}s")

    def _clean_memory(self) -> None:
        """清理内存和GPU缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@serve.deployment(
    name="TranslationPipe",
    num_replicas=1,
    max_ongoing_requests=global_config.MAX_PARALLEL_SEGMENTS,
    ray_actor_options={"num_cpus": 0.5},
    logging_config={"log_level": "INFO"}
)
class TranslationPipe:
    """分阶段流水线：翻译与合成阶段"""
    def __init__(self, translator_handle: DeploymentHandle, my_index_tts_handle: DeploymentHandle, duration_aligner_handle: DeploymentHandle, timestamp_adjuster_handle: DeploymentHandle, media_mixer_handle: DeploymentHandle, hls_manager_handle: DeploymentHandle):
        self.logger = logger
        self.supabase_client = SupabaseClient(config=global_config)
        self.config = global_config
        self.translator = translator_handle.options(stream=True)
        self.my_index_tts = my_index_tts_handle.options(stream=True)
        self.duration_aligner = duration_aligner_handle
        self.timestamp_adjuster = timestamp_adjuster_handle
        self.media_mixer = media_mixer_handle
        self.hls_manager = hls_manager_handle

    async def translate_task(self, task_id: str):
        self.logger.info(f"[{task_id}] Starting translation stage.")
        try:
            # 初始化HLS管理器
            init_ok = await self._init_hls(task_id)
            if not init_ok:
                return {"status": "error", "message": "HLS initialization failed."}

            # 运行翻译 -> 合成 -> HLS
            merged_segments = await self._run_translation_pipeline(task_id)
            result = await self._merge_segments(task_id, merged_segments)
            self.logger.info(f"[{task_id}] Translation finished with status: {result.get('status')}")
            return result
        except Exception as e:
            self.logger.exception(f"[{task_id}] Translation stage failed: {e}")
            await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f"Translation failed: {e}"})
            return {"status": "error", "message": f"Translation failed: {e}"}
        finally:
            self._clean_memory()

    async def _init_hls(self, task_id: str) -> bool:
        """初始化HLS管理器"""
        try:
            # 在内部创建 task_paths
            task_paths = TaskPaths(self.config, task_id)
            
            result = await self.hls_manager.create_manager.remote(task_id, task_paths)
            if isinstance(result, dict) and result.get("status") == "error":
                raise RuntimeError(f"HLS管理器创建失败: {result.get('message')}")
            self.logger.info(f"[{task_id}] HLS管理器初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"[{task_id}] HLS管理器初始化失败: {e}")
            if self.supabase_client:
                await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f"HLS初始化失败: {e}"})
            return False

    async def _run_translation_pipeline(self, task_id: str) -> List[str]:
        """运行从翻译到媒体混合的子流水线 (从数据库获取依赖信息)"""
        processed_tts_batches = 0
        added_hls_segments = 0
        start_time = time.time()
        current_time = 0.0
        merged_segments_paths = []
        batch_counter = 0

        # --- 从数据库获取所需信息 ---
        if not self.supabase_client:
            self.logger.error(f"[{task_id}] Supabase客户端未初始化，无法进行翻译处理")
            return []

        try:
            task_data = await self.supabase_client.get_task(task_id)
            if not task_data:
                self.logger.error(f"[{task_id}] 无法从数据库获取任务信息")
                return []

            target_language = task_data.get('target_language')
            generate_subtitle = task_data.get('generate_subtitle', False) # 提供默认值
            silent_video_path = task_data.get('silent_video_path')
            vocals_audio_path = task_data.get('vocals_audio_path')
            background_audio_path = task_data.get('background_audio_path')

            if not all([target_language, silent_video_path, vocals_audio_path]): # 背景音是可选的
                self.logger.error(f"[{task_id}] 数据库中缺少必要的任务信息 (语言、无声视频路径、人声路径)")
                await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': '数据库信息不完整'})
                return []

            # 重建 media_files 字典
            media_files = {
                'silent_video_path': silent_video_path,
                'vocals_audio_path': vocals_audio_path,
                'background_audio_path': background_audio_path # 可能为 None
            }
            # 重建 task_paths 对象
            task_paths = TaskPaths(self.config, task_id)

        except Exception as e:
            self.logger.error(f"[{task_id}] 获取或重建任务依赖信息失败: {e}", exc_info=True)
            await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f'获取任务依赖失败: {e}'})
            return []
        # --- 信息获取结束 ---


        # 1. 检查前置条件 (已在上面检查 supabase_client)
        # if not self.supabase_client:
        #     self.logger.error(f"[{task_id}] Supabase客户端未初始化，无法进行翻译处理")
        #     return []

        # 2. 获取句子 (这部分逻辑不变)
        try:
            sentences = await self.supabase_client.get_sentences(task_id, as_objects=True)
            if not sentences:
                self.logger.warning(f"[{task_id}] 数据库中没有检索到句子")
                # 如果没有句子，也认为翻译流程"完成"了，只是没有生成片段
                # 后续的 _merge_segments 会处理空列表
                return []
            self.logger.info(f"[{task_id}] 获取到{len(sentences)}个句子，开始翻译流程")
        except Exception as e:
            self.logger.error(f"[{task_id}] 获取句子失败: {e}")
            await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f"获取句子失败: {e}"})
            return []

        # 3. 主要处理流程 (后续逻辑使用上面获取或重建的变量)
        try:
            # 更新状态为翻译中
            await self.supabase_client.update_task(task_id, {'status': 'translating'})

            # 翻译流程 (按批次进行)
            async for translated_batch in self.translator.translate_sentences.remote(
                sentences,
                batch_size=int(self.config.TRANSLATION_BATCH_SIZE),
                target_language=target_language # <--- 使用获取到的 target_language
            ):
                if not translated_batch:
                    continue

                # TTS生成语音
                async for tts_batch in self.my_index_tts.generate_audio_stream.remote(translated_batch):
                    if not tts_batch:
                        continue

                    processed_tts_batches += 1
                    
                    # 时长对齐和调整
                    aligned_batch = await self.duration_aligner.remote(tts_batch, max_speed=1.5)
                    if not aligned_batch:
                        continue

                    adjusted_batch = await self.timestamp_adjuster.remote(
                        aligned_batch,
                        self.config.TARGET_SR,
                        current_time
                    )
                    if not adjusted_batch:
                        continue

                    # 更新时间戳位置和处理状态
                    current_time = adjusted_batch[-1].adjusted_start + adjusted_batch[-1].adjusted_duration
                    
                    # 第一个批次完成后，更新状态为 mixing
                    status_update = 'mixing' if processed_tts_batches == 1 else None
                    if status_update:
                        await self.supabase_client.update_task(task_id, {'status': status_update})

                    # 媒体混合 (使用重建的 media_files 和 task_paths)
                    output_path = await self.media_mixer.mix_media.remote(
                         adjusted_batch,
                         media_files=media_files, # <--- 使用重建的 media_files
                         task_paths=task_paths,   # <--- 使用重建的 task_paths
                         generate_subtitle=generate_subtitle, # <--- 使用获取到的 generate_subtitle
                         batch_counter=batch_counter,
                         task_id=task_id,
                         target_language=target_language
                     )
                    if not output_path:
                        self.logger.warning(f"[{task_id}] 媒体混合失败，跳过此批次")
                        continue

                    # HLS处理
                    hls_result = await self.hls_manager.add_segment.remote(
                        task_id,
                        output_path,
                        batch_counter + 1
                    )

                    if hls_result and hls_result.get("status") == "success":
                        added_hls_segments += 1
                        merged_segments_paths.append(output_path)
                        batch_counter += 1
                        self.logger.info(f"[{task_id}] HLS片段 {batch_counter} 添加成功")

                        # --- 新增：首次添加成功时，更新 hls_playlist_url ---
                        if added_hls_segments == 1 and self.supabase_client:
                            hls_relative_path = f"playlists/{task_id}/{task_paths.playlist_path.name}"
                            try:
                                await self.supabase_client.update_task(task_id, {
                                    'hls_playlist_url': hls_relative_path,
                                    # 'status': 'generating_hls' # 移除此行
                                })
                                self.logger.info(f"[{task_id}] HLS播放列表URL已更新到数据库: {hls_relative_path}")
                            except Exception as update_e:
                                self.logger.error(f"[{task_id}] 更新HLS播放列表URL到数据库失败: {update_e}")
                        # --- 新增结束 ---

                    else:
                        error_msg = hls_result.get('message') if hls_result else '未知错误'
                        self.logger.error(f"[{task_id}] 添加HLS片段失败: {error_msg}")

                    # 清理内存
                    self._clean_memory()

            self.logger.info(f"[{task_id}] 翻译流程完成，耗时: {time.time() - start_time:.2f}s, "
                            f"TTS批次: {processed_tts_batches}, HLS段: {added_hls_segments}")
            return merged_segments_paths

        except Exception as e:
            self.logger.exception(f"[{task_id}] 翻译流程异常: {e}")
            await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f"翻译流程错误: {e}"})
            return []

    async def _merge_segments(self, task_id: str, merged_segments: List[str]) -> Dict:
        """合并处理好的视频片段"""
        self.logger.info(f"[{task_id}] 开始合并 {len(merged_segments)} 个视频片段")

        # 在内部创建 task_paths
        task_paths = TaskPaths(self.config, task_id)
        
        # 处理无片段情况
        if not merged_segments:
            msg = "没有处理成功的视频片段可以合并"
            if self.supabase_client:
                await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': msg})
            return {"status": "error", "message": msg}

        try:
            # 1. 创建合并列表文件
            list_txt_path = task_paths.processing_dir / "concat_list.txt"
            async with aiofiles.open(list_txt_path, "w", encoding='utf-8') as f:
                for seg_mp4 in merged_segments:
                    formatted_path = str(Path(seg_mp4).resolve()).replace("\\", "/")
                    await f.write(f"file '{formatted_path}'\n")

            # 2. 执行视频合并
            final_output_path = task_paths.output_dir / f"final_{task_id}.mp4"
            merge_start_time = time.time()
            final_video_path = await concat_videos(str(list_txt_path), str(final_output_path))

            # 3. 处理合并结果
            if not final_video_path or not final_video_path.exists():
                msg = "视频合并失败，最终文件未生成"
                self.logger.error(f"[{task_id}] {msg}")
                if self.supabase_client:
                    await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': msg})
                return {"status": "error", "message": msg}

            # 4. 完成流程并更新状态
            final_video_path_str = str(final_video_path)
            merge_duration = time.time() - merge_start_time
            
            # 完成HLS列表
            await self.hls_manager.finalize_playlist.remote(task_id)
            
            if self.supabase_client:
                await self.supabase_client.update_task(task_id, {
                    'status': 'success',
                    'download_video_path': final_video_path_str,
                })
            
            self.logger.info(f"[{task_id}] 视频合并成功，耗时: {merge_duration:.2f}s")
            return {"status": "success", "message": "视频处理成功", "output_path": final_video_path_str}
            
        except Exception as e:
            msg = f"视频合并过程中出错: {e}"
            self.logger.error(f"[{task_id}] {msg}")
            if self.supabase_client:
                await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': msg})
            return {"status": "error", "message": msg}

    def _clean_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        
        # 绑定并部署分阶段流水线服务（预处理和翻译）
        preprocessing_pipe = PreprocessingPipe.bind(
            video_separator_handle=video_separator_handle,
            asr_model_handle=asr_handle
        )
        translation_pipe = TranslationPipe.bind(
            translator_handle=translator_handle,
            my_index_tts_handle=my_index_tts_handle,
            duration_aligner_handle=duration_aligner_handle,
            timestamp_adjuster_handle=timestamp_adjuster_handle,
            media_mixer_handle=media_mixer_handle,
            hls_manager_handle=hls_manager_handle
        )
        # 单独部署预处理和翻译两个应用
        serve.run(preprocessing_pipe, name="PreprocessingEngine", route_prefix=None)
        serve.run(translation_pipe,   name="TranslationEngine",   route_prefix=None)
        
        logger.info("流水线服务设置完成")
        return {"status": "deployed", "application": "PipelineEngine"}

    except Exception as e:
        logger.critical(f"设置流水线服务失败: {e}", exc_info=True)
        return None

# --- 脚本入口 ---
if __name__ == "__main__":
    setup_pipeline_services()