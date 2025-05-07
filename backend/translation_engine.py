import logging
import asyncio
import time
import sys
import os
import gc
import torch
import aiofiles
from typing import List, Optional, Dict, Any
from pathlib import Path

# Ray 和 Serve
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

# 项目模块
from config import Config, init_logging
from core.hls_manager import HLSManager
from core.translation.translator import Translator
from core.my_index_tts import MyIndexTTSDeployment
from core.timeadjust.duration_aligner import DurationAligner
from core.timeadjust.timestamp_adjuster import TimestampAdjuster
from core.media_mixer import MediaMixer
from utils.task_storage import TaskPaths
from utils.ffmpeg_utils import concat_videos
from core.supabase_client import SupabaseClient

# 初始化全局日志配置
init_logging()

logger = logging.getLogger(__name__)

# --- 全局配置 ---
global_config = Config()
# global_config.init_directories() # Removed: Launcher will handle this

# --- Ray Serve 翻译相关部署句柄创建 ---
try:
    hls_manager_handle = HLSManager.options(name="hls_manager", num_replicas=1, ray_actor_options={"num_cpus": 0.5}).bind()
    translator_handle = Translator.options(name="translator", num_replicas=1, max_ongoing_requests=3, ray_actor_options={"num_cpus": 0.5}).bind()
    simplifier_handle = Translator.options(name="simplifier", num_replicas=1, ray_actor_options={"num_cpus": 0.5}).bind()
    my_index_tts_handle = MyIndexTTSDeployment.options(name="my_index_tts", num_replicas=1, max_ongoing_requests=2, ray_actor_options={"num_cpus": 1, "num_gpus": 0.5}).bind(global_config)
    # DurationAligner 依赖 simplifier 和 TTS 句柄
    duration_aligner_handle = DurationAligner.options(name="duration_aligner", num_replicas=1, ray_actor_options={"num_cpus": 0.25}).bind(simplifier_handle, my_index_tts_handle)
    timestamp_adjuster_handle = TimestampAdjuster.options(name="timestamp_adjuster", num_replicas=1, ray_actor_options={"num_cpus": 0.5}).bind()
    media_mixer_handle = MediaMixer.options(name="media_mixer", num_replicas=1, ray_actor_options={"num_cpus": 0.5}).bind()
except Exception as e:
    logger.critical(f"创建翻译相关 Ray Serve 部署句柄时出错: {e}", exc_info=True)
    sys.exit(1)

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
        tts_batches = 0
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

                    tts_batches += 1
                    
                    # 时长对齐和调整
                    aligned_batch = await self.duration_aligner.remote(tts_batch, max_speed=1.2)
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
                    status_update = 'mixing' if tts_batches == 1 else None
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

                        # --- 首次添加成功时，更新 hls_playlist_url ---
                        if added_hls_segments == 1 and self.supabase_client:
                            hls_relative_path = f"playlists/{task_id}/{task_paths.playlist_path.name}"
                            try:
                                await self.supabase_client.update_task(task_id, {
                                    'hls_playlist_url': hls_relative_path,
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
                            f"TTS批次: {tts_batches}, HLS段: {added_hls_segments}")
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