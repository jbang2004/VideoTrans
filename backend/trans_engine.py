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
    name="TransEngine",
    num_replicas=1,
    max_ongoing_requests=global_config.MAX_PARALLEL_SEGMENTS,
    ray_actor_options={"num_cpus": 0.5},
    logging_config={"log_level": "INFO"}
)
class TransPipe:
    """分阶段流水线：翻译与合成阶段"""
    def __init__(self, translator_handle: DeploymentHandle, my_index_tts_handle: DeploymentHandle, duration_aligner_handle: DeploymentHandle, timestamp_adjuster_handle: DeploymentHandle, media_mixer_handle: DeploymentHandle, hls_manager_handle: DeploymentHandle):
        self.logger = logger
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
            # 创建任务路径对象，避免重复创建
            task_paths = TaskPaths(self.config, task_id)
            
            # 初始化HLS管理器
            init_ok = await self._init_hls(task_id, task_paths)
            if not init_ok:
                return {"status": "error", "message": "HLS initialization failed."}

            # 运行翻译 -> 合成 -> HLS
            merged_segments = await self._run_translation_pipeline(task_id, task_paths)
            result = await self._merge_segments(task_id, merged_segments, task_paths)
            self.logger.info(f"[{task_id}] Translation finished with status: {result.get('status')}")
            return result
        except Exception as e:
            self.logger.exception(f"[{task_id}] Translation stage failed: {e}")
            return {"status": "error", "message": f"Translation failed: {e}"}
        finally:
            self._clean_memory()

    async def _init_hls(self, task_id: str, task_paths: TaskPaths) -> bool:
        """初始化HLS管理器"""
        try:
            result = await self.hls_manager.create_manager.remote(task_id, task_paths)
            if isinstance(result, dict) and result.get("status") == "error":
                raise RuntimeError(f"HLS管理器创建失败: {result.get('message')}")
            self.logger.info(f"[{task_id}] HLS管理器初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"[{task_id}] HLS管理器初始化失败: {e}")
            return False

    async def _run_translation_pipeline(self, task_id: str, task_paths: TaskPaths) -> List[str]:
        """运行从翻译到媒体混合的子流水线 (从数据库获取依赖信息)"""
        added_hls_segments = 0
        start_time = time.time()
        current_time = 0.0
        merged_segments_paths = []
        batch_counter = 0

        # 3. 主要处理流程 (后续逻辑使用上面获取或重建的变量)
        try:
            # 翻译流程 (按批次进行)
            async for translated_batch in self.translator.translate_sentences.remote(
                task_id=task_id,
                batch_size=int(self.config.TRANSLATION_BATCH_SIZE)
            ):
                if not translated_batch:
                    self.logger.warning(f"[{task_id}] Translator 返回了一个空批次，继续处理下一个")
                    continue
                self.logger.info(f"[{task_id}] 从 Translator 收到 {len(translated_batch)} 个已翻译/处理的句子") # 添加日志

                # TTS生成语音
                async for tts_batch in self.my_index_tts.generate_audio_stream.remote(translated_batch):
                    if not tts_batch:
                        continue

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

                    # 媒体混合 (使用重建的 media_files 和传入的 task_paths)
                    output_path = await self.media_mixer.mix_media.remote(
                         adjusted_batch,
                         task_paths=task_paths,
                         batch_counter=batch_counter,
                         task_id=task_id
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

                    else:
                        error_msg = hls_result.get('message') if hls_result else '未知错误'
                        self.logger.error(f"[{task_id}] 添加HLS片段失败: {error_msg}")

                    # 清理内存
                    self._clean_memory()

            self.logger.info(f"[{task_id}] 翻译流程完成，耗时: {time.time() - start_time:.2f}s, "
                            f"HLS段: {added_hls_segments}")
            return merged_segments_paths

        except Exception as e:
            self.logger.exception(f"[{task_id}] 翻译流程异常: {e}")
            return []

    async def _merge_segments(self, task_id: str, merged_segments: List[str], task_paths: TaskPaths) -> Dict:
        """合并处理好的视频片段，现在委托给 HLSManager """
        self.logger.info(f"[{task_id}] TranslationPipe: 请求 HLSManager 最终化任务处理，包含 {len(merged_segments)} 个片段。")
        
        # 调用 HLSManager 的新方法进行合并和状态更新
        result = await self.hls_manager.finalize_merge.remote(
            task_id=task_id,
            all_processed_segment_paths=merged_segments,
            task_paths=task_paths
        )

        # 根据 HLSManager 返回的结果记录日志
        if result and result.get("status") == "success":
            self.logger.info(f"[{task_id}] TranslationPipe: HLSManager 成功完成任务处理。输出: {result.get('output_path', 'N/A')}")
        elif result:
            self.logger.error(f"[{task_id}] TranslationPipe: HLSManager 报告任务处理失败。消息: {result.get('message', '未知错误')}")
        else:
            self.logger.error(f"[{task_id}] TranslationPipe: HLSManager 返回了无效的响应或未返回响应。")
            # 提供一个默认的错误返回，以防 HLSManager 崩溃或返回 None
            return {"status": "error", "message": "HLSManager 未能处理最终化请求或返回无效响应"}

        return result

    def _clean_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 