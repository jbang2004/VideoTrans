import logging
import asyncio
import time
import sys
import os
import gc
import torch
from typing import List, Optional, Dict, Any
from pathlib import Path

# Ray 和 Serve
import ray
from ray import serve

# 项目模块
from config import Config, init_logging
from core.video_separator import VideoSeparator
from core.asr_model import ASRModel
from utils.task_storage import TaskPaths
from utils.ffmpeg_utils import get_duration
from core.supabase_client import SupabaseClient

# 初始化全局日志配置
init_logging()

logger = logging.getLogger(__name__)

# --- 全局配置 ---
global_config = Config()
# global_config.init_directories() # Removed: Launcher will handle this

# --- Ray Serve 预处理相关部署句柄创建 ---
try:
    video_separator_handle = VideoSeparator.options(name="video_separator", num_replicas=1, max_ongoing_requests=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.2}).bind()
    asr_handle = ASRModel.options(name="asr_model", num_replicas=1, max_ongoing_requests=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.3}).bind()
except Exception as e:
    logger.critical(f"创建预处理相关 Ray Serve 部署句柄时出错: {e}", exc_info=True)
    sys.exit(1)

@serve.deployment(
    name="PreprocessingPipe",
    num_replicas=1,
    max_ongoing_requests=global_config.MAX_PARALLEL_SEGMENTS,
    ray_actor_options={"num_cpus": 0.5},
    logging_config={"log_level": "INFO"}
)
class PreprocessingPipe:
    """分阶段流水线：预处理阶段，负责视频分离和ASR"""
    def __init__(self, video_separator_handle, asr_model_handle):
        self.logger = logger
        self.supabase_client = SupabaseClient(config=global_config)
        self.video_separator = video_separator_handle
        self.asr = asr_model_handle
        self.config = global_config

    async def __call__(self, task_id: str, video_path: str, target_language: str, generate_subtitle: bool):
        self.logger.info(f"[{task_id}] Starting preprocessing stage with video: {video_path}, lang: {target_language}, subtitles: {generate_subtitle}")
        
        # Logic from _run_sep_asr inlined here
        seg_start_time = time.time()
        media_files = None # Initialize media_files to ensure it's defined in all paths
        try:
            task_paths = TaskPaths(self.config, task_id)
            current_task = await self.supabase_client.get_task(task_id)
            if current_task and current_task.get('status') == 'preprocessed':
                self.logger.info(f"[{task_id}] Task already preprocessed, skipping separation and ASR.")
                # Ensure media_files is populated with paths from the existing task for consistent return
                media_files = {
                    'silent_video_path': current_task.get('silent_video_path'),
                    'vocals_audio_path': current_task.get('vocals_audio_path'),
                    'background_audio_path': current_task.get('background_audio_path')
                }
                # Directly return preprocessed status if already done
                self.logger.info(f"[{task_id}] Preprocessing already completed according to DB.")
                return {"status": "preprocessed", "message": "Preprocessing was already finished"}

            # 1. Video Separation
            duration = await get_duration(video_path)
            self.logger.info(f"[{task_id}] Video duration={duration:.2f}s. Starting separation.")
            
            separated_media = await self.video_separator.separate_video.remote(
                video_path,
                str(task_paths.media_dir),
            )
            if not separated_media or "vocals_audio_path" not in separated_media or not Path(separated_media["vocals_audio_path"]).exists():
                self.logger.warning(f"[{task_id}] Video separation failed or no vocals detected.")
                await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': 'Video separation failed or no vocals detected'})
                return {"status": "error", "message": "Video separation failed or no vocals detected"}
            
            media_files = separated_media # Assign to media_files for subsequent use
            await self.supabase_client.update_task(task_id, {**media_files, 'status': 'preprocessing'}) # status is already preprocessing, but this updates paths
            self.logger.info(f"[{task_id}] Video separation completed: {media_files}")

            # 2. ASR
            sentences = await self.asr.generate.remote(
                input=media_files["vocals_audio_path"],
                cache={},
                language="auto", # target_language could be used here if ASR model supports it
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
                task_id=task_id,
                task_paths=task_paths
            )
            if not sentences:
                self.logger.info(f"[{task_id}] ASR did not detect any speech.")
                # Still, the task is considered 'preprocessed' as separation worked, but with no speech.
                await self.supabase_client.update_task(task_id, {'status': 'preprocessed', 'error_message': 'ASR did not detect speech'})
                # Return media_files from separation, as that part was successful
                self.logger.info(f"[{task_id}] Preprocessing completed (no speech detected)." )
                return {"status": "preprocessed", "message": "Preprocessing finished (no speech detected)", "media_files": media_files}

            # 3. Store sentences and update task status
            response = await self.supabase_client.store_sentences(sentences, task_id)
            if not response or not response.data:
                # This is a critical error if storing sentences fails
                self.logger.error(f"[{task_id}] Failed to store sentences to Supabase.")
                await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': 'Failed to store ASR sentences'})
                return {"status": "error", "message": "Failed to store ASR sentences"}
            
            await self.supabase_client.update_task(task_id, {'status': 'preprocessed'})
            self.logger.info(f"[{task_id}] Preprocessing completed successfully with {len(sentences)} sentences.")
            return {"status": "preprocessed", "message": "Preprocessing finished successfully", "media_files": media_files}

        except Exception as e:
            self.logger.exception(f"[{task_id}] Error during preprocessing (separation/ASR): {e}")
            await self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f"Preprocessing (sep/ASR) error: {e}"})
            return {"status": "error", "message": f"Error during preprocessing: {e}"}
        finally:
            self._clean_memory()
            self.logger.info(f"[{task_id}] Separation and ASR operations took: {time.time() - seg_start_time:.2f}s")

    def _clean_memory(self) -> None:
        """清理内存和GPU缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# --- 脚本入口 --- (REMOVING THIS SECTION)
# if __name__ == "__main__":
#     setup_preprocessing_service() 