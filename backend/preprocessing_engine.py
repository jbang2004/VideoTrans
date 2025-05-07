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

# --- 脚本入口 --- (REMOVING THIS SECTION)
# if __name__ == "__main__":
#     setup_preprocessing_service() 