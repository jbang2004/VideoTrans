import logging
import time
import sys
import gc
import torch
from pathlib import Path

# Ray 和 Serve
from ray import serve

# 项目模块
from config import Config, init_logging
from core.video_separator import VideoSeparator
from core.asr_model import ASRModel
from utils.task_storage import TaskPaths
from utils.ffmpeg_utils import get_duration
# 初始化全局日志配置
init_logging()

logger = logging.getLogger(__name__)

# --- 全局配置 ---
global_config = Config()

# --- Ray Serve 预处理相关部署句柄创建 ---
try:
    video_separator_handle = VideoSeparator.options(name="video_separator", num_replicas=1, max_ongoing_requests=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.2}).bind()
    asr_handle = ASRModel.options(name="asr_model", num_replicas=1, max_ongoing_requests=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.3}).bind()
except Exception as e:
    logger.critical(f"创建预处理相关 Ray Serve 部署句柄时出错: {e}", exc_info=True)
    sys.exit(1)

@serve.deployment(
    name="PreEngine",
    num_replicas=1,
    max_ongoing_requests=global_config.MAX_PARALLEL_SEGMENTS,
    ray_actor_options={"num_cpus": 0.5},
    logging_config={"log_level": "INFO"}
)
class PreEngine:
    """分阶段流水线：预处理阶段，负责视频分离和ASR"""
    def __init__(self, video_separator_handle, asr_model_handle):
        self.logger = logger
        self.video_separator = video_separator_handle
        self.asr = asr_model_handle
        self.config = global_config

    async def __call__(self, task_id: str, video_path: str, target_language: str, generate_subtitle: bool):
        """
        执行预处理流水线：视频分离 + ASR
        
        Args:
            task_id: 任务ID
            video_path: 视频路径
            target_language: 目标语言
            generate_subtitle: 是否生成字幕
            
        Returns:
            Dict: 处理结果
        """
        self.logger.info(f"[{task_id}] Starting preprocessing stage with video: {video_path}, lang: {target_language}, subtitles: {generate_subtitle}")
        
        seg_start_time = time.time()
        try:
            task_paths = TaskPaths(self.config, task_id)

            # 2. 获取视频时长
            duration = await get_duration(video_path)
            self.logger.info(f"[{task_id}] Video duration={duration:.2f}s. Starting separation.")
            
            # 3. 视频分离 - 内部会更新数据库
            separated_media = await self.video_separator.separate_video.remote(
                video_path,
                str(task_paths.media_dir),
                task_id,
            )
            
            # 检查分离结果
            if not separated_media or "vocals_audio_path" not in separated_media or not Path(separated_media["vocals_audio_path"]).exists():
                self.logger.warning(f"[{task_id}] Video separation failed or no vocals detected.")
                return {"status": "error", "message": "Video separation failed or no vocals detected"}
            
            self.logger.info(f"[{task_id}] Video separation completed successfully.")

            # 4. ASR处理 - 内部会更新数据库
            sentences = await self.asr.generate.remote(
                input=separated_media["vocals_audio_path"],
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
                task_id=task_id,
                task_paths=task_paths,
            )
            
            # 检查ASR结果
            if not sentences:
                self.logger.info(f"[{task_id}] ASR did not detect any speech.")
                return {"status": "preprocessed", "message": "Preprocessing finished (no speech detected)"}

            self.logger.info(f"[{task_id}] Preprocessing completed successfully with {len(sentences)} sentences.")
            return {"status": "preprocessed", "message": "Preprocessing finished successfully"}

        except Exception as e:
            self.logger.exception(f"[{task_id}] Error during preprocessing (separation/ASR): {e}")
            return {"status": "error", "message": f"Error during preprocessing: {e}"}
        finally:
            self._clean_memory()
            self.logger.info(f"[{task_id}] Separation and ASR operations took: {time.time() - seg_start_time:.2f}s")

    def _clean_memory(self) -> None:
        """清理内存和GPU缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()