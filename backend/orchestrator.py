import logging
import asyncio
import time
import sys
import gc
import torch
from typing import List, Dict
from pathlib import Path

from ray import serve
from ray.serve.handle import DeploymentHandle # Still useful for type hints if needed

from config import Config
from utils.task_storage import TaskPaths
from core.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

@serve.deployment(
    name="MainOrchestratorDeployment", # This is the Deployment name for the orchestrator itself
    num_replicas=1, # Adjust as needed
    ray_actor_options={"num_cpus": 0.5}, # Adjust as needed
    logging_config={"log_level": "INFO"}
)
class MainOrchestrator:
    def __init__(self):
        self.logger = logger
        self.config = Config() # Global config
        self.supabase_client = SupabaseClient(config=self.config)

        self.video_separator_handle: DeploymentHandle = serve.get_deployment_handle("video_separator", app_name="VideoSeparatorApp")
        self.asr_handle: DeploymentHandle = serve.get_deployment_handle("asr_model", app_name="ASRApp")
        self.translator_handle: DeploymentHandle = serve.get_deployment_handle("translator", app_name="TranslatorApp").options(stream=True)
        self.my_index_tts_handle: DeploymentHandle = serve.get_deployment_handle("my_index_tts", app_name="TTSApp").options(stream=True)
        self.duration_aligner_handle: DeploymentHandle = serve.get_deployment_handle("duration_aligner", app_name="DurationAlignerApp")
        self.timestamp_adjuster_handle: DeploymentHandle = serve.get_deployment_handle("timestamp_adjuster", app_name="TimestampAdjusterApp")
        self.media_mixer_handle: DeploymentHandle = serve.get_deployment_handle("media_mixer", app_name="MediaMixerApp")
        self.hls_manager_handle: DeploymentHandle = serve.get_deployment_handle("hls_manager", app_name="HLSManagerApp")
        
        self.logger.info("MainOrchestrator initialized with all core actor handles.")

    async def run_preprocessing_pipeline(self, task_id: str, video_path: str, video_width: int, video_height: int, target_language: str, generate_subtitle: bool):
        """
        Orchestrates the preprocessing steps (formerly PreEngine logic).
        """
        self.logger.info(f"[{task_id}] Orchestrator: Starting preprocessing for video: {video_path}, lang: {target_language}, subtitles: {generate_subtitle}")
        asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'preprocessing'}))
        seg_start_time = time.time()
        
        try:
            task_paths = TaskPaths(self.config, task_id)

            # 1. Video Separation (formerly part of PreEngine)
            # VideoSeparator's separate_video method is expected to handle its own Supabase updates for media paths.
            separated_media = await self.video_separator_handle.separate_video.remote(
                video_path,
                str(task_paths.media_dir),
                video_width,
                video_height,
                task_id,
            )
            
            if not separated_media or "vocals_audio_path" not in separated_media or not Path(separated_media["vocals_audio_path"]).exists():
                self.logger.warning(f"[{task_id}] Orchestrator: Video separation failed or no vocals detected.")
                asyncio.create_task(self.supabase_client.update_task(task_id, {
                    'status': 'error',
                    'error_message': 'Video separation failed or no vocals detected'
                }))
                return {"status": "error", "message": "Video separation failed or no vocals detected"}
            
            self.logger.info(f"[{task_id}] Orchestrator: Video separation completed.")

            # 2. ASR Processing (formerly part of PreEngine)
            # ASRModel's generate method is expected to handle its own Supabase updates for sentences and task status.
            sentences = await self.asr_handle.generate.remote(
                input=separated_media["vocals_audio_path"],
                cache={}, # Default cache
                language="auto", # Default language detection
                use_itn=True, # Default ITN
                batch_size_s=60, # Default batch size
                merge_vad=False, # Default VAD merging
                task_id=task_id,
                task_paths=task_paths,
            )
            
            if not sentences: # ASRModel's generate now returns [] if no speech, and updates status.
                self.logger.info(f"[{task_id}] Orchestrator: ASR did not detect any speech (status updated by ASRModel).")
                return {"status": "preprocessed", "message": "Preprocessing finished (no speech detected)"} # Match PreEngine's original success response

            self.logger.info(f"[{task_id}] Orchestrator: Preprocessing completed successfully with {len(sentences)} sentences.")
            asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'preprocessed'}))
            return {"status": "preprocessed", "message": "Preprocessing finished successfully"} # Match PreEngine

        except Exception as e:
            self.logger.exception(f"[{task_id}] Orchestrator: Error during preprocessing: {e}")
            asyncio.create_task(self.supabase_client.update_task(task_id, {
                'status': 'error',
                'error_message': f"Error during preprocessing: {e}"
            }))
            return {"status": "error", "message": f"Error during preprocessing: {e}"}
        finally:
            self._clean_memory() # Keep memory cleaning practice
            self.logger.info(f"[{task_id}] Orchestrator: Preprocessing operations took: {time.time() - seg_start_time:.2f}s")

    async def run_tts_pipeline(self, task_id: str):
        """Orchestrates TTS conversion and HLS segment generation."""
        start_time = time.time()
        self.logger.info(f"[{task_id}] Orchestrator: 开始 TTS 流程")
        task_paths = TaskPaths(self.config, task_id)
        try:
            # Initialize HLS manager
            hls_init_response = await self.hls_manager_handle.create_manager.remote(task_id, task_paths)
            if not (isinstance(hls_init_response, dict) and hls_init_response.get("status") == "success"):
                self.logger.error(f"[{task_id}] HLS 管理器初始化失败: {hls_init_response}")
                return {"status": "error", "message": f"HLS init failed: {hls_init_response}"}
            added_hls_segments = 0
            current_audio_time_ms = 0.0
            processed_segment_paths = []
            batch_counter = 0

            # Generate audio stream and process
            async for tts_sentence_batch in self.my_index_tts_handle.generate_audio_stream.remote(task_id):
                if not tts_sentence_batch:
                    continue
                # Duration alignment
                aligned_batch = await self.duration_aligner_handle.remote(tts_sentence_batch, max_speed=1.2)
                if not aligned_batch:
                    continue
                # Timestamp adjustment
                adjusted_batch = await self.timestamp_adjuster_handle.remote(
                    aligned_batch,
                    self.config.TARGET_SR,
                    current_audio_time_ms
                )
                if not adjusted_batch:
                    continue
                # Update current_audio_time_ms
                last_sentence = adjusted_batch[-1]
                current_audio_time_ms = last_sentence.adjusted_start + last_sentence.adjusted_duration

                # Media mixing
                output_segment_path = await self.media_mixer_handle.mix_media.remote(
                    sentences_batch=adjusted_batch,
                    task_paths=task_paths,
                    batch_counter=batch_counter,
                    task_id=task_id
                )
                if not output_segment_path:
                    self.logger.warning(f"[{task_id}] 媒体混合失败，跳过批次 {batch_counter}")
                    continue

                # Add HLS segment
                hls_add_result = await self.hls_manager_handle.add_segment.remote(
                    task_id,
                    output_segment_path,
                    batch_counter + 1
                )
                if hls_add_result and hls_add_result.get("status") == "success":
                    added_hls_segments += 1
                    processed_segment_paths.append(output_segment_path)
                    batch_counter += 1
                    self.logger.info(f"[{task_id}] 添加 HLS 段 {batch_counter} 成功")
                else:
                    err_msg = hls_add_result.get('message') if hls_add_result else 'Unknown HLS add error'
                    self.logger.error(f"[{task_id}] 添加 HLS 段失败: {err_msg}")
                self._clean_memory()

            self.logger.info(f"[{task_id}] Orchestrator: TTS 完成，用时 {time.time() - start_time:.2f}s，生成段数 {added_hls_segments}")
            # Finalize HLS and merge
            result = await self.hls_manager_handle.finalize_merge.remote(
                task_id=task_id,
                all_processed_segment_paths=processed_segment_paths,
                task_paths=task_paths
            )
            if result and result.get("status") == "success":
                self.logger.info(f"[{task_id}] HLSManager 最终化成功: {result.get('output_path', 'N/A')}")
            else:
                err_msg = result.get('message') if result else 'Invalid finalize result'
                self.logger.error(f"[{task_id}] HLSManager 最终化失败: {err_msg}")
                return {"status": "error", "message": err_msg}
            return result
        except Exception as e:
            self.logger.exception(f"[{task_id}] Orchestrator: TTS 流程失败: {e}")
            return {"status": "error", "message": f"TTS pipeline failed: {e}"}
        finally:
            self._clean_memory()

    async def run_subtitle_translation_pipeline(self, task_id: str, target_language: str):
        """
        Orchestrates subtitle translation only.
        """
        start_time = time.time()
        self.logger.warning(f"[{task_id}] 开始字幕翻译")
        try:
            # translate_sentences 会自行更新任务状态为 'translating'
            async for translated_batch in self.translator_handle.translate_sentences.remote(
                task_id=task_id,
                target_language=target_language,
                batch_size=int(self.config.TRANSLATION_BATCH_SIZE)
            ):
                if not translated_batch:
                    self.logger.warning(f"[{task_id}] Orchestrator: Translator returned an empty batch.")
                    continue
                self.logger.info(f"[{task_id}] Orchestrator: Subtitle translation batch completed with {len(translated_batch)} sentences.")
                self._clean_memory()
            # 更新任务状态为字幕翻译完成
            asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'translated'}))
            self.logger.info(f"[{task_id}] Orchestrator: Subtitle translation completed in {time.time() - start_time:.2f}s.")
            return {"status": "success", "message": "字幕翻译完成"}
        except Exception as e:
            self.logger.exception(f"[{task_id}] Orchestrator: Subtitle translation pipeline error: {e}")
            asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f"Subtitle translation error: {e}"}))
            return {"status": "error", "message": f"Subtitle translation failed: {e}"}
        finally:
            self._clean_memory()

    def _clean_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 