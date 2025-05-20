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
        await self.supabase_client.update_task(task_id, {'status': 'preprocessing'})
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
                await self.supabase_client.update_task(task_id, {
                    'status': 'error',
                    'error_message': 'Video separation failed or no vocals detected'
                })
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
            await self.supabase_client.update_task(task_id, {'status': 'preprocessed'})
            return {"status": "preprocessed", "message": "Preprocessing finished successfully"} # Match PreEngine

        except Exception as e:
            self.logger.exception(f"[{task_id}] Orchestrator: Error during preprocessing: {e}")
            await self.supabase_client.update_task(task_id, {
                'status': 'error',
                'error_message': f"Error during preprocessing: {e}"
            })
            return {"status": "error", "message": f"Error during preprocessing: {e}"}
        finally:
            self._clean_memory() # Keep memory cleaning practice
            self.logger.info(f"[{task_id}] Orchestrator: Preprocessing operations took: {time.time() - seg_start_time:.2f}s")

    async def run_translation_pipeline(self, task_id: str):
        try:
            task_paths = TaskPaths(self.config, task_id)
            hls_init_response = await self.hls_manager_handle.create_manager.remote(task_id, task_paths)
            if not (isinstance(hls_init_response, dict) and hls_init_response.get("status") == "success"):
                self.logger.error(f"[{task_id}] HLS manager init failed: {hls_init_response}")
                return {"status": "error", "message": f"HLS init failed: {hls_init_response}"}

            segment_paths = await self._execute_tts_mixing_pipeline(task_id, task_paths)
            result = await self._finalize_hls_and_merge(task_id, segment_paths, task_paths)
            return result
        except Exception as e:
            self.logger.exception(f"[{task_id}] Translation pipeline failed: {e}")
            return {"status": "error", "message": f"Translation failed: {e}"}
        finally:
            self._clean_memory()

    async def _execute_tts_mixing_pipeline(self, task_id: str, task_paths: TaskPaths) -> List[str]:
        """Helper for the main TTS and mixing flow."""
        added_hls_segments = 0
        start_time = time.time()
        current_audio_time_ms = 0.0  # Tracks the end time of the last processed audio segment in milliseconds
        processed_segment_paths = [] # Store paths of successfully mixed .mp4 segments
        batch_counter = 0

        try:
            # Translator's translate_sentences handles its own Supabase updates for task status ('translating')
            # and individual sentence translations.
            async for translated_batch in self.translator_handle.translate_sentences.remote(
                task_id=task_id,
                batch_size=int(self.config.TRANSLATION_BATCH_SIZE) # Ensure it's int
            ):
                if not translated_batch:
                    self.logger.warning(f"[{task_id}] Orchestrator: Translator returned an empty batch.")
                    continue
                
                # TTS (MyIndexTTSDeployment's generate_audio_stream)
                async for tts_sentence_batch in self.my_index_tts_handle.generate_audio_stream.remote(translated_batch):
                    if not tts_sentence_batch:
                        continue

                    # Duration Alignment (DurationAligner)
                    aligned_batch = await self.duration_aligner_handle.remote(tts_sentence_batch, max_speed=1.2)
                    if not aligned_batch:
                        continue
                    
                    # Timestamp Adjustment (TimestampAdjuster)
                    adjusted_batch = await self.timestamp_adjuster_handle.remote(
                        aligned_batch,
                        self.config.TARGET_SR,
                        current_audio_time_ms # Pass current end time
                    )
                    if not adjusted_batch:
                        continue
                    
                    # Update current_audio_time_ms for the next batch
                    # It's the adjusted_start of the last sentence + its adjusted_duration
                    if adjusted_batch:
                        last_sentence_in_batch = adjusted_batch[-1]
                        current_audio_time_ms = last_sentence_in_batch.adjusted_start + last_sentence_in_batch.adjusted_duration

                    # Media Mixing (MediaMixer)
                    # MediaMixer's mix_media handles its own Supabase updates for task status ('mixing') on first batch.
                    # It needs task_paths for media locations.
                    output_segment_path = await self.media_mixer_handle.mix_media.remote(
                         sentences_batch=adjusted_batch, # Pass the processed batch
                         task_paths=task_paths,
                         batch_counter=batch_counter,
                         task_id=task_id
                     )
                    if not output_segment_path:
                        self.logger.warning(f"[{task_id}] Orchestrator: Media mixing failed for batch {batch_counter}, skipping.")
                        continue
                    
                    # Add mixed segment to HLS (HLSManager)
                    # HLSManager's add_segment handles its own Supabase updates for hls_playlist_url on first segment.
                    hls_add_result = await self.hls_manager_handle.add_segment.remote(
                        task_id,
                        output_segment_path, # Path to the .mp4 segment from MediaMixer
                        batch_counter + 1 # part_index for HLS
                    )

                    if hls_add_result and hls_add_result.get("status") == "success":
                        added_hls_segments += 1
                        processed_segment_paths.append(output_segment_path)
                        batch_counter += 1
                        self.logger.info(f"[{task_id}] Orchestrator: HLS segment {batch_counter} added successfully.")
                    else:
                        error_msg = hls_add_result.get('message') if hls_add_result else 'Unknown HLS add error'
                        self.logger.error(f"[{task_id}] Orchestrator: Failed to add HLS segment: {error_msg}")
                        # Decide if this is a fatal error for the whole pipeline or if we can continue

                    self._clean_memory() # Clean memory per batch

            self.logger.info(f"[{task_id}] Orchestrator: TTS-Mixing pipeline completed. Duration: {time.time() - start_time:.2f}s, Added HLS segments: {added_hls_segments}")
            return processed_segment_paths

        except Exception as e:
            self.logger.exception(f"[{task_id}] Orchestrator: Exception in TTS-Mixing pipeline: {e}")
            return [] # Return empty list on failure

    async def _finalize_hls_and_merge(self, task_id: str, merged_segments_paths: List[str], task_paths: TaskPaths) -> Dict:
        """Helper to finalize HLS playlist and merge video segments."""
        self.logger.info(f"[{task_id}] Orchestrator: Requesting HLSManager to finalize task, {len(merged_segments_paths)} segments.")
        
        # HLSManager's finalize_merge handles its own Supabase updates
        # for final status and download_video_path.
        result = await self.hls_manager_handle.finalize_merge.remote(
            task_id=task_id,
            all_processed_segment_paths=merged_segments_paths,
            task_paths=task_paths
        )

        if result and result.get("status") == "success":
            self.logger.info(f"[{task_id}] Orchestrator: HLSManager successfully finalized task. Output: {result.get('output_path', 'N/A')}")
        elif result:
            self.logger.error(f"[{task_id}] Orchestrator: HLSManager reported task finalization failure. Message: {result.get('message', 'Unknown error')}")
        else:
            self.logger.error(f"[{task_id}] Orchestrator: HLSManager returned invalid or no response for finalization.")
            return {"status": "error", "message": "Orchestrator: HLSManager failed to process finalization or returned invalid response"}
        
        return result

    def _clean_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 