import logging
import asyncio
import time
import gc
import torch
from typing import List, Dict
from pathlib import Path

from ray import serve
from ray.serve.handle import DeploymentHandle

from config import get_config
from utils.task_storage import TaskPaths
from core.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

@serve.deployment(
    name="MainOrchestratorDeployment",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5},
    logging_config={"log_level": "INFO"}
)
class MainOrchestrator:
    def __init__(self):
        self.logger = logger
        self.config = get_config()
        self.supabase_client = SupabaseClient(config=self.config)

        # 获取所有服务句柄
        self._init_service_handles()
        
        # 验证所有句柄是否正确初始化
        required_handles = [
            'video_separator_handle', 'asr_model_handle', 'my_index_tts_handle',
            'duration_aligner_handle', 'timestamp_adjuster_handle', 
            'media_mixer_handle', 'hls_manager_handle', 'translator_handle'
        ]
        
        missing_handles = [handle for handle in required_handles if not hasattr(self, handle)]
        if missing_handles:
            self.logger.error(f"缺少以下句柄: {missing_handles}")
            raise RuntimeError(f"服务句柄初始化失败: {missing_handles}")
        
        self.logger.info("MainOrchestrator初始化完成，所有服务句柄已就绪")

    def _init_service_handles(self):
        """初始化所有服务句柄"""
        handle_configs = [
            ("video_separator", "video_separator", "VideoSeparatorApp"),
            ("asr_model", "asr_model", "ASRApp"),
            ("my_index_tts", "my_index_tts", "TTSApp"),
            ("duration_aligner", "duration_aligner", "DurationAlignerApp"),
            ("timestamp_adjuster", "timestamp_adjuster", "TimestampAdjusterApp"),
            ("media_mixer", "media_mixer", "MediaMixerApp"),
            ("hls_manager", "hls_manager", "HLSManagerApp"),
            ("translator", "translator", "TranslatorApp")
        ]
        
        for attr_name, deployment_name, app_name in handle_configs:
            try:
                handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
                if attr_name == "my_index_tts":
                    handle = handle.options(stream=True)
                setattr(self, f"{attr_name}_handle", handle)
                self.logger.info(f"成功初始化 {attr_name}_handle")
            except Exception as e:
                self.logger.error(f"初始化 {attr_name}_handle 失败: {e}")
                raise

    async def _update_task_status(self, task_id: str, status: str, error_message: str = None):
        """统一的任务状态更新"""
        update_data = {'status': status}
        if error_message:
            update_data['error_message'] = error_message
        
        try:
            asyncio.create_task(self.supabase_client.update_task(task_id, update_data))
        except Exception as e:
            self.logger.warning(f"[{task_id}] 更新任务状态失败: {e}")

    def _clean_memory(self):
        """清理内存和GPU缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    async def run_preprocessing_pipeline(self, task_id: str, video_path: str, video_width: int, 
                                       video_height: int, target_language: str, generate_subtitle: bool):
        """执行预处理流水线"""
        start_time = time.time()
        self.logger.info(f"[{task_id}] 开始预处理: {video_path}, 语言: {target_language}")
        
        await self._update_task_status(task_id, 'preprocessing')
        
        try:
            task_paths = TaskPaths(self.config, task_id)
            # 确保任务目录存在
            await asyncio.to_thread(task_paths.create_directories)
            self.logger.info(f"[{task_id}] 任务目录已创建")

            # 视频分离
            separated_media = await self.video_separator_handle.separate_video.remote(
                video_path, str(task_paths.media_dir), video_width, video_height, task_id
            )
            
            if not separated_media or "vocals_audio_path" not in separated_media or \
               not Path(separated_media["vocals_audio_path"]).exists():
                await self._update_task_status(task_id, 'error', '视频分离失败或未检测到人声')
                return {"status": "error", "message": "视频分离失败或未检测到人声"}
            
            self.logger.info(f"[{task_id}] 视频分离完成")

            # ASR处理
            sentences = await self.asr_model_handle.generate.remote(
                input=separated_media["vocals_audio_path"],
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
                task_id=task_id,
                task_paths=task_paths,
            )
            
            if not sentences:
                self.logger.info(f"[{task_id}] ASR未检测到语音")
                return {"status": "preprocessed", "message": "预处理完成（未检测到语音）"}

            self.logger.info(f"[{task_id}] 预处理完成，获得 {len(sentences)} 个句子")
            await self._update_task_status(task_id, 'preprocessed')
            return {"status": "preprocessed", "message": "预处理完成"}

        except Exception as e:
            self.logger.exception(f"[{task_id}] 预处理错误: {e}")
            await self._update_task_status(task_id, 'error', f"预处理错误: {e}")
            return {"status": "error", "message": f"预处理错误: {e}"}
        finally:
            self._clean_memory()
            self.logger.info(f"[{task_id}] 预处理耗时: {time.time() - start_time:.2f}s")

    async def run_tts_pipeline(self, task_id: str):
        """执行TTS流水线"""
        start_time = time.time()
        self.logger.info(f"[{task_id}] 开始TTS流程")
        task_paths = TaskPaths(self.config, task_id)
        # 确保任务目录存在
        await asyncio.to_thread(task_paths.create_directories)
        self.logger.info(f"[{task_id}] 任务目录已创建")
        
        try:
            # 初始化HLS管理器
            hls_init_response = await self.hls_manager_handle.create_manager.remote(task_id, task_paths)
            if not (isinstance(hls_init_response, dict) and hls_init_response.get("status") == "success"):
                self.logger.error(f"[{task_id}] HLS管理器初始化失败: {hls_init_response}")
                return {"status": "error", "message": f"HLS初始化失败: {hls_init_response}"}

            # 处理TTS流
            result = await self._process_tts_stream(task_id, task_paths)
            
            self.logger.info(f"[{task_id}] TTS完成，耗时: {time.time() - start_time:.2f}s")
            return result

        except Exception as e:
            self.logger.exception(f"[{task_id}] TTS流程失败: {e}")
            return {"status": "error", "message": f"TTS流程失败: {e}"}
        finally:
            self._clean_memory()

    async def _process_tts_stream(self, task_id: str, task_paths: TaskPaths):
        """处理TTS流并生成HLS段"""
        added_hls_segments = 0
        current_audio_time_ms = 0.0
        processed_segment_paths = []
        batch_counter = 0

        # 生成音频流并处理
        async for tts_sentence_batch in self.my_index_tts_handle.generate_audio_stream.options(stream=True).remote(task_id):
            if not tts_sentence_batch:
                continue

            # 时长对齐
            aligned_batch = await self.duration_aligner_handle.remote(tts_sentence_batch, max_speed=1.2)
            if not aligned_batch:
                continue

            # 时间戳调整
            adjusted_batch = await self.timestamp_adjuster_handle.remote(
                aligned_batch, self.config.TARGET_SR, current_audio_time_ms
            )
            if not adjusted_batch:
                continue

            # 更新当前音频时间
            last_sentence = adjusted_batch[-1]
            current_audio_time_ms = last_sentence.adjusted_start + last_sentence.adjusted_duration

            # 媒体混合
            output_segment_path = await self.media_mixer_handle.mix_media.remote(
                sentences_batch=adjusted_batch,
                task_paths=task_paths,
                batch_counter=batch_counter,
                task_id=task_id
            )
            
            if not output_segment_path:
                self.logger.warning(f"[{task_id}] 媒体混合失败，跳过批次 {batch_counter}")
                continue

            # 添加HLS段
            hls_add_result = await self.hls_manager_handle.add_segment.remote(
                task_id, output_segment_path, batch_counter + 1
            )
            
            if hls_add_result and hls_add_result.get("status") == "success":
                added_hls_segments += 1
                processed_segment_paths.append(output_segment_path)
                batch_counter += 1
                self.logger.info(f"[{task_id}] 添加HLS段 {batch_counter} 成功")
            else:
                err_msg = hls_add_result.get('message') if hls_add_result else '未知HLS添加错误'
                self.logger.error(f"[{task_id}] 添加HLS段失败: {err_msg}")
            
            self._clean_memory()

        # 最终化HLS并合并
        result = await self.hls_manager_handle.finalize_merge.remote(
            task_id=task_id,
            all_processed_segment_paths=processed_segment_paths,
            task_paths=task_paths
        )
        
        if result and result.get("status") == "success":
            self.logger.info(f"[{task_id}] HLS最终化成功，生成段数: {added_hls_segments}")
        else:
            err_msg = result.get('message') if result else '无效的最终化结果'
            self.logger.error(f"[{task_id}] HLS最终化失败: {err_msg}")
            return {"status": "error", "message": err_msg}
        
        return result 

    async def run_translation_pipeline(self, task_id: str, target_language: str = "zh"):
        """执行翻译流水线"""
        start_time = time.time()
        self.logger.info(f"[{task_id}] 开始翻译流程，目标语言: {target_language}")
        
        try:
            # 从数据库获取句子数据
            await self.supabase_client.initialize()
            sentences = await self.supabase_client.get_sentences(task_id, as_objects=True)
            
            if not sentences:
                await self._update_task_status(task_id, 'translated')
                return {"status": "translated", "message": "没有需要翻译的句子"}
            
            self.logger.info(f"[{task_id}] 从数据库获取到 {len(sentences)} 个句子")
            
            # 执行翻译
            translated_count = 0
            async for batch_result in self.translator_handle.translate_sentences.options(stream=True).remote(
                sentences, batch_size=50, target_language=target_language
            ):
                translated_count += len(batch_result)
                self.logger.info(f"[{task_id}] 翻译进度: {translated_count}/{len(sentences)}")
                
                # 批量更新翻译结果到数据库
                for sentence in batch_result:
                    await self.supabase_client.update_sentence_translation(
                        task_id, sentence.sentence_id, sentence.trans_text
                    )
            
            await self._update_task_status(task_id, 'translated')
            self.logger.info(f"[{task_id}] 翻译完成，耗时: {time.time() - start_time:.2f}s")
            return {"status": "translated", "message": "翻译完成"}
            
        except Exception as e:
            self.logger.exception(f"[{task_id}] 翻译流程失败: {e}")
            await self._update_task_status(task_id, 'error', f"翻译失败: {e}")
            return {"status": "error", "message": f"翻译失败: {e}"}
        finally:
            self._clean_memory() 