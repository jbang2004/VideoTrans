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
from core.video_segmenter import VideoSegmenter
from core.video_separator import VideoSeparator
from core.asr_model import ASRModel
from core.translation.translator import Translator
from core.my_index_tts import MyIndexTTSDeployment
from core.timeadjust.duration_aligner import DurationAligner
from core.timeadjust.timestamp_adjuster import TimestampAdjuster
from core.media_mixer import MediaMixer
from utils.task_state import TaskState
from utils.ffmpeg_utils import concat_videos

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
    video_segmenter_handle = VideoSegmenter.options(name="video_segmenter", num_replicas=1, ray_actor_options={"num_cpus": 0.5}).bind()
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
        self.config = global_config
        self.sample_rate = self.config.TARGET_SR
        self.logger = logger
        # 动态获取句柄 - 确保 key 与绑定时一致
        self.state_manager = handles.get("StateManager_handle")
        self.hls_manager = handles.get("hls_manager_handle")
        self.video_segmenter = handles.get("video_segmenter_handle")
        self.video_separator = handles.get("video_separator_handle")
        self.asr = handles.get("asr_model_handle")
        self.translator = handles.get("translator_handle").options(stream=True)
        # 注意：simplifier 也是 Translator 类型，但绑定的 key 是 simplifier_handle
        self.simplifier = handles.get("simplifier_handle").options(stream=True)
        self.my_index_tts = handles.get("my_index_tts_handle").options(stream=True)
        self.duration_aligner = handles.get("duration_aligner_handle")
        self.timestamp_adjuster = handles.get("timestamp_adjuster_handle")
        self.media_mixer = handles.get("media_mixer_handle")
            
        # 检查关键句柄
        if not all([self.state_manager, self.hls_manager, self.video_segmenter,
                    self.video_separator, self.asr, self.translator, self.simplifier,
                    self.my_index_tts, self.duration_aligner, self.timestamp_adjuster,
                    self.media_mixer]):
            # 获取失败的句柄的原始绑定名称
            original_binding_keys = [k for k in handles.keys()] # 获取所有传入的 key
            expected_keys = [ # 列出 __init__ 中 get 使用的 key
                 "StateManager_handle", "hls_manager_handle", "video_segmenter_handle",
                 "video_separator_handle", "asr_model_handle", "translator_handle",
                 "simplifier_handle", "my_index_tts_handle", "duration_aligner_handle",
                 "timestamp_adjuster_handle", "media_mixer_handle"
            ]
            missing_or_failed = [key for key in expected_keys if handles.get(key) is None] # 找出哪些 key 获取失败了
            raise ValueError(f"VideoTransPipe 初始化失败，无法获取必要的句柄: {missing_or_failed}. 传入的句柄键: {original_binding_keys}")

        self.logger.info("VideoTransPipe 初始化完成 (重构版)")

    async def __call__(self, task_id: str, video_path: str = None, target_language: str = None, generate_subtitle: bool = False) -> Dict[str, Any]:
        """处理单个视频翻译任务（重构版）"""
        task_state: Optional[TaskState] = None
        start_pipeline_time = time.time()
        self.logger.info(f"[{task_id}] 任务开始处理。语言: {target_language}, 字幕: {generate_subtitle}")
        try:
            task_state = await self._init_task_state(task_id, video_path, target_language, generate_subtitle)
            if not task_state:
                return {"status": "error", "message": "任务状态初始化失败"}
            hls_ok = await self._init_hls(task_id, task_state)
            if not hls_ok:
                return {"status": "error", "message": "HLS管理器初始化失败"}
            segments = await self._segment_video(task_id, task_state)
            if not segments:
                return {"status": "error", "message": "视频分段失败"}
            task_state.segments = segments
            self.logger.info(f"[{task_id}] 视频分段完成，共 {len(segments)} 段")
            for seg_idx, (seg_start, seg_duration) in enumerate(segments):
                await self._segment_maker(task_id, task_state, seg_idx, seg_start, seg_duration)
            merge_result = await self._merge_segments(task_id, task_state)
            return merge_result
        except Exception as e:
            self.logger.exception(f"[{task_id}] 任务处理过程中发生顶层异常: {e}")
            if task_id and task_state:
                await self.state_manager.complete_task.remote(task_id, False, f"处理失败: {e}")
            return {"status": "error", "message": f"处理失败: {e}"}
        finally:
            await self._final_cleanup(task_id, task_state, start_pipeline_time)

    async def _init_task_state(self, task_id, video_path, target_language, generate_subtitle):
        try:
            if video_path and target_language:
                task_data = await self.state_manager.create_task.remote(task_id, video_path, target_language, generate_subtitle)
                task_state = task_data.get("task_state") if isinstance(task_data, dict) else None
                if task_state is None:
                    raise ValueError("StateManager create_task 未返回有效 task_state")
            else:
                task_state = await self.state_manager.get_task_state.remote(task_id)
            if not task_state:
                raise ValueError("无法获取或创建任务状态")
            return task_state
        except Exception as e:
            self.logger.exception(f"[{task_id}] 任务状态初始化失败: {e}")
            if task_id:
                await self.state_manager.complete_task.remote(task_id, False, f"任务状态初始化失败: {e}")
            return None

    async def _init_hls(self, task_id, task_state):
        try:
            hls_init_result = await self.hls_manager.create_manager.remote(task_id, task_state.task_paths)
            if isinstance(hls_init_result, dict) and hls_init_result.get("status") == "error":
                raise RuntimeError(f"HLS管理器创建失败: {hls_init_result.get('message')}")
            return True
        except Exception as e:
            await self.state_manager.complete_task.remote(task_id, False, f"HLS管理器初始化失败: {e}")
            return False

    async def _segment_video(self, task_id, task_state):
        segment_result = await self.video_segmenter.segment_video.remote(task_state.video_path)
        if segment_result["status"] != "success":
            msg = f"视频分段失败: {segment_result.get('message', '未知错误')}"
            await self.state_manager.complete_task.remote(task_id, False, msg)
            return None
        return segment_result["segments"]

    async def _segment_maker(self, task_id, task_state, seg_idx, seg_start, seg_duration):
        seg_start_time = time.time()
        self.logger.info(f"[{task_id}] 开始处理分段 {seg_idx+1}/{len(task_state.segments)}")
        try:
            media_files = await self.video_separator.separate_video.remote(
                task_state.video_path, seg_start, str(task_state.task_paths.media_dir),
                seg_idx, self.config.TARGET_SR, seg_duration
            )
            if not media_files or "vocals" not in media_files or not Path(media_files["vocals"]).exists():
                self.logger.warning(f"[{task_id}][分段{seg_idx}] 视频分离失败或无有效人声，跳过此段")
                return
            task_state.segment_media_files[seg_idx] = media_files
            
            # 传递task_id和segment_index给ASR模型
            sentences = await self.asr.generate.remote(
                input=media_files["vocals"],
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
                task_id=task_id,
                segment_index=seg_idx,
                task_paths=task_state.task_paths
            )
            
            if not sentences:
                self.logger.info(f"[{task_id}][分段{seg_idx}] ASR 未检测到语音，跳过此段后续处理")
                return
                
            current_segment_sentence_count = len(sentences)
            for i, s in enumerate(sentences):
                # 只设置必要的属性，sentence_id和task_id由sentence_tools设置
                s.segment_start = seg_start
                
            task_state.sentence_counter += current_segment_sentence_count
            self.logger.info(f"[{task_id}][分段{seg_idx}] ASR 完成: {current_segment_sentence_count} 个句子")
            pipeline_info = await self._run_translation_pipeline(sentences, task_state, seg_idx)
            self.logger.info(f"[{task_id}][分段{seg_idx}] 子流水线处理完成。Info: {pipeline_info}")
        except Exception as e:
            self.logger.exception(f"[{task_id}] 处理分段 {seg_idx} 时发生错误: {e}")
        finally:
            self._clean_memory()
            seg_end_time = time.time()
            self.logger.info(f"[{task_id}] 分段 {seg_idx+1}/{len(task_state.segments)} 处理耗时: {seg_end_time - seg_start_time:.2f}s")
            await asyncio.sleep(0.05)

    async def _merge_segments(self, task_id, task_state):
        self.logger.info(f"[{task_id}] 所有分段处理完毕，开始合并 {len(task_state.merged_segments)} 个片段")
        if not task_state.merged_segments:
            msg = "没有处理成功的视频片段可以合并"
            await self.state_manager.complete_task.remote(task_id, False, msg)
            return {"status": "error", "message": msg}
        list_txt_path = task_state.task_paths.processing_dir / "concat_list.txt"
        async with aiofiles.open(list_txt_path, "w", encoding='utf-8') as f:
            for seg_mp4 in task_state.merged_segments:
                formatted_path = str(Path(seg_mp4).resolve()).replace("\\", "/")
                await f.write(f"file '{formatted_path}'\n")
        final_output_path = task_state.task_paths.output_dir / f"final_{task_id}.mp4"
        merge_start_time = time.time()
        final_video_path = await concat_videos(str(list_txt_path), str(final_output_path))
        self.logger.info(f"[{task_id}] 视频合并耗时: {time.time() - merge_start_time:.2f}s")
        if final_video_path and final_video_path.exists():
            final_video_path_str = str(final_video_path)
            await self.hls_manager.finalize_playlist.remote(task_id)
            await self.state_manager.complete_task.remote(task_id, True, "视频处理成功", final_video_path_str)
            self.logger.info(f"[{task_id}] 任务成功完成！输出: {final_video_path_str}")
            return {"status": "success", "message": "视频处理成功", "output_path": final_video_path_str}
        else:
            msg = "视频合并失败，最终文件未生成"
            await self.state_manager.complete_task.remote(task_id, False, msg)
            return {"status": "error", "message": msg}

    async def _final_cleanup(self, task_id, task_state, start_pipeline_time):
        pipeline_end_time = time.time()
        self.logger.info(f"[{task_id}] 任务处理总耗时: {pipeline_end_time - start_pipeline_time:.2f}s")
        if task_state:
            if hasattr(task_state, 'segment_media_files'): task_state.segment_media_files.clear()
            if hasattr(task_state, 'merged_segments'): task_state.merged_segments.clear()
        self._clean_memory()
        # 可选：清理临时文件 await task_state.task_paths.cleanup(keep_output=True)

    def _clean_memory(self) -> None:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            self.logger.warning(f"执行内存清理时发生错误: {e}")

    async def _run_translation_pipeline(self, sentences: List, task_state: TaskState, seg_idx: int) -> Dict:
        """运行从翻译到媒体混合的子流水线 (精简版)"""
        task_id = task_state.task_id
        processed_tts_batches = 0
        added_hls_segments = 0
        start_sub_pipeline_time = time.time()

        try:
            # 1. 翻译 (流式)
            async for translated_batch in self.translator.translate_sentences.remote(
                sentences, 
                batch_size=int(self.config.TRANSLATION_BATCH_SIZE),
                target_language=task_state.target_language 
            ):
                if not translated_batch: continue
                try:
                    # 2. TTS (流式)
                    async for tts_batch in self.my_index_tts.generate_audio_stream.remote(translated_batch):
                        if not tts_batch: continue
                        processed_tts_batches += 1
                        try:
                            # 3. 时长对齐
                            aligned_batch = await self.duration_aligner.remote(tts_batch, max_speed=1.1)
                            if not aligned_batch: continue

                            # 4. 时间戳调整 (更新 task_state.current_time)
                            adjusted_batch = await self.timestamp_adjuster.remote(aligned_batch, self.config.TARGET_SR, task_state.current_time)
                            if not adjusted_batch: continue
                            # 更新全局时间戳为本批次最后一个句子的结束时间
                            task_state.current_time = adjusted_batch[-1].adjusted_start + adjusted_batch[-1].adjusted_duration
                            # self.logger.debug(f"[{task_id}] Updated current_time: {task_state.current_time:.2f} ms")

                            # 5. 媒体混合
                            output_path = await self.media_mixer.mix_media.remote(adjusted_batch, task_state)
                            if not output_path:
                                self.logger.warning(f"[{task_id}][分段{seg_idx}] 媒体混合失败，跳过此TTS批次")
                                continue

                            # 6. HLS 处理
                            hls_result = await self.hls_manager.add_segment.remote(task_id, output_path, task_state.batch_counter + 1)
                            if hls_result["status"] == "success":
                                added_hls_segments += 1
                                task_state.merged_segments.append(output_path)
                                task_state.batch_counter += 1
                                if not task_state.hls_ready:
                                    task_state.hls_ready = True
                                    await self.state_manager.update_task_progress.remote(task_id, batch_counter=task_state.batch_counter, hls_ready=True)
                                    self.logger.info(f"[{task_id}] HLS 播放已就绪！")
                                else:
                                    # 只更新 batch_counter (如果需要频繁更新进度)
                                    await self.state_manager.update_task_progress.remote(task_id, batch_counter=task_state.batch_counter)

                            else:
                                self.logger.error(f"[{task_id}][分段{seg_idx}] 添加 HLS 片段失败: {hls_result.get('message')}")
                                # 考虑是否删除 output_path

                        except Exception as e_inner:
                            self.logger.exception(f"[{task_id}][分段{seg_idx}] 处理TTS批次 {processed_tts_batches} 时出错: {e_inner}")
                        finally:
                            # 清理内部变量
                            if 'tts_batch' in locals(): del tts_batch
                            if 'aligned_batch' in locals(): del aligned_batch
                            if 'adjusted_batch' in locals(): del adjusted_batch
                            self._clean_memory() # 每个 TTS 批次后清理

                except Exception as e_outer:
                    self.logger.exception(f"[{task_id}][分段{seg_idx}] 处理翻译批次时出错: {e_outer}")
                finally:
                    if 'translated_batch' in locals(): del translated_batch
                    self._clean_memory() # 每个翻译批次后清理

        except Exception as e_pipeline:
            self.logger.exception(f"[{task_id}][分段{seg_idx}] 子流水线发生严重错误: {e_pipeline}")

        finally:
            self.logger.info(f"[{task_id}][分段{seg_idx}] 子流水线耗时: {time.time() - start_sub_pipeline_time:.2f}s. TTS批次: {processed_tts_batches}, HLS段: {added_hls_segments}")

        return {"processed_tts_batches": processed_tts_batches, "added_hls_segments": added_hls_segments}

# --- Ray 和 Serve 初始化与服务部署 ---
def init_ray(address="auto", namespace="videotrans", log_to_driver=True):
    """初始化或连接到Ray集群"""
    import ray
    is_connected = False
    if ray.is_initialized():
         try:
              ray.cluster_resources() # 检查连接
              is_connected = True
              logger.info(f"Ray 已连接到集群。")
         except Exception:
              logger.warning("Ray 已初始化但无法连接到集群，尝试重新初始化。")
              ray.shutdown() # 先关闭
              is_connected = False

    if not is_connected:
        try:
            ray.init(address=address, namespace=namespace, log_to_driver=log_to_driver, ignore_reinit_error=True)
            connected_address = ray.get_runtime_context().gcs_address or address
            logger.info(f"Ray 初始化/连接成功。地址: {connected_address}, 命名空间: {namespace}")
            is_connected = True
        except Exception as e:
            logger.error(f"Ray 初始化失败: {e}", exc_info=True)
    return is_connected

def start_serve(detached=False, http_host="0.0.0.0", http_port=8000):
    """启动Ray Serve服务"""
    try:
        serve.start(detached=detached, http_options={"host": http_host, "port": http_port})
        logger.info(f"Ray Serve 已启动。模式: {'Detached' if detached else 'Attached'}, 地址: http://{http_host}:{http_port}")
        return True
    except Exception as e:
        logger.error(f"Ray Serve 启动失败: {e}", exc_info=True)
        return False

def setup_pipeline_services():
    """设置和部署VideoTrans流水线所需的核心服务"""
    try:
        if not init_ray():
            logger.critical("无法初始化或连接到 Ray 集群。")
            return None
        if not start_serve():
             logger.critical("无法启动 Ray Serve。")
             return None

        logger.info("开始准备部署流水线应用...") # 修改日志信息
        # 仅仅收集 handles, 不再单独部署
        handles_to_deploy = {
            "StateManager": state_manager_handle,
            "hls_manager": hls_manager_handle,
            "video_segmenter": video_segmenter_handle,
            "video_separator": video_separator_handle,
            "asr_model": asr_handle,
            "translator": translator_handle,
            "simplifier": simplifier_handle,
            "my_index_tts": my_index_tts_handle,
            "duration_aligner": duration_aligner_handle,
            "timestamp_adjuster": timestamp_adjuster_handle,
            "media_mixer": media_mixer_handle,
        }

        # --- 移除独立部署每个组件的循环 ---
        # for name, handle in handles_to_deploy.items():
        #     serve.run(handle, name=name, route_prefix=None)
        #     logger.info(f"部署完成: {name}")

        # logger.info("所有服务组件部署请求已发送。等待服务稳定...")
        # time.sleep(5) # 不再需要等待独立部署

        # 绑定主管道，传递所有句柄
        # 使用更简洁的方式传递 handles
        pipeline_app = VideoTransPipe.bind(**{name + "_handle": handle for name, handle in handles_to_deploy.items()})

        # 部署主管道应用，Ray Serve 会自动部署依赖
        serve.run(pipeline_app, name="PipelineEngine", route_prefix=None)
        logger.info(f"主管道应用 PipelineEngine 部署请求已发送。Ray Serve 将处理依赖项。")

        # 打印最终部署状态 (保留此部分用于验证)
        logger.info("\n等待部署稳定并检查状态...")
        # 可以适当增加等待时间，确保所有依赖都启动
        time.sleep(15) # 等待时间可能需要根据实际情况调整

        logger.info("\n当前 Ray Serve 部署状态:")
        try:
            apps_status = serve.status().applications
            if "PipelineEngine" in apps_status:
                 app_status = apps_status["PipelineEngine"]
                 logger.info(f"  应用: PipelineEngine, 状态: {app_status.status}")
                 # 可以在这里添加更详细的部署状态打印
                 # for dep_name, dep_status in app_status.deployments.items():
                 #      logger.info(f"    部署: {dep_name}, 状态: {dep_status.status}, Replicas: {len(dep_status.replicas)}")
            else:
                 logger.warning("PipelineEngine 应用尚未部署或状态不可用。")

        except Exception as e:
             logger.warning(f"获取 Serve 状态时出错: {e}")

        logger.info("\n流水线服务设置完成。等待请求...")
        # 返回 PipelineEngine 的部署状态，而不是所有独立组件
        return {"status": "deployed", "application": "PipelineEngine"}

    except Exception as e:
        logger.critical(f"设置流水线服务失败: {e}", exc_info=True)
        return None

# --- 脚本入口 ---
if __name__ == "__main__":
    setup_pipeline_services()