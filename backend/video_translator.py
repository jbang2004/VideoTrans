# ---------------------------------
# backend/video_translator.py (重构后)
# ---------------------------------
import logging
import os
from typing import List, Dict, Any
from pathlib import Path
import ray

from core.asr_model_actor import SenseAutoModelActor
from core.clear_voice_actor import ClearVoiceActor
from core.translation.translator_actor import TranslatorActor
from core.model_in_actor import ModelInActor
from core.tts_token_gen_actor import TtsTokenGenActor  # 新导入
from core.audio_gen_actor import AudioGenActor  # 新导入
from utils.media_utils import get_video_duration, get_audio_segments, extract_segment_task
from pipeline_scheduler import PipelineScheduler
from utils.task_storage import TaskPaths
from config import Config
from utils.task_state import TaskState

from utils.ffmpeg_utils import FFmpegTool
from utils.video_utils import concat_video_segments

logger = logging.getLogger(__name__)

class ViTranslator:
    """
    全局持有大模型(ASR/TTS/翻译)对象, ...
    """
    def __init__(self, config: Config = None):
        self.logger = logger
        self.config = config or Config()
        
        # 初始化Ray（如果尚未初始化）
        if not ray.is_initialized():
            context = ray.init(ignore_reinit_error=True)
            self.logger.info(f"Ray 初始化完成，Dashboard URL: {context.dashboard_url}")
            
        self._init_global_models()

    def _init_global_models(self):
        self.logger.info("[ViTranslator] 初始化模型和工具...")
        
        # 创建音频分离器Actor
        self.audio_separator_actor = ClearVoiceActor.options(
            num_gpus=self.config.CLEARVOICE_ACTOR_NUM_GPUS,
            name="clear_voice_separator"
        ).remote(model_name='MossFormer2_SE_48K')
        
        self.sense_model_actor = SenseAutoModelActor.options(
            num_gpus=self.config.ASR_ACTOR_NUM_GPUS,
            name="sense_asr_model"
        ).remote()
        
        # 移除cosyvoice_model_actor，替换为TtsTokenGenActor和AudioGenActor
        self.tts_token_gen_actor = TtsTokenGenActor.options(
            num_gpus=0.2,
            name="tts_token_generator"
        ).remote()
        
        self.audio_gen_actor = AudioGenActor.options(
            num_gpus=0.3,
            name="audio_generator"
        ).remote()
        
        # 使用配置中的目标采样率
        self.target_sr = self.config.TARGET_SR

        # 创建翻译Actor
        translation_model = (self.config.TRANSLATION_MODEL or "deepseek").strip().lower()
        api_key = self.config.DEEPSEEK_API_KEY if translation_model == "deepseek" else self.config.GEMINI_API_KEY
        self.translator_actor = TranslatorActor.options(
            num_cpus=self.config.TRANSLATOR_ACTOR_NUM_CPUS,
            name="translator"
        ).remote(api_key=api_key, model_type=translation_model)

        # 修改ModelInActor初始化方式，不再传递cosyvoice_actor
        self.model_in_actor = ModelInActor.options(
            num_cpus=self.config.MODELIN_ACTOR_NUM_CPUS,
            name="model_in"
        ).remote()

        # 创建FFmpegTool实例，用于共享
        self.ffmpeg_tool = FFmpegTool()

        self.logger.info("[ViTranslator] 初始化完成")

    async def init_task_state(
        self,
        video_path: str,
        task_id: str,
        task_paths: TaskPaths,
        target_language="zh",
        generate_subtitle: bool = False,
    ) -> TaskState:
        """
        初始化任务状态并返回
        """
        self.logger.info(
            f"[init_task_state] 初始化任务: {video_path}, task_id={task_id}, target_language={target_language}, generate_subtitle={generate_subtitle}"
        )
        
        # 初始化任务状态
        task_state = TaskState(
            task_id=task_id,
            video_path=video_path,
            task_paths=task_paths,
            target_language=target_language,
            generate_subtitle=generate_subtitle
        )
        
        return task_state

    async def trans_video(
        self,
        task_state: TaskState,
        hls_manager_actor=None,
    ) -> Dict[str, Any]:
        """
        入口：对整段视频进行处理。包括分段、ASR、翻译、TTS、混音、生成 HLS 等。
        """
        self.logger.info(
            f"[trans_video] 开始处理视频: {task_state.video_path}, task_id={task_state.task_id}, target_language={task_state.target_language}, generate_subtitle={task_state.generate_subtitle}"
        )
        
        # 修改pipeline初始化，传递新的actor
        pipeline = PipelineScheduler(
            translator_actor=self.translator_actor,
            model_in_actor=self.model_in_actor,
            tts_token_gen_actor=self.tts_token_gen_actor,  # 新参数
            audio_gen_actor=self.audio_gen_actor,  # 新参数
            simplifier=self.translator_actor,  # 使用translator_actor作为simplifier
            config=self.config,
            sample_rate=self.target_sr,  # 使用target_sr作为采样率
            max_speed=1.2,  # 设置最大语速阈值
            hls_manager_actor=hls_manager_actor,  # 将hls_manager_actor作为参数传递
            audio_separator_actor=self.audio_separator_actor  # 传递audio_separator_actor
        )

        try:
            # 1. 获取视频总时长
            duration = await get_video_duration.remote(task_state.video_path)
            
            # 2. 划分分段
            segments = await get_audio_segments.remote(
                duration=duration, 
                segment_minutes=self.config.SEGMENT_MINUTES, 
                min_segment_minutes=self.config.MIN_SEGMENT_MINUTES
            )
            self.logger.info(f"总长度={duration:.2f}s, 分段数={len(segments)}, 任务ID={task_state.task_id}")

            if not segments:
                self.logger.warning(f"没有可用分段 -> 任务ID={task_state.task_id}")
                return {"status": "error", "message": "无法获取有效分段"}

            # 保存分段信息到task_state
            task_state.segments = segments
            
            # 3. 遍历所有分段：直接调用pipeline处理
            for i, (seg_start, seg_dur) in enumerate(segments):
                await pipeline.push_sentences_to_pipeline(
                    task_state=task_state,
                    segment_index=i,
                    segment_start=seg_start,
                    sense_model_actor=self.sense_model_actor,
                    segment_duration=seg_dur
                )

            # 5. 合并所有处理后的视频段落
            final_video_path = self._concat_segment_mp4s(task_state, hls_manager_actor)
            if final_video_path is not None and final_video_path.exists():
                self.logger.info(f"翻译后的完整视频已生成: {final_video_path}")
                
                # 添加清理逻辑：
                import torch
                torch.cuda.empty_cache()
                self.logger.info("调用 torch.cuda.empty_cache()，已释放未使用的 GPU 显存")

                return {
                    "status": "success",
                    "message": "视频翻译完成",
                    "final_video_path": str(final_video_path)
                }
            else:
                self.logger.warning("无法合并生成最终MP4文件")
                return {"status": "error", "message": "HLS完成，但无法合并出最终MP4"}

        except Exception as e:
            self.logger.exception(f"[trans_video] 任务ID={task_state.task_id} 出错: {e}")
            return {"status": "error", "message": str(e)}

    def _concat_segment_mp4s(self, task_state: TaskState, hls_manager_actor=None) -> Path:
        """
        把 pipeline_scheduler _mixing_worker 产出的所有 segment_xxx.mp4
        用 ffmpeg concat 合并成 final_{task_state.task_id}.mp4
        如果成功再删除这些小片段。
        """
        # 创建最终输出路径
        final_path = task_state.task_paths.output_dir / f"final_{task_state.task_id}.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 直接使用video_utils中的concat_video_segments
        return concat_video_segments(
            task_state=task_state,
            output_path=final_path,
            ffmpeg_tool=self.ffmpeg_tool,
            hls_manager_actor=hls_manager_actor
        )
