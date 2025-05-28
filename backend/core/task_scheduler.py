import asyncio
import logging
from typing import Dict, List
from ray import serve
from ray.serve.handle import DeploymentHandle

from core.state_machine import TaskStateMachine, TaskState, StateTransition
from core.supabase_client import SupabaseClient
from config import get_config

logger = logging.getLogger(__name__)

@serve.deployment(
    name="TaskScheduler",
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5},
    logging_config={"log_level": "INFO"}
)
class TaskScheduler:
    """状态驱动的任务调度器"""
    
    def __init__(self):
        self.config = get_config()
        self.supabase_client = SupabaseClient(config=self.config)
        self.state_machine = TaskStateMachine(self.supabase_client)
        self.actor_handles: Dict[str, DeploymentHandle] = {}
        self.running = False
        self.poll_interval = 5  # 轮询间隔（秒）
        
        # 初始化actor句柄
        self._init_actor_handles()
        
        logger.info("TaskScheduler初始化完成")
    
    def _init_actor_handles(self):
        """初始化所有actor句柄"""
        handle_configs = [
            ("video_separator", "video_separator", "VideoSeparatorApp"),
            ("asr_model", "asr_model", "ASRApp"),
            ("translator", "translator", "TranslatorApp"),
            ("my_index_tts", "my_index_tts", "TTSApp"),
            ("duration_aligner", "duration_aligner", "DurationAlignerApp"),
            ("timestamp_adjuster", "timestamp_adjuster", "TimestampAdjusterApp"),
            ("media_mixer", "media_mixer", "MediaMixerApp"),
            ("hls_manager", "hls_manager", "HLSManagerApp")
        ]
        
        for actor_name, deployment_name, app_name in handle_configs:
            try:
                handle = serve.get_deployment_handle(deployment_name, app_name=app_name)
                if actor_name == "my_index_tts":
                    handle = handle.options(stream=True)
                self.actor_handles[actor_name] = handle
                logger.info(f"成功初始化 {actor_name} 句柄")
            except Exception as e:
                logger.error(f"初始化 {actor_name} 句柄失败: {e}")
    
    async def start_scheduler(self):
        """启动调度器"""
        if self.running:
            return {"status": "already_running"}
        
        self.running = True
        logger.info("任务调度器启动")
        
        # 启动后台任务
        asyncio.create_task(self._scheduler_loop())
        return {"status": "started"}
    
    async def stop_scheduler(self):
        """停止调度器"""
        self.running = False
        logger.info("任务调度器停止")
        return {"status": "stopped"}
    
    async def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                await self._process_pending_tasks()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"调度器循环错误: {e}")
                await asyncio.sleep(self.poll_interval)
    
    async def _process_pending_tasks(self):
        """处理待处理的任务"""
        # 获取所有需要处理的状态
        processable_states = [
            TaskState.UPLOADED,
            TaskState.PREPROCESSED,
            TaskState.TRANSLATED,
            TaskState.TTS_COMPLETED,
            TaskState.ALIGNED,
            TaskState.MIXED
        ]
        
        for state in processable_states:
            tasks = await self.state_machine.get_tasks_by_state(state)
            for task in tasks:
                await self._dispatch_task(task, state)
    
    async def _dispatch_task(self, task: Dict, current_state: TaskState):
        """分发任务给相应的actor"""
        task_id = task['task_id']
        actor_name = StateTransition.get_responsible_actor(current_state)
        
        if not actor_name or actor_name not in self.actor_handles:
            logger.error(f"任务 {task_id} 状态 {current_state.value} 没有对应的actor")
            return
        
        try:
            # 更新状态为处理中
            next_state = self._get_processing_state(current_state)
            if next_state:
                await self.state_machine.transition_state(task_id, next_state)
            
            # 调用相应的actor处理
            actor_handle = self.actor_handles[actor_name]
            
            if actor_name == "video_separator":
                await self._process_video_separation(task_id, task, actor_handle)
            elif actor_name == "asr_model":
                await self._process_asr(task_id, task, actor_handle)
            elif actor_name == "translator":
                await self._process_translation(task_id, task, actor_handle)
            elif actor_name == "my_index_tts":
                await self._process_tts(task_id, task, actor_handle)
            elif actor_name == "duration_aligner":
                await self._process_alignment(task_id, task, actor_handle)
            elif actor_name == "media_mixer":
                await self._process_mixing(task_id, task, actor_handle)
            elif actor_name == "hls_manager":
                await self._process_hls(task_id, task, actor_handle)
                
        except Exception as e:
            logger.error(f"处理任务 {task_id} 失败: {e}")
            await self.state_machine.transition_state(task_id, TaskState.ERROR, str(e))
    
    def _get_processing_state(self, current_state: TaskState) -> TaskState:
        """获取处理中状态"""
        state_mapping = {
            TaskState.UPLOADED: TaskState.PREPROCESSING,
            TaskState.PREPROCESSED: TaskState.TRANSLATING,
            TaskState.TRANSLATED: TaskState.TTS_PROCESSING,
            TaskState.TTS_COMPLETED: TaskState.ALIGNING,
            TaskState.ALIGNED: TaskState.MIXING,
            TaskState.MIXED: TaskState.HLS_PROCESSING
        }
        return state_mapping.get(current_state)
    
    async def _process_video_separation(self, task_id: str, task: Dict, actor_handle):
        """处理视频分离"""
        # 获取视频信息
        video = await self.supabase_client.get_video(task['video_id'])
        if not video:
            raise Exception("视频信息不存在")
        
        # 调用视频分离actor
        result = await actor_handle.separate_video.remote(
            video['storage_path'],
            f"tasks/{task_id}/media",
            video.get('width', 1920),
            video.get('height', 1080),
            task_id
        )
        
        if result and result.get("status") == "success":
            await self.state_machine.transition_state(task_id, TaskState.PREPROCESSED)
        else:
            raise Exception(f"视频分离失败: {result}")
    
    async def _process_asr(self, task_id: str, task: Dict, actor_handle):
        """处理ASR"""
        from utils.task_storage import TaskPaths
        task_paths = TaskPaths(self.config, task_id)
        
        # 获取人声音频路径
        vocals_path = task_paths.media_dir / "vocals.wav"
        if not vocals_path.exists():
            raise Exception("人声音频文件不存在")
        
        # 调用ASR actor
        sentences = await actor_handle.generate.remote(
            input=str(vocals_path),
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
            merge_vad=False,
            task_id=task_id,
            task_paths=task_paths,
        )
        
        if sentences:
            await self.state_machine.transition_state(task_id, TaskState.PREPROCESSED)
        else:
            await self.state_machine.transition_state(task_id, TaskState.COMPLETED)  # 无语音内容
    
    async def _process_translation(self, task_id: str, task: Dict, actor_handle):
        """处理翻译"""
        # 这里可以调用翻译actor或直接跳过
        # 如果不需要翻译，直接转到下一状态
        await self.state_machine.transition_state(task_id, TaskState.TTS_PROCESSING)
    
    async def _process_tts(self, task_id: str, task: Dict, actor_handle):
        """处理TTS"""
        from utils.task_storage import TaskPaths
        task_paths = TaskPaths(self.config, task_id)
        
        # 初始化HLS管理器
        hls_handle = self.actor_handles["hls_manager"]
        hls_init_response = await hls_handle.create_manager.remote(task_id, task_paths)
        
        if not (isinstance(hls_init_response, dict) and hls_init_response.get("status") == "success"):
            raise Exception(f"HLS初始化失败: {hls_init_response}")
        
        # 处理TTS流
        await self._process_tts_stream(task_id, task_paths, actor_handle)
        await self.state_machine.transition_state(task_id, TaskState.COMPLETED)
    
    async def _process_tts_stream(self, task_id: str, task_paths, tts_handle):
        """处理TTS流"""
        current_audio_time_ms = 0.0
        processed_segment_paths = []
        batch_counter = 0
        
        # 获取其他actor句柄
        aligner_handle = self.actor_handles["duration_aligner"]
        adjuster_handle = self.actor_handles["timestamp_adjuster"]
        mixer_handle = self.actor_handles["media_mixer"]
        hls_handle = self.actor_handles["hls_manager"]
        
        # 处理TTS流
        async for tts_sentence_batch in tts_handle.generate_audio_stream.remote(task_id):
            if not tts_sentence_batch:
                continue
            
            # 时长对齐
            aligned_batch = await aligner_handle.remote(tts_sentence_batch, max_speed=1.2)
            if not aligned_batch:
                continue
            
            # 时间戳调整
            adjusted_batch = await adjuster_handle.remote(
                aligned_batch, self.config.TARGET_SR, current_audio_time_ms
            )
            if not adjusted_batch:
                continue
            
            # 更新当前音频时间
            last_sentence = adjusted_batch[-1]
            current_audio_time_ms = last_sentence.adjusted_start + last_sentence.adjusted_duration
            
            # 媒体混合
            output_segment_path = await mixer_handle.mix_media.remote(
                sentences_batch=adjusted_batch,
                task_paths=task_paths,
                batch_counter=batch_counter,
                task_id=task_id
            )
            
            if output_segment_path:
                # 添加HLS段
                hls_result = await hls_handle.add_segment.remote(
                    task_id, output_segment_path, batch_counter + 1
                )
                
                if hls_result and hls_result.get("status") == "success":
                    processed_segment_paths.append(output_segment_path)
                    batch_counter += 1
        
        # 最终化HLS
        await hls_handle.finalize_merge.remote(
            task_id=task_id,
            all_processed_segment_paths=processed_segment_paths,
            task_paths=task_paths
        )
    
    async def _process_alignment(self, task_id: str, task: Dict, actor_handle):
        """处理时长对齐"""
        # 实现对齐逻辑
        await self.state_machine.transition_state(task_id, TaskState.ALIGNED)
    
    async def _process_mixing(self, task_id: str, task: Dict, actor_handle):
        """处理媒体混合"""
        # 实现混合逻辑
        await self.state_machine.transition_state(task_id, TaskState.MIXED)
    
    async def _process_hls(self, task_id: str, task: Dict, actor_handle):
        """处理HLS"""
        # 实现HLS处理逻辑
        await self.state_machine.transition_state(task_id, TaskState.COMPLETED)
    
    async def get_scheduler_status(self):
        """获取调度器状态"""
        return {
            "running": self.running,
            "poll_interval": self.poll_interval,
            "actor_count": len(self.actor_handles)
        } 