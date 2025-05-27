import ray
import sys
import logging
import asyncio
from config import Config
from ray import serve
import torch
from core.supabase_client import SupabaseClient

@serve.deployment(
    name="asr_model",
    ray_actor_options={"num_cpus": 1, "num_gpus": 0.3},
    logging_config={"log_level": "INFO"}
)
class ASRModel:
    """
    ASR模型，负责语音识别
    """
    def __init__(self):
        """初始化ASR模型"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化ASR模型Actor")
        self.config = Config()
        self.supabase_client = SupabaseClient(config=self.config)
        
        # 添加系统路径
        for path in self.config.SYSTEM_PATHS:
            if path not in sys.path:
                sys.path.append(path)
                self.logger.info(f"添加系统路径: {path}")
        
        try:
            # 导入并初始化ASR模型
            from core.auto_sense import SenseAutoModel
            
            # 直接使用硬编码参数，与原始代码保持一致
            self.model = SenseAutoModel(
                config=self.config,
                model="iic/SenseVoiceSmall",
                remote_code="./models/SenseVoice/model.py",
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                vad_kwargs={"max_single_segment_time": 30000},
                spk_model="cam++",
                trust_remote_code=True,
                disable_update=True,
                device="cuda"
            )
            
            self.logger.info("ASR模型加载完成")
        except Exception as e:
            self.logger.error(f"ASR模型加载失败: {str(e)}")
            raise
    
    async def generate(self, input, task_id=None, task_paths=None, **kwargs):
        """
        执行ASR模型生成方法
        
        Args:
            input: 输入音频文件路径
            task_id: 任务ID
            task_paths: 任务路径对象
            **kwargs: 其他参数
            
        Returns:
            识别结果，句子列表
        """
        sentences = None
        
        try:
            self.logger.info(f"开始ASR识别音频: {input if isinstance(input, str) else '(已加载音频)'}")
            
            # 创建一个包含所有参数的字典，但将显式参数放在前面
            call_kwargs = {
                'task_id': task_id,
                'task_paths': task_paths,
                **kwargs  # 将原始kwargs合并进来
            }
                
            # 使用asyncio.to_thread包装同步调用，传递合并后的参数字典
            sentences = await asyncio.to_thread(self.model.generate, input, **call_kwargs)
            
            # 处理ASR结果
            if not sentences or len(sentences) == 0:
                self.logger.info(f"[{task_id}] ASR没有检测到语音")
                if task_id:
                    asyncio.create_task(self.supabase_client.update_task(task_id, {
                        'status': 'preprocessed', 
                        'error_message': 'ASR did not detect speech'
                    }))
                return []
                
            # 存储句子到数据库
            if task_id:
                response = await self.supabase_client.store_sentences(sentences, task_id)
                if not response or not response.data:
                    self.logger.error(f"[{task_id}] 存储句子到Supabase失败")
                    asyncio.create_task(self.supabase_client.update_task(task_id, {
                        'status': 'error', 
                        'error_message': 'Failed to store ASR sentences'
                    }))
                    return []
                    
                # 更新任务状态为预处理完成
                await self.supabase_client.update_task(task_id, {'status': 'preprocessed'})
                self.logger.info(f"[{task_id}] ASR识别完成，获得 {len(sentences)} 个句子并已保存到数据库")
            else:
                self.logger.info(f"ASR识别完成，获得 {len(sentences)} 个句子")
                
            return sentences
        except Exception as e:
            self.logger.error(f"ASR识别失败: {str(e)}")
            if task_id:
                await self.supabase_client.update_task(task_id, {
                    'status': 'error', 
                    'error_message': f"ASR error: {e}"
                })
            raise
        finally:
            # 已有的GPU清理 - 保持不变
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.debug("ASRModel: Cleared GPU cache.")
    