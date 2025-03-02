import ray
import logging
import psutil
import time
import numpy as np
from threading import Lock

logger = logging.getLogger(__name__)

class GPUResourceManager:
    """GPU资源自动分配管理器"""
    
    def __init__(self, total_gpus=1.0, min_gpu=0.1, check_interval=5):
        self.total_gpus = total_gpus  # 总GPU资源
        self.min_gpu = min_gpu         # 最小分配单位
        self.check_interval = check_interval  # 检查间隔(秒)
        self.lock = Lock()
        self.last_check = 0
        self.usage_history = []        # 保存历史使用率
        self.history_window = 3        # 历史窗口大小
        
        # 模型优先级和基础资源需求
        self.model_priority = {
            "cosyvoice": 0.7,    # TTS模型优先级高，基础资源需求0.7
            "sense_asr": 0.3,    # ASR模型优先级低，基础资源需求0.3
        }
    
    def _get_current_load(self):
        """获取当前系统负载"""
        try:
            # 1. CPU负载检查
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # 2. 内存使用检查
            memory_percent = psutil.virtual_memory().percent
            # 3. GPU使用率检查 (可选，需要安装额外库如pynvml)
            
            # 综合负载分数 (0-100)
            load_score = (cpu_percent + memory_percent) / 2
            return load_score
        except Exception as e:
            logger.error(f"获取系统负载失败: {e}")
            return 50  # 默认中等负载
    
    def _check_ray_resources(self):
        """检查Ray集群资源状态"""
        try:
            resources = ray.available_resources()
            gpu_available = resources.get("GPU", 0)
            return {
                "gpu_available": gpu_available,
                "total_resources": resources
            }
        except Exception as e:
            logger.error(f"获取Ray资源失败: {e}")
            return {"gpu_available": 0.1, "total_resources": {}}
    
    def get_allocation(self, model_type):
        """根据负载情况为指定模型类型分配GPU资源"""
        with self.lock:  # 防止并发分配冲突
            # 避免频繁检查
            current_time = time.time()
            if current_time - self.last_check < self.check_interval:
                # 使用缓存的最近分配决策
                if len(self.usage_history) > 0:
                    load_score = np.mean(self.usage_history)
                else:
                    load_score = 50  # 默认中等负载
            else:
                # 更新负载并缓存
                load_score = self._get_current_load()
                self.usage_history.append(load_score)
                if len(self.usage_history) > self.history_window:
                    self.usage_history.pop(0)
                self.last_check = current_time
            
            # 检查Ray资源
            ray_resources = self._check_ray_resources()
            gpu_available = ray_resources["gpu_available"]
            
            # 负载级别分类 (0-100)
            if load_score < 30:  # 低负载
                scale_factor = 1.0
            elif load_score < 70:  # 中等负载 
                scale_factor = 0.8
            else:  # 高负载
                scale_factor = 0.6
            
            # 根据模型类型和优先级分配资源
            base_allocation = self.model_priority.get(model_type, 0.1)
            allocation = max(self.min_gpu, base_allocation * scale_factor)
            
            # 检查是否超出可用资源
            if allocation > gpu_available and gpu_available > self.min_gpu:
                logger.warning(f"GPU资源不足，调整分配: {allocation} -> {gpu_available}")
                allocation = gpu_available
            
            logger.info(f"模型 {model_type} 分配GPU: {allocation:.2f} (负载: {load_score:.1f}, 缩放: {scale_factor:.1f})")
            return allocation 