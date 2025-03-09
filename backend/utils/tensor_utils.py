import torch
import logging

logger = logging.getLogger(__name__)

def ensure_cpu_tensors(obj, depth=0, max_depth=10, path=""):
    """
    递归地确保对象中的所有PyTorch张量都在CPU上。
    
    Args:
        obj: 要处理的对象
        depth: 当前递归深度
        max_depth: 最大递归深度，防止无限递归
        path: 当前对象的路径，用于调试
        
    Returns:
        处理后的对象，所有张量都在CPU上
    """
    if depth > max_depth:
        logger.warning(f"达到最大递归深度 {max_depth}，路径: {path}")
        return obj
    
    # 处理None
    if obj is None:
        return None
    
    # 处理张量
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            logger.debug(f"将CUDA张量移动到CPU，路径: {path}")
            return obj.cpu()
        return obj
    
    # 处理字典
    elif isinstance(obj, dict):
        return {k: ensure_cpu_tensors(v, depth+1, max_depth, f"{path}.{k}") for k, v in obj.items()}
    
    # 处理列表或元组
    elif isinstance(obj, (list, tuple)):
        cls = type(obj)
        return cls(ensure_cpu_tensors(item, depth+1, max_depth, f"{path}[{i}]") for i, item in enumerate(obj))
    
    # 处理自定义对象
    elif hasattr(obj, '__dict__'):
        for attr_name, attr_value in list(obj.__dict__.items()):
            # 跳过私有属性和方法
            if attr_name.startswith('_'):
                continue
                
            try:
                # 尝试处理属性
                setattr(obj, attr_name, ensure_cpu_tensors(
                    attr_value, depth+1, max_depth, f"{path}.{attr_name}"
                ))
            except Exception as e:
                logger.warning(f"处理属性 {attr_name} 时出错: {e}")
        return obj
    
    # 其他类型直接返回
    return obj 