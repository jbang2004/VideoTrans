import msgpack
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import base64
import json

logger = logging.getLogger(__name__)

class NumpyHandler:
    """处理 numpy 数组的序列化和反序列化"""
    
    @staticmethod
    def encode(obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy__": True,
                "data": base64.b64encode(obj.tobytes()).decode('ascii'),
                "dtype": str(obj.dtype),
                "shape": obj.shape
            }
        return obj
    
    @staticmethod
    def decode(obj):
        if isinstance(obj, dict) and obj.get("__numpy__", False):
            data = base64.b64decode(obj["data"])
            dtype = np.dtype(obj["dtype"])
            shape = obj["shape"]
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        return obj

class SentenceSerializer:
    """句子对象的序列化和反序列化工具类"""
    
    @staticmethod
    def serialize(obj: Any) -> bytes:
        """
        将对象序列化为 msgpack 格式的二进制数据
        
        Args:
            obj: 要序列化的对象
            
        Returns:
            bytes: 序列化后的二进制数据
        """
        try:
            # 处理特殊类型（如 numpy 数组）
            def default(obj):
                # 处理 numpy 标量类型
                if isinstance(obj, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.bool_)):
                    return bool(obj)
                if isinstance(obj, (np.ndarray,)):
                    return NumpyHandler.encode(obj)
                
                # 处理 Sentence 对象 - 将其转换为字典
                if hasattr(obj, '__dict__'):
                    return {
                        "__sentence__": True,
                        "data": {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
                    }
                
                # 其他类型的处理
                try:
                    return str(obj)
                except:
                    logger.warning(f"无法序列化的类型: {type(obj)}")
                    return None
            
            # 序列化为 msgpack 格式
            return msgpack.packb(obj, default=default, use_bin_type=True)
        except Exception as e:
            logger.error(f"序列化失败: {e}")
            # 失败时返回一个空字典的序列化结果
            return msgpack.packb({}, use_bin_type=True)
    
    @staticmethod
    def deserialize(data: bytes) -> Any:
        """
        将 msgpack 格式的二进制数据反序列化为对象
        
        Args:
            data: 要反序列化的二进制数据
            
        Returns:
            Any: 反序列化后的对象
        """
        try:
            # 定义 object_hook 函数处理特殊类型
            def object_hook(obj):
                if isinstance(obj, dict) and obj.get("__sentence__", False):
                    return type('Sentence', (), obj["data"])
                return NumpyHandler.decode(obj)
            
            # 反序列化
            return msgpack.unpackb(data, object_hook=object_hook, raw=False)
        except Exception as e:
            logger.error(f"反序列化失败: {e}")
            return {}

    @staticmethod
    def serialize_to_redis(obj: Any) -> str:
        """
        将对象序列化为适合存储在 Redis 中的字符串
        
        Args:
            obj: 要序列化的对象
            
        Returns:
            str: 序列化后的字符串
        """
        # 先序列化为 msgpack 二进制，再编码为 base64 字符串
        binary_data = SentenceSerializer.serialize(obj)
        return base64.b64encode(binary_data).decode('ascii')
    
    @staticmethod
    def deserialize_from_redis(data: str) -> Any:
        """
        从 Redis 中获取的字符串反序列化为对象
        
        Args:
            data: 要反序列化的字符串
            
        Returns:
            Any: 反序列化后的对象
        """
        # 先解码 base64 字符串为二进制，再反序列化
        binary_data = base64.b64decode(data.encode('ascii'))
        return SentenceSerializer.deserialize(binary_data)
