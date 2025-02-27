import logging
import grpc
from pathlib import Path
import asyncio
from .proto import hls_service_pb2
from .proto import hls_service_pb2_grpc
from config import Config

logger = logging.getLogger(__name__)

class HLSClient:
    """HLS gRPC客户端"""
    
    def __init__(self, config):
        self.config = config
        self.channel = None
        self.stub = None
        self._max_retries = 3
        self._retry_delay = 1.0  # 重试延迟（秒）
        
    @classmethod
    async def create(cls, config):
        """异步工厂方法创建客户端实例"""
        self = cls(config)
        await self._init_client()
        return self
        
    async def _init_client(self):
        """异步初始化客户端"""
        try:
            host = self.config.HLS_SERVICE_HOST
            port = self.config.HLS_SERVICE_PORT
            self.channel = grpc.aio.insecure_channel(f"{host}:{port}")
            self.stub = hls_service_pb2_grpc.HLSServiceStub(self.channel)
        except Exception as e:
            logger.error(f"初始化HLS客户端失败: {str(e)}")
            raise

    async def _ensure_connection(self):
        """确保连接可用，如果断开则尝试重连"""
        if not self.channel or self.channel.get_state() in [grpc.ChannelConnectivity.SHUTDOWN, grpc.ChannelConnectivity.TRANSIENT_FAILURE]:
            await self._init_client()
            
    async def _retry_rpc(self, rpc_func, *args, **kwargs):
        """带重试机制的 RPC 调用"""
        for attempt in range(self._max_retries):
            try:
                await self._ensure_connection()
                return await rpc_func(*args, **kwargs)
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    if attempt < self._max_retries - 1:
                        logger.warning(f"RPC调用失败，{self._retry_delay}秒后重试 (尝试 {attempt + 1}/{self._max_retries})")
                        await asyncio.sleep(self._retry_delay)
                        continue
                raise
        return False
        
    async def init_task(self, task_id: str) -> bool:
        """初始化HLS任务"""
        try:
            async def _init():
                request = hls_service_pb2.InitTaskRequest(task_id=task_id)
                response = await self.stub.InitTask(request)
                if not response.success:
                    logger.error(f"初始化任务失败: {response.message}")
                return response.success
            return await self._retry_rpc(_init)
        except Exception as e:
            logger.error(f"调用InitTask失败: {str(e)}")
            return False
            
    async def add_segment(self, task_id: str, segment_path: Path, segment_index: int) -> bool:
        """添加视频片段"""
        try:
            async def _add():
                request = hls_service_pb2.AddSegmentRequest(
                    task_id=task_id,
                    segment_path=str(segment_path),
                    segment_index=segment_index
                )
                response = await self.stub.AddSegment(request)
                if not response.success:
                    logger.error(f"添加片段失败: {response.message}")
                return response.success
            return await self._retry_rpc(_add)
        except Exception as e:
            logger.error(f"调用AddSegment失败: {str(e)}")
            return False
            
    async def finalize_task(self, task_id: str) -> bool:
        """完成HLS任务"""
        try:
            async def _finalize():
                request = hls_service_pb2.FinalizeTaskRequest(task_id=task_id)
                response = await self.stub.FinalizeTask(request)
                if not response.success:
                    logger.error(f"完成任务失败: {response.message}")
                return response.success
            return await self._retry_rpc(_finalize)
        except Exception as e:
            logger.error(f"调用FinalizeTask失败: {str(e)}")
            return False
            
    async def cleanup_task(self, task_id: str) -> bool:
        """清理任务资源"""
        try:
            async def _cleanup():
                request = hls_service_pb2.CleanupTaskRequest(task_id=task_id)
                response = await self.stub.CleanupTask(request)
                if not response.success:
                    logger.error(f"清理任务失败: {response.message}")
                return response.success
            return await self._retry_rpc(_cleanup)
        except Exception as e:
            logger.error(f"调用CleanupTask失败: {str(e)}")
            return False
            
    async def close(self):
        """关闭gRPC通道"""
        if self.channel:
            await self.channel.close() 