import logging
from concurrent import futures
import grpc
from pathlib import Path

from .proto import hls_service_pb2
from .proto import hls_service_pb2_grpc
from .service import HLSService
from ..storage_service import LocalStorageService
from config import Config

logger = logging.getLogger(__name__)

class HLSServicer(hls_service_pb2_grpc.HLSServiceServicer):
    """HLS gRPC服务实现"""
    
    def __init__(self, config: Config):
        self.config = config
        self.storage_service = LocalStorageService(config)
        self.hls_service = HLSService(config, self.storage_service)
        self._active_tasks = set()  # 跟踪活动的任务
        
    async def InitTask(self, request, context):
        """初始化HLS任务"""
        try:
            task_id = request.task_id
            if task_id in self._active_tasks:
                return hls_service_pb2.InitTaskResponse(
                    success=False,
                    message=f"任务 {task_id} 已存在"
                )
            
            await self.hls_service.init_task(task_id)
            self._active_tasks.add(task_id)
            return hls_service_pb2.InitTaskResponse(
                success=True,
                message=f"任务 {task_id} 初始化成功"
            )
        except Exception as e:
            logger.error(f"初始化任务失败: {str(e)}")
            return hls_service_pb2.InitTaskResponse(
                success=False,
                message=str(e)
            )
    
    async def AddSegment(self, request, context):
        """添加视频片段"""
        try:
            task_id = request.task_id
            if task_id not in self._active_tasks:
                return hls_service_pb2.AddSegmentResponse(
                    success=False,
                    message=f"任务 {task_id} 不存在或未初始化"
                )
                
            segment_path = Path(request.segment_path)
            if not segment_path.exists():
                return hls_service_pb2.AddSegmentResponse(
                    success=False,
                    message=f"片段文件不存在: {segment_path}"
                )
                
            success = await self.hls_service.add_segment(
                task_id,
                segment_path,
                request.segment_index
            )
            
            if success:
                return hls_service_pb2.AddSegmentResponse(
                    success=True,
                    message="片段添加成功",
                    segment_url=f"/segments/{task_id}/segment_{request.segment_index}.ts"
                )
            else:
                return hls_service_pb2.AddSegmentResponse(
                    success=False,
                    message="片段添加失败"
                )
                
        except Exception as e:
            logger.error(f"添加片段失败: {str(e)}")
            return hls_service_pb2.AddSegmentResponse(
                success=False,
                message=str(e)
            )
    
    async def FinalizeTask(self, request, context):
        """完成HLS任务"""
        try:
            task_id = request.task_id
            if task_id not in self._active_tasks:
                return hls_service_pb2.FinalizeTaskResponse(
                    success=False,
                    message=f"任务 {task_id} 不存在或未初始化"
                )
            
            await self.hls_service.finalize_task(task_id)
            return hls_service_pb2.FinalizeTaskResponse(
                success=True,
                message=f"任务 {task_id} 完成",
                playlist_url=f"/playlists/{task_id}.m3u8"
            )
        except Exception as e:
            logger.error(f"完成任务失败: {str(e)}")
            return hls_service_pb2.FinalizeTaskResponse(
                success=False,
                message=str(e)
            )
    
    async def CleanupTask(self, request, context):
        """清理任务资源"""
        try:
            task_id = request.task_id
            if task_id not in self._active_tasks:
                return hls_service_pb2.CleanupTaskResponse(
                    success=False,
                    message=f"任务 {task_id} 不存在或未初始化"
                )
            
            await self.hls_service.cleanup_task(task_id)
            self._active_tasks.remove(task_id)
            return hls_service_pb2.CleanupTaskResponse(
                success=True,
                message=f"任务 {task_id} 资源已清理"
            )
        except Exception as e:
            logger.error(f"清理任务失败: {str(e)}")
            return hls_service_pb2.CleanupTaskResponse(
                success=False,
                message=str(e)
            )

def serve(config: Config):
    """启动 gRPC 服务器"""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    hls_service_pb2_grpc.add_HLSServiceServicer_to_server(
        HLSServicer(config), server
    )
    server.add_insecure_port(f"{config.HLS_GRPC_HOST}:{config.HLS_GRPC_PORT}")
    return server 