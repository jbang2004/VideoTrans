import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import ray
from ray import serve
import httpx

from config import get_config, init_logging
from core.supabase_client import SupabaseClient

# 初始化配置和日志
config = get_config()
init_logging()

logger = logging.getLogger(__name__)

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局 SupabaseClient 实例
supabase_client = SupabaseClient(config=config)

@app.on_event("startup")
async def startup_supabase():
    """FastAPI 启动时初始化 Supabase 客户端"""
    await supabase_client.initialize()

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}
)
@serve.ingress(app)
class VideoTransAPI:
    """视频翻译API服务"""
    def __init__(self):
        self.logger = logger
        self.config = config
        self.supabase_client = supabase_client
        
        try:
            self.orchestrator_handle = serve.get_deployment_handle(
                "MainOrchestratorDeployment", 
                app_name="MainOrchestratorApp"
            )
            self.logger.info("VideoTransAPI初始化完成")
        except Exception as e:
            self.logger.error(f"VideoTransAPI初始化失败: {e}", exc_info=True)
            raise RuntimeError(f"无法连接到MainOrchestrator: {e}")

    async def _download_video_with_retry(self, bucket_name: str, storage_path: str, max_retries: int = 3):
        """带重试的视频下载"""
        for attempt in range(1, max_retries + 1):
            try:
                data = await self.supabase_client.download_file(bucket_name, storage_path)
                if data:
                    return data
            except Exception as e:
                self.logger.warning(f"第{attempt}次下载失败: {e}")
                if attempt < max_retries:
                    self.supabase_client.client = None
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    raise HTTPException(status_code=500, detail=f"下载视频失败: {e}")
        
        raise HTTPException(status_code=500, detail="下载视频返回空内容")

    async def _save_video_file(self, data: bytes, local_path: Path):
        """保存视频文件"""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(local_path, "wb") as f:
                await f.write(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存视频文件失败: {e}")

    @app.post("/api/preprovideo")
    async def preprovideo(self, videoId: str = Body(..., embed=True)):
        """接收前端 videoId，下载视频并触发预处理流水线"""
        try:
            # 获取视频信息
            try:
                video = await self.supabase_client.get_video(videoId)
            except httpx.ConnectError:
                raise HTTPException(status_code=500, detail="获取视频信息失败，请稍后重试")
            
            if not video:
                raise HTTPException(status_code=404, detail="视频记录不存在")

            storage_path = video.get("storage_path")
            bucket_name = video.get("bucket_name")

            # 下载视频
            data = await self._download_video_with_retry(bucket_name, storage_path)

            # 保存视频文件
            task_dir = self.config.TASKS_DIR / videoId
            filename = Path(storage_path).name
            local_video_path = task_dir / filename
            await self._save_video_file(data, local_video_path)

            # 创建任务
            task_data = {
                "video_id": videoId,
                "video_path_supabase": storage_path,
                "download_video_path": str(local_video_path),
                "status": "pending"
            }
            
            resp = await self.supabase_client.store_task(task_data)
            if not resp or not resp.data:
                raise HTTPException(status_code=500, detail="创建任务失败")
            
            new_task_id = resp.data[0].get("id") or resp.data[0].get("task_id")

            # 启动预处理流水线
            self.orchestrator_handle.run_preprocessing_pipeline.remote(
                task_id=new_task_id,
                video_path=str(local_video_path),
                video_width=video.get("video_width", -1),
                video_height=video.get("video_height", -1),
                target_language="zh",
                generate_subtitle=False
            )
            
            return JSONResponse(content={
                "status": "preprocessing",
                "task_id": new_task_id,
                "message": "预处理已开始"
            })

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"预处理失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"预处理失败: {e}")

    @app.post("/api/tts")
    async def tts(self, task_id: str = Body(..., embed=True)):
        """触发 TTS 合成流程"""
        try:
            task = await self.supabase_client.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            
            await self.supabase_client.update_task(task_id, {'status': 'tts'})
            self.orchestrator_handle.run_tts_pipeline.remote(task_id)
            
            return JSONResponse(content={
                'status': 'tts', 
                'task_id': task_id, 
                'message': 'TTS 合成已开始'
            })
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"触发 TTS 失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"无法触发 TTS: {e}")

def setup_server():
    """初始化Ray Serve服务器，部署API服务"""
    # 连接到Ray集群
    if not ray.is_initialized():
        ray.init(address="auto", namespace="videotrans", ignore_reinit_error=True)
        logger.info("已连接到Ray集群")

    # 检查主编排器应用
    try:
        serve.get_app_handle("MainOrchestratorApp")
        logger.info("成功连接到 MainOrchestratorApp")
    except Exception as e:
        logger.error(f"连接 MainOrchestratorApp 失败: {e}")
        raise RuntimeError(f"无法连接到核心应用: {e}")

    # 部署API服务
    video_api = VideoTransAPI.bind()
    serve.run(video_api, name="VideoAPI", route_prefix="/", blocking=True)
    logger.info("API服务部署完成")

if __name__ == "__main__":
    setup_server()