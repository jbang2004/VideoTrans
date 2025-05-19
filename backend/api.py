import sys
from pathlib import Path
import logging
import asyncio
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles
import ray
from ray import serve
import os
import time
import httpx

from config import Config, init_logging
from core.supabase_client import SupabaseClient

config = Config()
config.init_directories()

# 初始化全局日志配置
init_logging()

sys.path.extend(config.SYSTEM_PATHS)

logger = logging.getLogger(__name__)

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(current_dir / "templates"))

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 0.5}
)
@serve.ingress(app)
class VideoTransAPI:
    """视频翻译API服务"""
    def __init__(self):
        """初始化 API 服务，获取 MainOrchestrator 应用句柄"""
        self.logger = logger
        try:
            # 获取主编排器句柄
            # "MainOrchestratorDeployment" is the @serve.deployment name in orchestrator.py
            # "MainOrchestratorApp" is the serve.run name in launcher.py
            self.orchestrator_handle = serve.get_deployment_handle("MainOrchestratorDeployment", app_name="MainOrchestratorApp")
            
            self.supabase_client = SupabaseClient(config=config)
            self.logger.info("VideoTransAPI initialized with MainOrchestrator handle.")
        except Exception as e:
            self.logger.error(f"VideoTransAPI initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"VideoTransAPI cannot connect to MainOrchestrator: {e}")

    @app.get("/")
    async def index(self, request: Request):
        """首页"""
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/api/preprovideo")
    async def preprovideo(self, videoId: str = Body(..., embed=True)):
        """
        接收前端 videoId，下载视频并触发预处理流水线
        """
        video = await self.supabase_client.get_video(videoId)
        if not video:
            raise HTTPException(status_code=404, detail="视频记录不存在")
        storage_path = video.get("storage_path")
        bucket_name = video.get("bucket_name")

        client = await self.supabase_client._ensure_client()
        # 下载视频到内存
        try:
            data = await client.storage.from_(bucket_name).download(storage_path)
        except httpx.ConnectError as ce:
            logger.warning(f"下载视频时连接错误，重试一次: {ce}")
            data = await client.storage.from_(bucket_name).download(storage_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"下载视频失败: {e}")
        if not data:
            raise HTTPException(status_code=500, detail="下载视频返回空内容")

        task_dir = config.TASKS_DIR / videoId
        task_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(storage_path).name
        local_video_path = task_dir / filename
        try:
            async with aiofiles.open(local_video_path, "wb") as f:
                await f.write(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"保存视频文件失败: {e}")

        task_data = {
            "video_id": videoId,
            "video_path_supabase": storage_path,
            "download_video_path": str(local_video_path),
            "status": "uploaded"
        }
        logger.warning(f"task_data: {task_data}")
        resp = await self.supabase_client.store_task(task_data)
        if not resp or not resp.data:
            raise HTTPException(status_code=500, detail="创建任务失败")
        new_task_id = resp.data[0].get("id") or resp.data[0].get("task_id")

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

    @app.get("/task/{task_id}")
    async def get_task_status(self, task_id: str):
        """获取任务状态"""
        try:
            # 使用Supabase客户端获取任务状态
            task = await self.supabase_client._ensure_client()
            task = await self.supabase_client.get_task(task_id)
            
            if not task:
                return JSONResponse(content={
                    "status": "error",
                    "message": "任务不存在",
                    "progress": 0
                })
            
            # 根据任务状态计算进度
            progress = 0
            status = task.get('status', 'unknown')
            
            # 两阶段流程进度映射
            if status == 'preprocessing':
                progress = 10
            elif status == 'preprocessed':
                progress = 40
            elif status == 'translating':
                progress = 50
            elif status == 'mixing':
                progress = 85
            elif status == 'success':
                progress = 100
            
            response_data = {
                "status": status,
                "message": task.get('error_message', '处理中') if status == 'error' else '处理中',
                "progress": progress,
                "hls_ready": False
            }
            
            # 只要 hls_playlist_url 存在，就认为 HLS 已就绪
            if task.get('hls_playlist_url'):
                response_data["hls_url"] = task.get('hls_playlist_url')
                response_data["hls_ready"] = True
            
            # 如果状态为成功，更新消息并添加下载链接
            if status == 'success':
                response_data["message"] = "处理完成"
                response_data["download_url"] = f"/download/{task_id}"
            
            return JSONResponse(content=response_data)
            
        except Exception as e:
            self.logger.error(f"获取任务状态失败: {str(e)}", exc_info=True)
            return JSONResponse(content={
                "status": "error",
                "message": f"获取状态失败: {str(e)}",
                "progress": 0
            })

    @app.get("/playlists/{task_id}/{filename}")
    async def serve_playlist(self, task_id: str, filename: str):
        """提供HLS播放列表"""
        try:
            # 修改路径，使用task_id子目录
            playlist_path = config.PUBLIC_DIR / "playlists" / task_id / filename
            if not playlist_path.exists():
                # 尝试不带task_id的路径（向后兼容）
                playlist_path = config.PUBLIC_DIR / "playlists" / filename
                if not playlist_path.exists():
                    logger.error(f"播放列表未找到: {playlist_path}")
                    raise HTTPException(status_code=404, detail="播放列表未找到")
            
            logger.info(f"提供播放列表: {playlist_path}")
            async with aiofiles.open(playlist_path, mode='rb') as f:
                content = await f.read()
                
            return Response(
                content=content,
                media_type='application/vnd.apple.mpegurl',
                headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Access-Control-Allow-Origin": "*"
            }
            )
        except Exception as e:
            logger.error(f"服务播放列表失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/segments/{task_id}/{filename}")
    async def serve_segments(self, task_id: str, filename: str):
        """提供HLS视频片段"""
        try:
            segment_path = config.PUBLIC_DIR / "segments" / task_id / filename
            if not segment_path.exists():
                logger.error(f"片段文件未找到: {segment_path}")
                raise HTTPException(status_code=404, detail="片段文件未找到")
            
            # 使用StreamingResponse而非静态文件
            return StreamingResponse(
                open(segment_path, mode="rb"),
                media_type='video/MP2T',
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        except Exception as e:
            logger.error(f"服务视频片段失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/download/{task_id}")
    async def download_translated_video(self, task_id: str):
        """下载翻译后的视频"""
        final_video_path = config.TASKS_DIR / task_id / "output" / f"final_{task_id}.mp4"
        if not final_video_path.exists():
            raise HTTPException(status_code=404, detail="最终视频文件尚未生成或已被删除")
        return FileResponse(
            str(final_video_path),
            media_type='video/mp4',
            filename=f"final_{task_id}.mp4",
        )

    @app.post("/translate/{task_id}")
    async def translate_video(self, task_id: str):
        """触发翻译与合成流水线"""
        try:
            # 获取任务信息，确保预处理已完成
            task = await self.supabase_client.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在。")
            current_status = task.get('status')
            if current_status != 'preprocessed':
                raise HTTPException(status_code=400, detail=f"任务状态为 '{current_status}'。仅当状态为 'preprocessed' 时才能开始翻译。")

            # Dispatch to MainOrchestrator
            self.orchestrator_handle.run_translation_pipeline.remote(task_id)
            self.logger.info(f"成功触发 MainOrchestrator for translation: {task_id}")

            # 更新状态
            await self.supabase_client.update_task(task_id, {'status': 'translating'})

            return JSONResponse(content={
                'status': 'translating',
                'task_id': task_id,
                'message': '翻译与合成已开始'
            })
        except HTTPException as e:
            raise e
        except Exception as e:
            self.logger.error(f"调用 MainOrchestrator for translation 失败: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"无法开始翻译: {e}")

# 静态文件挂载
app.mount("/playlists", 
    StaticFiles(directory=str(config.PUBLIC_DIR / "playlists"), 
    check_dir=True), 
    name="playlists")

app.mount("/segments", 
    StaticFiles(
        directory=str(config.PUBLIC_DIR / "segments"), 
        check_dir=True
    ), 
    name="segments")

def setup_server():
    """初始化Ray Serve服务器，部署API服务"""
    # 直接连接到已有的Ray集群
    if not ray.is_initialized():
        ray.init(address="auto", namespace="videotrans", ignore_reinit_error=True)
        logger.info("已连接到Ray集群")

    # 检查预处理和翻译应用是否已部署
    try:
        serve.get_app_handle("MainOrchestratorApp")
        logger.info("成功连接到已部署的 MainOrchestratorApp 应用")
    except Exception as e:
        logger.error(f"连接 MainOrchestratorApp 应用失败，请确保 orchestrator.py 已经成功启动并由 launcher.py 部署: {e}")
        raise RuntimeError(f"无法连接到核心 MainOrchestratorApp 应用: {e}")

    # 直接部署 API 服务
    video_api = VideoTransAPI.bind()
    serve.run(video_api, name="VideoAPI", route_prefix="/", blocking=True)

    logger.info("API服务部署完成: VideoAPI")

if __name__ == "__main__":
    setup_server()