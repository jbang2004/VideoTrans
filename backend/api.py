import sys
from pathlib import Path
import logging
import uuid
import asyncio
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles
import ray
from ray import serve
import os
import time

from config import Config
config = Config()
config.init_directories()

sys.path.extend(config.SYSTEM_PATHS)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(name)s | L%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# 引入StateManager和VideoTransPipe
from core.state_manager import StateManager
from pipeline_scheduler import VideoTransPipe

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
    ray_actor_options={"num_cpus": 0.5}  # 减少CPU需求
)
@serve.ingress(app)
class VideoTransAPI:
    """
    视频翻译API服务
    使用StateManager进行任务状态管理，PipelineEngine进行视频翻译
    """
    def __init__(self, state_manager_handle=None, pipeline_handle=None):
        """初始化API服务，接收StateManager和PipelineEngine的handles"""
        self.state_manager = state_manager_handle or serve.get_deployment_handle("StateManager", app_name="StateManager")
        self.pipeline = pipeline_handle or serve.get_deployment_handle("VideoTransPipe", app_name="PipelineEngine")
        self.logger = logger
        self.logger.info("VideoTransAPI初始化完成")

    @app.get("/")
    async def index(self, request: Request):
        """首页"""
        return templates.TemplateResponse("index.html", {"request": request})

    @app.post("/upload")
    async def upload_video(
        self,
        video: UploadFile = File(...),
        target_language: str = Form("zh"),
        generate_subtitle: bool = Form(False),  # 是否烧制字幕
    ):
        """
        上传视频接口
        
        Args:
            video: 视频文件
            target_language: 目标语言
            generate_subtitle: 是否生成字幕
        """
        try:
            if not video:
                raise HTTPException(status_code=400, detail="没有文件上传")
            
            if not video.content_type.startswith('video/'):
                raise HTTPException(status_code=400, detail="只支持视频文件")
                
            if target_language not in ["zh", "en", "ja", "ko"]:
                raise HTTPException(status_code=400, detail=f"不支持的目标语言: {target_language}")
            
            # 生成任务ID
            task_id = str(uuid.uuid4())
            self.logger.info(f"新建任务ID: {task_id}, 目标语言: {target_language}, 生成字幕: {generate_subtitle}")
            
            # 保存上传的视频文件
            input_dir = config.TASKS_DIR / task_id / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = input_dir / f"original_{video.filename}"
            try:
                async with aiofiles.open(video_path, "wb") as f:
                    content = await video.read()
                    await f.write(content)
            except Exception as e:
                self.logger.error(f"保存文件失败: {str(e)}")
                raise HTTPException(status_code=500, detail="文件保存失败")
            
            self.logger.info(f"正在创建任务状态: {task_id}")
            # 通过StateManager创建任务状态
            task_data = await self.state_manager.create_task.remote(
                task_id=task_id,
                video_path=str(video_path),
                target_language=target_language,
                generate_subtitle=generate_subtitle
            )
            
            # 在调用VideoTransPipe之前添加日志
            self.logger.info(f"正在调用VideoTransPipe处理任务: {task_id}")
            try:
                # 确保正确调用并等待结果
                result = await self.pipeline.remote(task_id)
                self.logger.info(f"成功触发VideoTransPipe处理: {task_id}, 结果: {result}")
            except Exception as e:
                self.logger.error(f"调用VideoTransPipe失败: {str(e)}", exc_info=True)
            
            return JSONResponse(content={
                'status': 'processing',
                'task_id': task_id,
                'message': '视频上传成功，开始翻译'
            })
        except HTTPException as e:
            raise e
        except Exception as e:
            self.logger.error(f"上传处理失败: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/task/{task_id}")
    async def get_task_status(self, task_id: str):
        """获取任务状态"""
        result = await self.state_manager.get_task_status.remote(task_id)
        if not result:
            return JSONResponse(content={
                "status": "error",
                "message": "任务不存在",
                "progress": 0
            })
        return JSONResponse(content=result)

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
                    "Cache-Control": "public, max-age=3600",
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

# 部署服务
def setup_server():
    """初始化Ray Serve服务器，部署必要的服务"""
    if not ray.is_initialized():
        ray.init(address="auto")
    
    serve.start(detached=False)
    
    # 1. 首先部署StateManager
    state_manager = StateManager.bind(config)
    serve.run(state_manager, name="StateManager", route_prefix=None)
    
    # 等待确保StateManager就绪
    logger.info("等待确保StateManager就绪...")
    time.sleep(2)
    
    # 2. 部署PipelineEngine
    from pipeline_scheduler import app as pipeline_app
    serve.run(pipeline_app, name="PipelineEngine", route_prefix=None)
    
    # 等待确保PipelineEngine就绪
    logger.info("等待确保PipelineEngine就绪...")
    time.sleep(1)
    
    # 3. 部署API服务
    video_api = VideoTransAPI.bind(
        state_manager_handle=serve.get_deployment_handle("StateManager", app_name="StateManager"),
        pipeline_handle=serve.get_deployment_handle("VideoTransPipe", app_name="PipelineEngine")
    )
    serve.run(video_api, name="VideoAPI", route_prefix="/", blocking=True)
    
    logger.info("服务部署完成: StateManager, PipelineEngine, VideoAPI")

if __name__ == "__main__":
    setup_server()