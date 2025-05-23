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
        """初始化 API 服务，获取 MainOrchestrator 应用句柄"""
        self.logger = logger
        try:
            # 获取主编排器句柄
            # "MainOrchestratorDeployment" is the @serve.deployment name in orchestrator.py
            # "MainOrchestratorApp" is the serve.run name in launcher.py
            self.orchestrator_handle = serve.get_deployment_handle("MainOrchestratorDeployment", app_name="MainOrchestratorApp")
            
            self.supabase_client = supabase_client
            self.logger.info("VideoTransAPI initialized with MainOrchestrator handle.")
        except Exception as e:
            self.logger.error(f"VideoTransAPI initialization failed: {e}", exc_info=True)
            raise RuntimeError(f"VideoTransAPI cannot connect to MainOrchestrator: {e}")

    @app.post("/api/preprovideo")
    async def preprovideo(self, videoId: str = Body(..., embed=True)):
        """
        接收前端 videoId，下载视频并触发预处理流水线
        """
        try:
            video = await self.supabase_client.get_video(videoId)
        except httpx.ConnectError as ce:
            logger.error(f"获取视频信息时连接错误: {ce}")
            raise HTTPException(status_code=500, detail="获取视频信息失败，请稍后重试")
        if not video:
            raise HTTPException(status_code=404, detail="视频记录不存在")
        storage_path = video.get("storage_path")
        bucket_name = video.get("bucket_name")

        # 下载视频到内存（最多重试 3 次）
        data = None
        last_exc = None
        for attempt in range(1, 4):
            try:
                data = await self.supabase_client.download_file(bucket_name, storage_path)
                break
            except Exception as e:
                last_exc = e
                logger.warning(f"第{attempt}次下载视频失败: {e}")
                # 清空客户端以便重新初始化
                self.supabase_client.client = None
                await asyncio.sleep(2 ** (attempt - 1))
        if data is None:
            raise HTTPException(status_code=500, detail=f"下载视频失败: {last_exc}")
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

    @app.post("/api/translate_subtitles")
    async def translate_subtitles(self, task_id: str = Body(...), target_language: str = Body(...)):
        """触发字幕翻译流程"""
        try:
            task = await self.supabase_client.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在。")
            # 每次请求都更新状态为 translating、存储目标语言并清空历史翻译
            asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'translating', 'target_language': target_language}))
            await self.supabase_client.clear_sentence_translations(task_id)
            # 调用编排，仅传递 task_id 和目标语言
            self.orchestrator_handle.run_subtitle_translation_pipeline.remote(task_id, target_language)
            self.logger.info(f"成功触发字幕翻译 Orchestrator: {task_id}, target_language: {target_language}")
            return JSONResponse(content={
                'status': 'translating', 'task_id': task_id, 'message': '字幕翻译已开始'
            })
        except HTTPException as e:
            raise e
        except Exception as e:
            self.logger.error(f"调用字幕翻译 Orchestrator 失败: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"无法开始字幕翻译: {e}")

    @app.post("/api/tts")
    async def tts(self, task_id: str = Body(..., embed=True)):
        """触发 TTS 合成流程"""
        try:
            task = await self.supabase_client.get_task(task_id)
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            await self.supabase_client.update_task(task_id, {'status': 'tts'})
            self.orchestrator_handle.run_tts_pipeline.remote(task_id)
            return JSONResponse(content={'status': 'tts', 'task_id': task_id, 'message': 'TTS 合成已开始'})
        except HTTPException as e:
            raise e
        except Exception as e:
            self.logger.error(f"触发 TTS 失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"无法触发 TTS: {e}")

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