import sys
from config import get_config

config = get_config()
sys.path.extend(config.SYSTEM_PATHS)

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
import uuid
from video_translator import ViTranslator
from core.hls_manager import HLSManager
from utils.task_storage import TaskPaths
import asyncio
from typing import Dict
from fastapi.staticfiles import StaticFiles
import aiofiles

app = FastAPI(debug=config.DEBUG)

# 初始化视频翻译器（全局单例）
vi_translator = ViTranslator(config=config)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置日志
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# 获取当前文件的目录
current_dir = Path(__file__).parent

# 初始化所有必要的目录
config.init_directories()

# 配置模板
templates = Jinja2Templates(directory=str(current_dir / "templates"))

# 存储任务结果
task_results: Dict[str, dict] = {}

# 挂载静态文件目录
app.mount("/playlists", StaticFiles(directory=str(config.PUBLIC_DIR / "playlists")), name="playlists")

@app.get("/")
async def index(request: Request):
    """提供主页"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    target_language: str = Form("zh")
):
    """处理视频上传"""
    try:
        if not video:
            raise HTTPException(status_code=400, detail="没有文件上传")
        
        # 检查文件类型
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="只支持视频文件")
            
        # 验证目标语言
        if target_language not in ["zh", "en", "ja", "ko"]:
            raise HTTPException(status_code=400, detail=f"不支持的目标语言: {target_language}")
        
        # 生成任务ID和创建任务路径
        task_id = str(uuid.uuid4())
        task_paths = TaskPaths(config, task_id)
        task_paths.create_directories()
        
        # 保存上传的文件
        video_path = task_paths.input_dir / f"original_{video.filename}"
        try:
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            await task_paths.cleanup()
            raise HTTPException(status_code=500, detail="文件保存失败")
        
        # 创建 HLS 管理器
        hls_manager = HLSManager(config, task_id, task_paths)
        
        # 创建后台任务，直接传入所需参数
        task = asyncio.create_task(vi_translator.trans_video(
            video_path=str(video_path),
            task_id=task_id,
            task_paths=task_paths,
            hls_manager=hls_manager,
            target_language=target_language
        ))
        
        # 存储任务信息
        task_results[task_id] = {
            "status": "processing",
            "message": "视频处理中",
            "progress": 0
        }
        
        # 设置任务完成回调
        async def on_task_complete(task):
            try:
                result = await task
                if result.get('status') == 'success':
                    task_results[task_id].update({
                        "status": "success",
                        "message": "处理完成",
                        "progress": 100
                    })
                    # 清理处理文件，保留输出
                    # await task_paths.cleanup(keep_output=True)
                else:
                    task_results[task_id].update({
                        "status": "error",
                        "message": result.get('message', '处理失败'),
                        "progress": 0
                    })
                    # 清理所有文件
                    await task_paths.cleanup()
            except Exception as e:
                logger.error(f"任务处理失败: {str(e)}")
                task_results[task_id].update({
                    "status": "error",
                    "message": str(e),
                    "progress": 0
                })
                # 清理所有文件
                await task_paths.cleanup()
        
        task.add_done_callback(lambda t: asyncio.create_task(on_task_complete(t)))
        
        return {
            'status': 'processing',
            'task_id': task_id,
            'message': '视频上传成功，开始处理'
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"上传处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    result = task_results.get(task_id)
    if not result:
        return {
            "status": "error",
            "message": "任务不存在",
            "progress": 0
        }
    return result

@app.get("/playlists/{task_id}/{filename}")
async def serve_playlist(task_id: str, filename: str):
    """服务 m3u8 播放列表文件"""
    try:
        playlist_path = config.PUBLIC_DIR / "playlists" / filename
        if not playlist_path.exists():
            logger.error(f"播放列表未找到: {playlist_path}")
            raise HTTPException(status_code=404, detail="播放列表未找到")
        
        logger.info(f"提供播放列表: {playlist_path}")
        return FileResponse(
            str(playlist_path), 
            media_type='application/vnd.apple.mpegurl',
            headers={
                "Cache-Control": "public, max-age=3600",  # 1小时的缓存时间
                "Access-Control-Allow-Origin": "*"
            }
        )
    except Exception as e:
        logger.error(f"服务播放列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/segments/{task_id}/{filename}")
async def serve_segments(task_id: str, filename: str):
    """服务视频片段文件"""
    try:
        segment_path = config.PUBLIC_DIR / "segments" / task_id / filename
        if not segment_path.exists():
            logger.error(f"片段文件未找到: {segment_path}")
            raise HTTPException(status_code=404, detail="片段文件未找到")
        
        logger.info(f"提供视频片段: {segment_path}")
        return FileResponse(
            str(segment_path),
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level=config.LOG_LEVEL.lower())