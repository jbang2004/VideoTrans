# ------------------------------
# backend/api.py  (完整可复制版本)
# ------------------------------
import sys
from pathlib import Path
import logging
import uuid
import asyncio
from typing import Dict

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import aiofiles

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

from video_translator import ViTranslator
from utils.task_storage import TaskPaths
from fastapi import BackgroundTasks

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

# 初始化服务
vi_translator = None

@app.on_event("startup")
async def startup_event():
    global vi_translator
    vi_translator = await ViTranslator(config=config).initialize()

task_results: Dict[str, dict] = {}

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_video(
    video: UploadFile = File(...),
    target_language: str = Form("zh"),
    generate_subtitle: bool = Form(False),
):
    try:
        if not video:
            raise HTTPException(status_code=400, detail="没有文件上传")
        
        if not video.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="只支持视频文件")
            
        if target_language not in ["zh", "en", "ja", "ko"]:
            raise HTTPException(status_code=400, detail=f"不支持的目标语言: {target_language}")
        
        task_id = str(uuid.uuid4())
        task_dir = config.TASKS_DIR / task_id
        task_paths = TaskPaths(task_dir=task_dir, config=config, task_id=task_id)
        task_paths.create_directories()
        
        video_path = task_paths.input_dir / f"original_{video.filename}"
        try:
            async with aiofiles.open(video_path, "wb") as f:
                content = await video.read()
                await f.write(content)
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            raise HTTPException(status_code=500, detail="文件保存失败")

        # 创建新的事件循环来处理视频转换
        loop = asyncio.get_event_loop()
        task = loop.create_task(vi_translator.trans_video(
            video_path=str(video_path),
            task_id=task_id,
            task_paths=task_paths,
            target_language=target_language,
            generate_subtitle=generate_subtitle,
        ))
        
        task_results[task_id] = {
            "status": "processing",
            "message": "视频处理中",
            "progress": 0
        }
        
        async def on_task_complete(t):
            try:
                result = await t
                if result.get('status') == 'success':
                    task_results[task_id].update({
                        "status": "success",
                        "message": "处理完成",
                        "progress": 100
                    })
                else:
                    task_results[task_id].update({
                        "status": "error",
                        "message": result.get('message', '处理失败'),
                        "progress": 0
                    })
            except Exception as e:
                logger.error(f"任务处理失败: {str(e)}")
                task_results[task_id].update({
                    "status": "error",
                    "message": str(e),
                    "progress": 0
                })
        
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
    result = task_results.get(task_id)
    if not result:
        return {
            "status": "error",
            "message": "任务不存在",
            "progress": 0
        }
    return result

# 挂载静态文件服务
app.mount("/playlists", StaticFiles(directory=str(config.PUBLIC_DIR / "playlists")), name="playlists")
app.mount("/segments", StaticFiles(directory=str(config.PUBLIC_DIR / "segments")), name="segments")

@app.get("/download/{task_id}")
async def download_translated_video(task_id: str):
    final_video_path = config.TASKS_DIR / task_id / "output" / f"final_{task_id}.mp4"
    if not final_video_path.exists():
        raise HTTPException(status_code=404, detail="最终视频文件尚未生成或已被删除")
    return FileResponse(
        str(final_video_path),
        media_type='video/mp4',
        filename=f"final_{task_id}.mp4",
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level="info"
    )
