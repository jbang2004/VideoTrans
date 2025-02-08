import os
import uuid
import aio_pika
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from shared.config import load_config
from shared.connection import RabbitMQConnection
from shared.ffmpeg_utils import FFmpegTool

app = FastAPI(title="CosySenseDubber API Gateway")
config = load_config()
ffmpeg_tool = FFmpegTool()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(current_dir / "templates"))

rabbit_conn = RabbitMQConnection()

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None

@app.on_event("startup")
async def startup():
    for d in [config.STORAGE_DIR, config.TASKS_DIR, config.PUBLIC_DIR]:
        d.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_model=TaskResponse)
async def upload_video(
    video: UploadFile,
    target_language: str = Form("zh"),
    generate_subtitle: bool = Form(False)
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file uploaded")
    if not video.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Only video files are allowed.")
    if target_language not in ["zh", "en", "ja", "ko"]:
        raise HTTPException(status_code=400, detail=f"Unsupported target language: {target_language}")

    task_id = str(uuid.uuid4())
    task_dir = config.TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    video_path = task_dir / f"original_{video.filename}"
    try:
        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save video file")

    # 获取视频时长
    duration = await ffmpeg_tool.get_duration(str(video_path))
    if duration <= 0:
        raise HTTPException(status_code=400, detail="Invalid video duration")

    task_data = {
        "task_id": task_id,
        "video_path": str(video_path),
        "target_language": target_language,
        "generate_subtitle": generate_subtitle,
        "duration": duration
    }

    channel = await rabbit_conn.connect()
    await channel.declare_queue("asr_queue", durable=True)
    await channel.default_exchange.publish(
        aio_pika.Message(
            body=json.dumps(task_data).encode(),
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT
        ),
        routing_key="asr_queue"
    )

    return TaskResponse(
        task_id=task_id,
        status="processing",
        message="Video uploaded successfully. Processing started."
    )

@app.get("/task/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    task_dir = config.TASKS_DIR / task_id
    if not task_dir.exists():
        return TaskResponse(task_id=task_id, status="error", message="Task not found")
    final_video = task_dir / "output" / f"final_{task_id}.mp4"
    if final_video.exists():
        return TaskResponse(task_id=task_id, status="completed", message="Video processing completed")
    error_file = task_dir / "error.txt"
    if error_file.exists():
        return TaskResponse(task_id=task_id, status="failed", message=error_file.read_text().strip())
    return TaskResponse(task_id=task_id, status="processing", message="Task is being processed")

app.mount("/public", FileResponse, name="public")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
