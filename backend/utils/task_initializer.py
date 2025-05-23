import logging
import asyncio
from typing import Dict, Any

from config import Config
from utils.task_storage import TaskPaths
from core.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

async def init_task(
    config: Config,
    supabase_client: SupabaseClient, 
    task_id: str, 
    video_path: str, 
    target_language: str, 
    generate_subtitle: bool
) -> bool:
    """
    Initializes task directories and confirms/updates task parameters in Supabase.
    Assumes the task record is already created by the API layer.
    Returns True if successful, False otherwise.
    """
    try:
        if not video_path or not target_language: # Basic validation
            logger.error(f"[{task_id}] Missing video_path or target_language for init_task.")
            return False

        task_paths = TaskPaths(config, task_id)
        await asyncio.to_thread(task_paths.create_directories)
        logger.info(f"[{task_id}] Task directories ensured by init_task: {task_paths.task_dir}")

        task_data_to_update = {
            'target_language': target_language,
            'generate_subtitle': generate_subtitle,
            'download_video_path': str(video_path),
        }
        
        existing_task = await supabase_client.get_task(task_id)
        if existing_task:
            asyncio.create_task(supabase_client.update_task(task_id, task_data_to_update))
            logger.info(f"[{task_id}] Task parameters异步更新/确认 by init_task.")
        else:
            logger.warning(f"[{task_id}] Task not found during init_task. API layer should have created it.")
            return False

        return True
    except Exception as e:
        logger.error(f"[{task_id}] init_task failed: {e}", exc_info=True)
        if task_id and supabase_client:
            try:
                asyncio.create_task(supabase_client.update_task(task_id, {
                    'status': 'error', 
                    'error_message': f"init_task utility failed: {e}"
                }))
            except Exception as su_e:
                logger.error(f"[{task_id}] Failed to update Supabase with init_task error: {su_e}")
        return False 