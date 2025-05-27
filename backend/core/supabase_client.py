import os
import numpy as np
from supabase._async.client import AsyncClient, create_client
from supabase.lib.client_options import ClientOptions
from config import get_config
import logging
from core.sentence_tools import Sentence
import httpx
import asyncio
import functools

logger = logging.getLogger(__name__)

def sanitize_for_json(value):
    """处理数据以确保可以 JSON 序列化"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    elif isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    elif isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    else:
        return value

def retry_on_connection_error(retries=3, backoff_factor=2):
    """统一的连接错误重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_exc = None
            for attempt in range(1, retries + 1):
                try:
                    return await func(self, *args, **kwargs)
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.HTTPError) as e:
                    last_exc = e
                    logger.warning(f"{func.__name__} 第{attempt}次失败: {e}")
                    self.client = None
                    if attempt < retries:
                        await asyncio.sleep(backoff_factor ** (attempt - 1))
            logger.error(f"{func.__name__} 重试{retries}次后失败: {last_exc}", exc_info=True)
            raise last_exc
        return wrapper
    return decorator

class SupabaseClient:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = None
        logger.info("SupabaseClient初始化完成")

    async def _ensure_client(self):
        """确保客户端已初始化"""
        if self.client is None:
            try:
                options = ClientOptions(postgrest_client_timeout=10.0)
                self.client = await create_client(
                    self.config.SUPABASE_URL,
                    self.config.SUPABASE_KEY,
                    options=options
                )
                logger.info("Supabase客户端创建成功")
            except Exception as e:
                logger.error(f"创建Supabase客户端失败: {e}", exc_info=True)
                raise
        return self.client

    @retry_on_connection_error()
    async def store_task(self, task_data):
        """存储任务信息"""
        client = await self._ensure_client()
        response = await client.table('tasks').insert(task_data).execute()
        new_id = response.data[0].get('task_id') if response.data else None
        logger.info(f"存储任务 {new_id} 成功")
        return response

    @retry_on_connection_error()
    async def update_task(self, task_id, update_data):
        """更新任务信息"""
        client = await self._ensure_client()
        return await client.table('tasks').update(update_data).eq('task_id', task_id).execute()

    @retry_on_connection_error()
    async def get_task(self, task_id):
        """获取任务信息"""
        client = await self._ensure_client()
        response = await client.table('tasks').select('*').eq('task_id', task_id).execute()
        logger.info(f"获取任务 {task_id}，找到: {len(response.data) > 0}")
        return response.data[0] if response.data else None

    @retry_on_connection_error()
    async def store_sentences(self, sentences, task_id):
        """批量存储句子信息"""
        if not sentences:
            logger.warning(f"任务 {task_id} 没有句子数据需要存储")
            return None
            
        try:
            client = await self._ensure_client()
            json_sentences = []
            
            for idx, s in enumerate(sentences):
                sentence_data = {
                    'task_id': task_id,
                    'sentence_index': idx,
                    'raw_text': getattr(s, 'raw_text', ''),
                    'start_ms': getattr(s, 'start', 0),
                    'end_ms': getattr(s, 'end', 0),
                    'speaker_id': getattr(s, 'speaker_id', -1),
                    'target_duration_ms': getattr(s, 'target_duration', None) or (getattr(s, 'end', 0) - getattr(s, 'start', 0)),
                    'speech_duration_ms': getattr(s, 'speech_duration', 0.0),
                    'audio_prompt_path': getattr(s, 'audio', None),
                    'is_first': getattr(s, 'is_first', False),
                    'is_last': getattr(s, 'is_last', False),
                    'ending_silence_ms': getattr(s, 'ending_silence', 0.0)
                }
                json_sentences.append(sanitize_for_json(sentence_data))

            if json_sentences:
                response = await client.table('sentences').insert(json_sentences).execute()
                logger.info(f"存储 {len(json_sentences)} 个句子到任务 {task_id}")
                return response
            else:
                logger.warning(f"任务 {task_id}: 没有有效的句子数据可存储")
                return None
                
        except Exception as e:
            logger.error(f"存储句子失败 (任务 {task_id}): {e}", exc_info=True)
            raise

    @retry_on_connection_error()
    async def get_sentences(self, task_id, as_objects=False):
        """获取任务的所有句子，按索引排序"""
        client = await self._ensure_client()
        response = await client.table('sentences').select('*').eq('task_id', task_id).order('sentence_index').execute()
        logger.info(f"获取任务 {task_id} 的句子，数量: {len(response.data)}")
        
        if not as_objects:
            return response.data
        
        # 转换为 Sentence 对象列表
        sentences = []
        for data in response.data:
            sentence = Sentence(
                task_id=task_id,
                sentence_id=data.get('sentence_index', -1),
                raw_text=data.get('raw_text', ''),
                start=data.get('start_ms', 0.0),
                end=data.get('end_ms', 0.0),
                speaker_id=data.get('speaker_id', -1),
                target_duration=data.get('target_duration_ms'),
                audio=data.get('audio_prompt_path', ""),
                trans_text=data.get('trans_text', '') or "",
                is_first=data.get('is_first', False),
                is_last=data.get('is_last', False),
                ending_silence=data.get('ending_silence_ms', 0.0)
            )
            sentence.speech_duration = data.get('speech_duration_ms', 0.0)
            sentences.append(sentence)
        
        logger.info(f"成功转换 {len(sentences)} 个句子 (任务 {task_id})")
        return sentences

    @retry_on_connection_error()
    async def update_sentence_translation(self, task_id: str, sentence_index: int, trans_text: str):
        """更新单个句子的翻译文本"""
        client = await self._ensure_client()
        return await client.table('sentences').update({'trans_text': trans_text or ''}).eq('task_id', task_id).eq('sentence_index', sentence_index).execute()

    @retry_on_connection_error()
    async def get_video(self, video_id):
        """获取视频信息"""
        client = await self._ensure_client()
        response = await client.table('videos').select('*').eq('id', video_id).execute()
        return response.data[0] if response.data else None

    async def initialize(self):
        """初始化客户端"""
        await self._ensure_client()

    @retry_on_connection_error()
    async def download_file(self, bucket_name: str, storage_path: str) -> bytes:
        """下载文件"""
        client = await self._ensure_client()
        return await client.storage.from_(bucket_name).download(storage_path)

    @retry_on_connection_error()
    async def upload_file(self, bucket_name: str, storage_path: str, file_bytes: bytes, upsert: bool = True):
        """上传文件"""
        client = await self._ensure_client()
        # Supabase Python客户端可能需要字符串格式的upsert参数
        file_options = {"upsert": "true" if upsert else "false"}
        return await client.storage.from_(bucket_name).upload(storage_path, file_bytes, file_options)

    @retry_on_connection_error()
    async def clear_sentence_translations(self, task_id: str):
        """清空任务的所有句子翻译"""
        client = await self._ensure_client()
        return await client.table('sentences').update({'trans_text': ''}).eq('task_id', task_id).execute() 