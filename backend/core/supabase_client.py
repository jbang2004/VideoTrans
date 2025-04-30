import os
import numpy as np
from supabase._async.client import AsyncClient, create_client
from supabase.lib.client_options import AsyncClientOptions as ClientOptions
from config import Config
import logging
import json

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

class SupabaseClient:
    def __init__(self, config=None):
        self.config = config or Config()
        self.client = None
        # 只初始化配置，实际客户端会在第一次使用时创建
        logger.info("SupabaseClient instance initialized with config, client will be created on first use.")

    async def _ensure_client(self):
        """确保客户端已初始化"""
        if self.client is None:
            try:
                # 使用异步方式创建客户端
                self.client = await create_client(
                    self.config.SUPABASE_URL,
                    self.config.SUPABASE_KEY
                )
                logger.info("Supabase async client created successfully.")
            except Exception as e:
                logger.error(f"Failed to create Supabase client: {e}", exc_info=True)
                raise
        return self.client

    async def store_task(self, task_data):
        """存储任务信息"""
        try:
            client = await self._ensure_client()
            # 确保数据类型正确
            for key, value in task_data.items():
                if isinstance(value, (list, dict)):
                    task_data[key] = json.dumps(value)
            
            response = await client.table('tasks').insert(task_data).execute()
            logger.info(f"Stored task {task_data.get('task_id')}. Response count: {len(response.data) if response.data else 0}")
            return response
        except Exception as e:
            logger.error(f"Error storing task {task_data.get('task_id')}: {e}", exc_info=True)
            return None

    async def update_task(self, task_id, update_data):
        """更新任务信息"""
        try:
            client = await self._ensure_client()
            # 确保 updated_at 字段总是被更新
            if 'updated_at' not in update_data:
                update_data['updated_at'] = 'now()'
            
            # 确保数据类型正确
            for key, value in update_data.items():
                if isinstance(value, (list, dict)):
                    update_data[key] = json.dumps(value)
            
            response = await client.table('tasks').update(update_data).eq('task_id', task_id).execute()
            logger.info(f"Updated task {task_id}. Response count: {len(response.data) if response.data else 0}")
            return response
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {e}", exc_info=True)
            return None

    async def get_task(self, task_id):
        """获取任务信息"""
        try:
            client = await self._ensure_client()
            response = await client.table('tasks').select('*').eq('task_id', task_id).execute()
            logger.info(f"Fetched task {task_id}. Found: {len(response.data) > 0}")
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching task {task_id}: {e}", exc_info=True)
            return None

    async def store_sentences(self, sentences, task_id):
        """批量存储句子信息，并在内部处理数据转换"""
        if not sentences:
            logger.warning(f"No sentences data to store for task {task_id}.")
            return None
            
        try:
            client = await self._ensure_client()

            json_sentences = [] # 直接构建最终用于插入的列表
            for idx, s in enumerate(sentences):
                # 1. 从原始对象提取数据并构建基础字典
                speaker_id = getattr(s, 'speaker_id', -1)
                start_ms = getattr(s, 'start', 0)
                end_ms = getattr(s, 'end', 0)
                
                # 优先使用 s.target_duration，否则回退到 end - start
                target_duration = getattr(s, 'target_duration', None)
                target_duration_ms = target_duration if target_duration is not None else (end_ms - start_ms)
                # 确保时长不为负
                target_duration_ms = max(0, target_duration_ms)

                # 获取 audio_prompt_path (可能为 None)
                audio_prompt_path = getattr(s, 'audio', None)

                raw_sentence_data = {
                    'task_id': task_id,
                    'sentence_index': idx,
                    'raw_text': getattr(s, 'raw_text', ''),
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'speaker_id': speaker_id,
                    'target_duration_ms': target_duration_ms,
                    'audio_prompt_path': audio_prompt_path  # 添加此字段
                }
                
                # 2. 处理特殊类型（如 NumPy），得到可以直接插入的数据字典
                #    sanitize_for_json 会递归处理嵌套结构中的 NumPy 类型
                #    假设数据库驱动能处理 Python dict/list 到 JSON/JSONB 列
                sanitized_data = sanitize_for_json(raw_sentence_data)

                # 3. 直接添加处理后的字典 (移除了内部循环和 json.dumps)
                json_sentences.append(sanitized_data)

            # 4. 执行数据库插入
            if not json_sentences:
                 logger.warning(f"Task {task_id}: No sentences were processed successfully for storage.")
                 return None
                 
            response = await client.table('sentences').insert(json_sentences).execute()
            logger.info(f"Stored {len(json_sentences)} sentences for task {task_id}. Response count: {len(response.data) if response.data else 0}")
            return response
        except AttributeError as ae:
             logger.error(f"Error processing sentence attributes for task {task_id}. Ensure sentence objects have expected attributes (raw_text, start, end, optionally speaker_id). Error: {ae}", exc_info=True)
             return None
        except Exception as e:
            # Log the first raw sentence data for debugging if error occurs
            first_raw_data = {} 
            if sentences:
                try:
                    s = sentences[0]
                    first_raw_data = {
                        'task_id': task_id, 
                        'sentence_index': 0,
                        'raw_text': getattr(s, 'raw_text', ''),
                        'start_ms': getattr(s, 'start', 0),
                        'end_ms': getattr(s, 'end', 0),
                        'speaker_id': getattr(s, 'speaker_id', -1),
                        'target_duration_ms': getattr(s, 'end', 0) - getattr(s, 'start', 0)
                    }
                except Exception as inner_e:
                    logger.error(f"Error extracting data from first sentence for logging: {inner_e}")

            logger.error(f"Error storing sentences for task {task_id}. First raw sentence data: {first_raw_data}. Error: {e}", exc_info=True)
            return None

    async def get_sentences(self, task_id):
        """获取任务的所有句子，按索引排序"""
        try:
            client = await self._ensure_client()
            response = await client.table('sentences').select('*').eq('task_id', task_id).order('sentence_index').execute()
            logger.info(f"Fetched sentences for task {task_id}. Count: {len(response.data)}")
            return response.data
        except Exception as e:
            logger.error(f"Error fetching sentences for task {task_id}: {e}", exc_info=True)
            return [] 