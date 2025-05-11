import os
import numpy as np
from supabase._async.client import AsyncClient, create_client
from supabase.lib.client_options import ClientOptions
from config import Config
import logging
from core.sentence_tools import Sentence

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
        logger.info("SupabaseClient初始化完成")

    async def _ensure_client(self):
        """确保客户端已初始化"""
        if self.client is None:
            try:
                # 设置 postgrest 超时和重试选项
                options = ClientOptions(
                    postgrest_client_timeout=30.0,  # 设置 postgrest 客户端超时时间为30秒
                )
                self.client = await create_client(
                    self.config.SUPABASE_URL,
                    self.config.SUPABASE_KEY,
                    options=options
                )
                logger.info("Supabase客户端创建成功 (postgrest_timeout=30s)")
            except Exception as e:
                logger.error(f"创建Supabase客户端失败: {e}", exc_info=True)
                raise
        return self.client

    async def store_task(self, task_data):
        """存储任务信息"""
        try:
            client = await self._ensure_client()
            response = await client.table('tasks').insert(task_data).execute()
            logger.info(f"存储任务 {task_data.get('task_id')} 成功，响应数量: {len(response.data) if response.data else 0}")
            return response
        except Exception as e:
            logger.error(f"存储任务 {task_data.get('task_id')} 失败: {e}", exc_info=True)
            return None

    async def update_task(self, task_id, update_data):
        """更新任务信息"""
        try:
            client = await self._ensure_client()
            response = await client.table('tasks').update(update_data).eq('task_id', task_id).execute()
            logger.info(f"更新任务 {task_id} 成功，响应数量: {len(response.data) if response.data else 0}")
            return response
        except Exception as e:
            logger.error(f"更新任务 {task_id} 失败: {e}", exc_info=True)
            return None

    async def get_task(self, task_id):
        """获取任务信息"""
        try:
            client = await self._ensure_client()
            response = await client.table('tasks').select('*').eq('task_id', task_id).execute()
            logger.info(f"获取任务 {task_id}，找到: {len(response.data) > 0}")
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"获取任务 {task_id} 失败: {e}", exc_info=True)
            return None

    async def store_sentences(self, sentences, task_id):
        """批量存储句子信息"""
        if not sentences:
            logger.warning(f"任务 {task_id} 没有句子数据需要存储")
            return None
            
        try:
            client = await self._ensure_client()
            json_sentences = []
            
            for idx, s in enumerate(sentences):
                # 从原始对象提取数据
                speaker_id = getattr(s, 'speaker_id', -1)
                start_ms = getattr(s, 'start', 0)
                end_ms = getattr(s, 'end', 0)
                
                # 计算目标持续时间
                target_duration = getattr(s, 'target_duration', None)
                target_duration_ms = target_duration if target_duration is not None else (end_ms - start_ms)
                target_duration_ms = max(0, target_duration_ms)

                # 构建句子数据
                sentence_data = {
                    'task_id': task_id,
                    'sentence_index': idx,
                    'raw_text': getattr(s, 'raw_text', ''),
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'speaker_id': speaker_id,
                    'target_duration_ms': target_duration_ms,
                    'speech_duration_ms': getattr(s, 'speech_duration', 0.0),
                    'audio_prompt_path': getattr(s, 'audio', None),
                    'is_first': getattr(s, 'is_first', False),
                    'is_last': getattr(s, 'is_last', False),
                    'ending_silence_ms': getattr(s, 'ending_silence', 0.0)
                }
                
                # 处理特殊类型数据
                json_sentences.append(sanitize_for_json(sentence_data))

            # 执行数据库插入
            if json_sentences:
                response = await client.table('sentences').insert(json_sentences).execute()
                logger.info(f"存储 {len(json_sentences)} 个句子到任务 {task_id}，响应数量: {len(response.data) if response.data else 0}")
                return response
            else:
                logger.warning(f"任务 {task_id}: 没有有效的句子数据可存储")
                return None
                
        except AttributeError as ae:
            logger.error(f"处理句子属性时出错 (任务 {task_id}): {ae}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"存储句子失败 (任务 {task_id}): {e}", exc_info=True)
            return None

    async def get_sentences(self, task_id, as_objects=False):
        """
        获取任务的所有句子，按索引排序
        
        Args:
            task_id: 任务ID
            as_objects: 是否将结果转换为Sentence对象列表
        """
        try:
            client = await self._ensure_client()
            response = await client.table('sentences').select('*').eq('task_id', task_id).order('sentence_index').execute()
            logger.info(f"获取任务 {task_id} 的句子，数量: {len(response.data)}")
            
            # 如果不需要转换为对象，直接返回原始数据
            if not as_objects:
                return response.data
                
            # 将原始数据转换为Sentence对象
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
                # 从数据库记录中读取 speech_duration_ms 并赋值
                sentence.speech_duration = data.get('speech_duration_ms', 0.0)
                
                sentences.append(sentence)
                
            logger.info(f"成功将 {len(sentences)} 个句子转换为对象 (任务 {task_id})")
            return sentences
            
        except Exception as e:
            logger.error(f"获取句子失败 (任务 {task_id}): {e}", exc_info=True)
            return [] 

    async def update_sentence_translation(self, task_id: str, sentence_index: int, trans_text: str):
        """更新单个句子的翻译文本"""
        try:
            client = await self._ensure_client()
            # 确保 trans_text 不是 None，如果是 None，可以考虑存储空字符串或按需处理
            update_data = {'trans_text': trans_text if trans_text is not None else ""}
            response = await client.table('sentences').update(update_data).eq('task_id', task_id).eq('sentence_index', sentence_index).execute()
            
            # 更详细的日志和错误检查
            if response.data and len(response.data) > 0:
                # logger.debug(f"更新句子翻译 {task_id}-{sentence_index} 成功.")
                pass
            elif response.status_code not in [200, 201, 204]: # 201 for insert, 204 for no content success
                logger.error(f"更新句子翻译 {task_id}-{sentence_index} 可能失败。状态码: {response.status_code}, 响应: {response.error}")
            # else: logger.debug(f"更新句子翻译 {task_id}-{sentence_index} 未找到匹配项或无内容更新.")

            return response
        except Exception as e:
            logger.error(f"更新句子翻译 {task_id}-{sentence_index} 异常: {e}", exc_info=True)
            return None 