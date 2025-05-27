import logging
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional
from core.supabase_client import SupabaseClient
from config import Config

logger = logging.getLogger(__name__)

class HLSStorageManager:
    """HLS存储管理器 - 负责将HLS文件上传到Supabase Storage"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.supabase_client = SupabaseClient(config=self.config)
        self.logger = logging.getLogger(__name__)
        
        # HLS文件存储的bucket名称
        self.hls_bucket = self.config.HLS_STORAGE_BUCKET
        
        self.logger.info("HLS存储管理器已初始化")
    
    async def upload_segment(self, task_id: str, segment_file_path: str, segment_name: str) -> Dict:
        """
        上传单个TS分段文件到Supabase Storage
        
        Args:
            task_id: 任务ID
            segment_file_path: 本地分段文件路径
            segment_name: 分段文件名
            
        Returns:
            Dict: 包含上传结果的字典
        """
        try:
            # 新的存储路径：每个任务一个文件夹
            storage_path = f"{task_id}/{segment_name}"
            
            # 读取文件内容
            async with aiofiles.open(segment_file_path, 'rb') as f:
                file_bytes = await f.read()
            
            # 上传到Supabase Storage，使用upsert模式覆盖现有文件
            result = await self.supabase_client.upload_file(
                bucket_name=self.hls_bucket,
                storage_path=storage_path,
                file_bytes=file_bytes,
                upsert=True
            )
            
            self.logger.info(f"[{task_id}] 成功上传分段文件: {segment_name}")
            return {
                "status": "success",
                "storage_path": storage_path,
                "segment_name": segment_name,
                "file_size": len(file_bytes)
            }
            
        except Exception as e:
            self.logger.error(f"[{task_id}] 上传分段文件失败 {segment_name}: {e}")
            return {
                "status": "error",
                "message": f"上传分段文件失败: {str(e)}",
                "segment_name": segment_name
            }
    
    async def upload_playlist(self, task_id: str, playlist_content: str) -> Dict:
        """
        上传M3U8播放列表到Supabase Storage
        
        Args:
            task_id: 任务ID
            playlist_content: M3U8播放列表内容
            
        Returns:
            Dict: 包含上传结果的字典
        """
        try:
            # 新的存储路径：每个任务一个文件夹
            storage_path = f"{task_id}/playlist.m3u8"
            
            # 将字符串转换为字节
            file_bytes = playlist_content.encode('utf-8')
            
            # 上传到Supabase Storage，使用upsert模式覆盖现有文件
            result = await self.supabase_client.upload_file(
                bucket_name=self.hls_bucket,
                storage_path=storage_path,
                file_bytes=file_bytes,
                upsert=True
            )
            
            self.logger.info(f"[{task_id}] 成功上传播放列表: {storage_path}")
            return {
                "status": "success",
                "storage_path": storage_path,
                "file_size": len(file_bytes),
                "public_url": f"{self.config.SUPABASE_URL}/storage/v1/object/public/{self.hls_bucket}/{storage_path}"
            }
            
        except Exception as e:
            self.logger.error(f"[{task_id}] 上传播放列表失败: {e}")
            return {
                "status": "error",
                "message": f"上传播放列表失败: {str(e)}"
            }
    
    async def batch_upload_segments(self, task_id: str, segment_files: List[str]) -> Dict:
        """
        批量上传多个TS分段文件
        
        Args:
            task_id: 任务ID
            segment_files: 分段文件路径列表
            
        Returns:
            Dict: 包含批量上传结果的字典
        """
        uploaded_segments = []
        failed_segments = []
        
        # 并发上传分段文件（限制并发数量避免过载）
        semaphore = asyncio.Semaphore(5)  # 最多同时上传5个文件
        
        async def upload_single_segment(segment_path: str):
            async with semaphore:
                segment_name = Path(segment_path).name
                result = await self.upload_segment(task_id, segment_path, segment_name)
                if result["status"] == "success":
                    uploaded_segments.append(result)
                else:
                    failed_segments.append(result)
        
        # 创建上传任务
        upload_tasks = [upload_single_segment(segment_path) for segment_path in segment_files]
        
        # 等待所有上传完成
        await asyncio.gather(*upload_tasks, return_exceptions=True)
        
        self.logger.info(f"[{task_id}] 批量上传完成: 成功 {len(uploaded_segments)}, 失败 {len(failed_segments)}")
        
        return {
            "status": "success" if not failed_segments else "partial",
            "uploaded_count": len(uploaded_segments),
            "failed_count": len(failed_segments),
            "uploaded_segments": uploaded_segments,
            "failed_segments": failed_segments
        }
    
    async def update_playlist_with_storage_urls(self, playlist_content: str, task_id: str) -> str:
        """
        更新播放列表中的分段URL为相对路径（适用于新的存储结构）
        
        Args:
            playlist_content: 原始播放列表内容
            task_id: 任务ID
            
        Returns:
            str: 更新后的播放列表内容
        """
        try:
            lines = playlist_content.split('\n')
            updated_lines = []
            
            for line in lines:
                # 如果是TS文件行，使用相对路径
                if line.strip().endswith('.ts'):
                    segment_name = Path(line.strip()).name
                    # 使用相对路径，因为m3u8和ts文件在同一个文件夹下
                    updated_line = segment_name
                    updated_lines.append(updated_line)
                else:
                    updated_lines.append(line)
            
            return '\n'.join(updated_lines)
            
        except Exception as e:
            self.logger.error(f"[{task_id}] 更新播放列表URL失败: {e}")
            return playlist_content
    
    async def cleanup_local_files(self, task_id: str, file_paths: List[str]) -> Dict:
        """
        清理本地HLS文件（在上传到Storage后）
        
        Args:
            task_id: 任务ID
            file_paths: 要清理的文件路径列表
            
        Returns:
            Dict: 清理结果
        """
        cleaned_files = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                path_obj = Path(file_path)
                if path_obj.exists():
                    await asyncio.to_thread(path_obj.unlink)
                    cleaned_files.append(file_path)
                    self.logger.debug(f"[{task_id}] 已清理本地文件: {file_path}")
            except Exception as e:
                failed_files.append({"file_path": file_path, "error": str(e)})
                self.logger.warning(f"[{task_id}] 清理本地文件失败 {file_path}: {e}")
        
        self.logger.info(f"[{task_id}] 本地文件清理完成: 成功 {len(cleaned_files)}, 失败 {len(failed_files)}")
        
        return {
            "status": "success" if not failed_files else "partial",
            "cleaned_count": len(cleaned_files),
            "failed_count": len(failed_files),
            "cleaned_files": cleaned_files,
            "failed_files": failed_files
        }
    
    async def get_playlist_public_url(self, task_id: str) -> str:
        """
        获取播放列表的公共访问URL
        
        Args:
            task_id: 任务ID
            
        Returns:
            str: 播放列表的公共URL
        """
        storage_path = f"{task_id}/playlist.m3u8"
        return f"{self.config.SUPABASE_URL}/storage/v1/object/public/{self.hls_bucket}/{storage_path}"
    
    async def check_playlist_exists(self, task_id: str) -> bool:
        """
        检查播放列表是否已存在
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 播放列表是否存在
        """
        try:
            storage_path = f"{task_id}/playlist.m3u8"
            # 尝试下载文件来检查是否存在
            await self.supabase_client.download_file(self.hls_bucket, storage_path)
            return True
        except Exception:
            return False
    
    async def get_existing_playlist_content(self, task_id: str) -> Optional[str]:
        """
        获取已存在的播放列表内容
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[str]: 播放列表内容，如果不存在则返回None
        """
        try:
            storage_path = f"{task_id}/playlist.m3u8"
            file_bytes = await self.supabase_client.download_file(self.hls_bucket, storage_path)
            return file_bytes.decode('utf-8')
        except Exception as e:
            self.logger.debug(f"[{task_id}] 获取现有播放列表失败: {e}")
            return None 