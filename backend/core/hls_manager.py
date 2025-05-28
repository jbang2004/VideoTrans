# core/hls_manager.py
import logging
import m3u8
import os.path
import shutil
import time
import asyncio
from pathlib import Path
from typing import Union, Optional, Dict, List
from ray import serve

import aiofiles
from utils.ffmpeg_utils import hls_segment, concat_videos
from utils.task_storage import TaskPaths
from config import Config
from core.supabase_client import SupabaseClient
from core.hls_storage_manager import HLSStorageManager

logger = logging.getLogger(__name__)

@serve.deployment(
    name="hls_manager",
    logging_config={"log_level": "INFO"}
)
class HLSManager:
    """HLS流媒体管理器 - 支持多任务管理"""
    def __init__(self):
        self.config = Config()
        self.supabase_client = SupabaseClient(config=self.config)
        self.hls_storage_manager = HLSStorageManager(config=self.config)
        self.logger = logging.getLogger(__name__)
        
        # 存储每个任务的HLS管理信息
        self.task_managers = {}
        
        # 并发锁，每个任务一个锁
        self.locks = {}
        
        self.logger.info("HLS管理器已初始化")

    async def create_manager(self, task_id: str, task_paths: TaskPaths) -> Dict:
        """
        为特定任务创建HLS管理器
        
        Args:
            task_id: 任务ID
            task_paths: 任务路径信息
            
        Returns:
            Dict: 包含状态信息的字典
        """
        # 如果任务已存在，直接返回
        if task_id in self.task_managers:
            self.logger.info(f"任务 {task_id} 的HLS管理器已存在")
            return {"status": "success", "message": "任务HLS管理器已存在"}
        
        # 创建任务锁
        self.locks[task_id] = asyncio.Lock()
        
        async with self.locks[task_id]:
            try:
                playlist_path = Path(task_paths.playlist_path)
                segments_dir = Path(task_paths.segments_dir)
                
                # 创建播放列表
                playlist = m3u8.M3U8()
                playlist.version = 3
                playlist.target_duration = 10  # 与实际分段时长保持一致
                playlist.media_sequence = 0
                playlist.playlist_type = 'EVENT'  # EVENT类型支持实时流
                playlist.is_endlist = False
                
                # 优化流式播放的配置
                playlist.allow_cache = False  # 禁用缓存，确保播放器获取最新版本
                playlist.program_date_time = None  # 可以添加时间戳支持
                
                sequence_number = 0
                has_segments = False
                
                # 尝试从Storage恢复现有播放列表
                if self.config.ENABLE_HLS_STORAGE:
                    try:
                        existing_content = await self.hls_storage_manager.get_existing_playlist_content(task_id)
                        if existing_content:
                            existing_playlist = m3u8.loads(existing_content)
                            if existing_playlist.segments:
                                # 恢复现有片段
                                playlist.segments = existing_playlist.segments.copy()
                                sequence_number = len(existing_playlist.segments)
                                has_segments = True
                                playlist.media_sequence = existing_playlist.media_sequence or 0
                                self.logger.info(f"[{task_id}] 从Storage恢复了 {len(existing_playlist.segments)} 个HLS片段")
                    except Exception as e:
                        self.logger.debug(f"[{task_id}] 无法从Storage恢复播放列表: {e}")
                
                # 存储任务管理信息
                self.task_managers[task_id] = {
                    "task_paths": task_paths,
                    "playlist_path": playlist_path,
                    "segments_dir": segments_dir,
                    "playlist": playlist,
                    "sequence_number": sequence_number,
                    "has_segments": has_segments,
                    "segment_time": 10,  # 默认分段时间为10秒
                    "created_at": time.time()
                }
                
                # 保存初始播放列表
                await self._save_playlist(task_id)
                
                self.logger.info(f"已为任务 {task_id} 创建HLS管理器")
                return {"status": "success", "message": "HLS管理器创建成功"}
            except Exception as e:
                error_message = f"为任务 {task_id} 创建HLS管理器失败: {e}"
                self.logger.error(error_message)
                # Update Supabase task status to error
                if self.supabase_client:
                    try:
                        asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': f"HLS管理器初始化失败: {e}"}))
                    except Exception as db_update_e:
                        self.logger.error(f"任务 {task_id}: 更新数据库状态失败 (HLS创建失败时): {db_update_e}")

                # 清理已创建的部分资源
                if task_id in self.task_managers:
                    del self.task_managers[task_id]
                if task_id in self.locks:
                    del self.locks[task_id]
                return {"status": "error", "message": f"创建HLS管理器失败: {str(e)}"}

    async def _save_playlist(self, task_id: str) -> None:
        """
        保存特定任务的播放列表到文件
        
        Args:
            task_id: 任务ID
        """
        if task_id not in self.task_managers:
            raise ValueError(f"任务 {task_id} 的HLS管理器不存在")
            
        manager = self.task_managers[task_id]
        playlist = manager["playlist"]
        playlist_path = manager["playlist_path"]
        
        try:
            for segment in playlist.segments:
                # 确保 URI 带有斜杠
                if segment.uri is not None and not segment.uri.startswith('/'):
                    segment.uri = '/' + segment.uri

            self.logger.info(f"保存播放列表到: {playlist_path}, 任务ID={task_id}")
            # 确保目录存在
            playlist_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用asyncio.to_thread避免阻塞
            await asyncio.to_thread(self._write_playlist, playlist, playlist_path)
                
            self.logger.info(f"播放列表已更新，总计{len(playlist.segments)}个分段, 任务ID={task_id}")
        except Exception as e:
            self.logger.error(f"保存播放列表失败: {e}, 任务ID={task_id}")
            raise
    
    def _write_playlist(self, playlist, playlist_path):
        """同步写入播放列表文件"""
        with open(playlist_path, 'w', encoding='utf-8') as f:
            content = playlist.dumps()
            f.write(content)
    
    async def _upload_playlist_to_storage(self, task_id: str) -> None:
        """
        将播放列表上传到Supabase Storage（支持增量更新）
        
        Args:
            task_id: 任务ID
        """
        if task_id not in self.task_managers:
            raise ValueError(f"任务 {task_id} 的HLS管理器不存在")
            
        try:
            manager = self.task_managers[task_id]
            playlist = manager["playlist"]
            
            # 获取播放列表内容并更新为相对路径
            playlist_content = playlist.dumps()
            updated_content = await self.hls_storage_manager.update_playlist_with_storage_urls(
                playlist_content, task_id
            )
            
            # 上传到Storage（这会覆盖之前的版本，但内存中的playlist对象包含了所有片段）
            upload_result = await self.hls_storage_manager.upload_playlist(task_id, updated_content)
            
            if upload_result["status"] == "success":
                self.logger.info(f"[{task_id}] 播放列表已上传到Storage: {upload_result['storage_path']} (包含 {len(playlist.segments)} 个片段)")
                
                # 更新数据库中的HLS播放列表URL为Storage的公共URL
                storage_url = upload_result["public_url"]
                if self.supabase_client:
                    try:
                        asyncio.create_task(self.supabase_client.update_task(task_id, {
                            'hls_playlist_url': storage_url,
                        }))
                        self.logger.info(f"[{task_id}] HLS播放列表Storage URL已更新到数据库: {storage_url}")
                    except Exception as update_e:
                        self.logger.error(f"[{task_id}] 更新HLS播放列表Storage URL到数据库失败: {update_e}")
            else:
                self.logger.error(f"[{task_id}] 播放列表上传到Storage失败: {upload_result.get('message', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"[{task_id}] 上传播放列表到Storage失败: {e}")
            raise

    async def add_segment(self, task_id: str, video_path: Union[str, Path], part_index: int) -> Dict:
        """
        添加新的视频片段到播放列表
        
        Args:
            task_id: 任务ID
            video_path: 视频片段路径
            part_index: 部分索引
            
        Returns:
            Dict: 包含状态信息的字典
        """
        if task_id not in self.task_managers:
            self.logger.error(f"任务 {task_id} 的HLS管理器不存在")
            return {"status": "error", "message": "任务HLS管理器不存在"}
            
        if task_id not in self.locks:
            self.locks[task_id] = asyncio.Lock()
            
        start_time = time.time()
        
        async with self.locks[task_id]:
            try:
                manager = self.task_managers[task_id]
                segments_dir = manager["segments_dir"]
                playlist = manager["playlist"]
                sequence_number = manager["sequence_number"]
                segment_time = manager["segment_time"]
                task_paths = manager["task_paths"]
                
                was_first_segment = not manager["has_segments"]

                self.logger.info(f"开始处理HLS片段 {part_index}, 任务ID={task_id}")
                segments_dir.mkdir(parents=True, exist_ok=True)

                segment_filename = f'segment_{sequence_number:04d}_%03d.ts'
                segment_pattern = str(segments_dir / segment_filename)
                temp_playlist_path = task_paths.processing_dir / f'temp_{part_index}.m3u8'

                # 使用异步函数
                await hls_segment(
                    input_path=str(video_path),
                    segment_pattern=segment_pattern,
                    playlist_path=str(temp_playlist_path),
                    hls_time=segment_time
                )

                # 加入分段
                # 使用asyncio.to_thread避免阻塞
                temp_m3u8 = await asyncio.to_thread(m3u8.load, str(temp_playlist_path))
                
                discontinuity_segment = m3u8.Segment(discontinuity=True)
                playlist.add_segment(discontinuity_segment)

                # 收集新生成的分段文件路径，用于上传到Storage
                new_segment_files = []
                for segment in temp_m3u8.segments:
                    local_segment_path = segments_dir / Path(segment.uri).name
                    new_segment_files.append(str(local_segment_path))
                    # 使用相对路径，因为m3u8和ts文件在同一个文件夹下
                    segment.uri = Path(segment.uri).name
                    playlist.segments.append(segment)

                # 上传新分段文件到Supabase Storage（如果启用）
                if self.config.ENABLE_HLS_STORAGE and new_segment_files:
                    upload_result = await self.hls_storage_manager.batch_upload_segments(task_id, new_segment_files)
                    if upload_result["status"] in ["success", "partial"]:
                        self.logger.info(f"[{task_id}] 分段文件上传完成: {upload_result['uploaded_count']}/{len(new_segment_files)}")
                    else:
                        self.logger.warning(f"[{task_id}] 分段文件上传失败")

                # 更新序列号
                manager["sequence_number"] += len(temp_m3u8.segments)
                manager["has_segments"] = True
                
                # 确保播放列表不标记为结束，以便实时加载
                playlist.is_endlist = False
                
                # 保存本地播放列表
                await self._save_playlist(task_id)
                
                # 上传更新后的播放列表到Storage（如果启用）
                if self.config.ENABLE_HLS_STORAGE:
                    await self._upload_playlist_to_storage(task_id)

                # HLS播放列表URL现在在_upload_playlist_to_storage方法中更新为Storage URL
                
                # 删除临时播放列表
                if temp_playlist_path.exists():
                    await asyncio.to_thread(temp_playlist_path.unlink)
                
                elapsed = time.time() - start_time
                self.logger.info(f"已添加片段 {part_index} 到HLS流，耗时 {elapsed:.2f}s, 任务ID={task_id}")
                return {"status": "success", "part_index": part_index}
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"添加HLS片段失败: {e}，耗时 {elapsed:.2f}s, 任务ID={task_id}")
                return {"status": "error", "message": f"添加HLS片段失败: {str(e)}"}

    async def finalize_playlist(self, task_id: str) -> Dict:
        """
        标记播放列表为完成状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 包含状态信息的字典
        """
        if task_id not in self.task_managers:
            self.logger.error(f"任务 {task_id} 的HLS管理器不存在")
            return {"status": "error", "message": "任务HLS管理器不存在"}
            
        if task_id not in self.locks:
            self.locks[task_id] = asyncio.Lock()
            
        async with self.locks[task_id]:
            try:
                manager = self.task_managers[task_id]
                playlist = manager["playlist"]
                has_segments = manager["has_segments"]
                
                if has_segments:
                    playlist.is_endlist = True
                    await self._save_playlist(task_id)
                    
                    # 上传最终的播放列表到Storage（如果启用）
                    if self.config.ENABLE_HLS_STORAGE:
                        await self._upload_playlist_to_storage(task_id)
                        self.logger.info(f"播放列表已保存并上传到Storage，标记为完成状态, 任务ID={task_id}")
                        return {"status": "success", "message": "播放列表已标记为完成并上传到Storage"}
                    else:
                        self.logger.info(f"播放列表已保存，标记为完成状态, 任务ID={task_id}")
                        return {"status": "success", "message": "播放列表已标记为完成"}
                else:
                    self.logger.warning(f"播放列表为空，不标记为结束状态, 任务ID={task_id}")
                    return {"status": "warning", "message": "播放列表为空，未标记为完成"}
            except Exception as e:
                self.logger.error(f"完成播放列表失败: {e}, 任务ID={task_id}")
                return {"status": "error", "message": f"完成播放列表失败: {str(e)}"}

    async def get_has_segments(self, task_id: str) -> Dict:
        """
        获取任务是否有分段
        
        Args:
            task_id: 任务ID
            
        Returns:
            Dict: 包含状态和has_segments值的字典
        """
        if task_id not in self.task_managers:
            return {"status": "error", "message": "任务HLS管理器不存在", "has_segments": False}
            
        manager = self.task_managers[task_id]
        return {"status": "success", "has_segments": manager["has_segments"]}
    
    async def clean_old_tasks(self, max_age_hours: int = 24) -> Dict:
        """
        清理旧任务资源
        
        Args:
            max_age_hours: 最大保留小时数，默认24小时
            
        Returns:
            Dict: 包含清理信息的字典
        """
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        tasks_to_clean = []
        
        # 标识需要清理的任务
        for task_id, manager in self.task_managers.items():
            if now - manager["created_at"] > max_age_seconds:
                tasks_to_clean.append(task_id)
        
        # 执行清理
        cleaned_count = 0
        for task_id in tasks_to_clean:
            if task_id in self.locks:
                async with self.locks[task_id]:
                    if task_id in self.task_managers:
                        del self.task_managers[task_id]
                        cleaned_count += 1
                del self.locks[task_id]
        
        self.logger.info(f"已清理 {cleaned_count} 个过期任务的HLS资源")
        return {"status": "success", "cleaned_count": cleaned_count}

    async def finalize_merge(self, task_id: str, all_processed_segment_paths: List[str], task_paths: TaskPaths) -> Dict:
        """最终化任务处理：结束播放列表，合并片段，并更新数据库状态"""
        self.logger.info(f"[{task_id}] HLSManager: 开始最终化任务处理，包含 {len(all_processed_segment_paths)} 个片段。")

        # 1. 结束HLS播放列表
        await self.finalize_playlist(task_id)
        self.logger.info(f"[{task_id}] HLSManager: 播放列表已标记为结束并上传到Storage。")

        # 2. 合并处理好的视频片段
        self.logger.info(f"[{task_id}] HLSManager: 开始合并 {len(all_processed_segment_paths)} 个视频片段。")
        
        # 处理无片段情况
        if not all_processed_segment_paths:
            msg = "HLSManager: 没有处理成功的视频片段可以合并。"
            self.logger.error(f"[{task_id}] {msg}")
            if self.supabase_client:
                try:
                    asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': msg}))
                except Exception as db_update_e:
                    self.logger.error(f"[{task_id}] HLSManager: 更新数据库状态(无片段)失败: {db_update_e}")
            return {"status": "error", "message": msg}

        try:
            # 确保输出目录存在
            task_paths.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"[{task_id}] HLSManager: 确保输出目录存在: {task_paths.output_dir}")
            
            # 创建合并列表文件
            list_txt_path = task_paths.processing_dir / "concat_list.txt"
            async with aiofiles.open(list_txt_path, "w", encoding='utf-8') as f:
                for seg_mp4 in all_processed_segment_paths:
                    # 确保路径是绝对的并且格式正确
                    formatted_path = str(Path(seg_mp4).resolve()).replace("\\", "/")
                    await f.write(f"file '{formatted_path}'\n")
            self.logger.info(f"[{task_id}] HLSManager: 合并列表文件已创建: {list_txt_path}")

            # 执行视频合并
            final_output_path = task_paths.output_dir / f"final_{task_id}.mp4"
            merge_start_time = time.time()
            self.logger.info(f"[{task_id}] HLSManager: 开始调用 concat_videos, 输出到 {final_output_path}")
            
            # concat_videos 应该是异步的，如果不是，需要用 asyncio.to_thread 包装
            # 假设 concat_videos 是异步的
            final_video_path_obj = await concat_videos(str(list_txt_path), str(final_output_path))

            # 处理合并结果
            if not final_video_path_obj or not final_video_path_obj.exists():
                msg = "HLSManager: 视频合并失败，最终文件未生成。"
                self.logger.error(f"[{task_id}] {msg}")
                if self.supabase_client:
                    try:
                        asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': msg}))
                    except Exception as db_update_e:
                         self.logger.error(f"[{task_id}] HLSManager: 更新数据库状态(合并失败)失败: {db_update_e}")
                return {"status": "error", "message": msg}

            # 完成流程并更新状态
            final_video_path_str = str(final_video_path_obj)
            merge_duration = time.time() - merge_start_time
            
            if self.supabase_client:
                try:
                    asyncio.create_task(self.supabase_client.update_task(task_id, {
                        'status': 'completed',
                        'download_video_path': final_video_path_str,
                    }))
                    self.logger.info(f"[{task_id}] HLSManager: 任务状态成功，下载路径已异步更新到数据库。")
                except Exception as db_update_e:
                    self.logger.error(f"[{task_id}] HLSManager: 更新数据库状态(成功)失败: {db_update_e}")
                    # 即使数据库更新失败，合并本身是成功的，所以仍然返回成功
                    return {"status": "success", "message": "视频处理成功，但数据库更新失败", "output_path": final_video_path_str}

            # 3. 清理本地HLS文件（如果启用Storage且配置了清理）
            if self.config.ENABLE_HLS_STORAGE and self.config.CLEANUP_LOCAL_HLS_FILES:
                try:
                    if task_id in self.task_managers:
                        manager = self.task_managers[task_id]
                        segments_dir = manager["segments_dir"]
                        playlist_path = manager["playlist_path"]
                        
                        # 收集所有本地HLS文件
                        local_hls_files = []
                        if segments_dir.exists():
                            local_hls_files.extend([str(f) for f in segments_dir.glob("*.ts")])
                        if playlist_path.exists():
                            local_hls_files.append(str(playlist_path))
                        
                        if local_hls_files:
                            cleanup_result = await self.hls_storage_manager.cleanup_local_files(task_id, local_hls_files)
                            self.logger.info(f"[{task_id}] HLS本地文件清理完成: {cleanup_result['cleaned_count']}/{len(local_hls_files)}")
                except Exception as cleanup_e:
                    self.logger.warning(f"[{task_id}] HLS本地文件清理失败: {cleanup_e}")

            self.logger.info(f"[{task_id}] HLSManager: 视频合并成功，耗时: {merge_duration:.2f}s")
            return {"status": "success", "message": "视频处理成功", "output_path": final_video_path_str}
            
        except Exception as e:
            msg = f"HLSManager: 视频合并过程中出错: {e}"
            self.logger.exception(f"[{task_id}] {msg}") # Log with stack trace
            if self.supabase_client:
                try:
                    asyncio.create_task(self.supabase_client.update_task(task_id, {'status': 'error', 'error_message': msg}))
                except Exception as db_update_e:
                    self.logger.error(f"[{task_id}] HLSManager: 更新数据库状态(合并异常)失败: {db_update_e}")
            return {"status": "error", "message": msg} 