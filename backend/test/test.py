#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import uuid
import asyncio
import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 添加项目路径
project_dir = Path(__file__).parent.parent.parent
sys.path.append(str(project_dir))
sys.path.append(str(project_dir / "backend"))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("load_test")


async def run_task(task_id: str, pipeline, video_path: str, target_language: str) -> Tuple[Dict, Dict]:
    """运行单个视频处理任务并记录性能数据
    
    Args:
        task_id: 任务ID
        pipeline: Ray Serve部署的pipeline句柄
        video_path: 视频文件路径
        target_language: 目标语言
        
    Returns:
        包含任务结果和性能指标的元组 (result, performance)
    """
    logger.info(f"开始处理任务: {task_id}, 视频: {video_path}")
    
    try:
        # 使用最新的Ray API调用方式调用pipeline处理视频
        result = await pipeline.remote(
            task_id=task_id,
            video_path=video_path,
            target_language=target_language,
            generate_subtitle=False
        )
        
        # 检查任务状态和性能指标
        if not result or not isinstance(result, dict):
            logger.error(f"任务 {task_id} 返回的结果无效或不是字典类型")
            return {"status": "error", "message": "无效的返回结果"}, {}
            
        status = result.get("status")
        message = result.get("message", "")
        logger.info(f"任务 {task_id} 状态: {status}, 消息: {message}")
        
        # 获取并打印性能指标
        perf = result.get("performance", {})
        if perf and "total" in perf and "duration" in perf["total"]:
            total_time = perf["total"]["duration"]
            logger.info(f"任务 {task_id} 总处理时间: {total_time:.2f}秒")
            
        # 打印各阶段性能指标
        if perf and "stages" in perf:
            stages = perf["stages"]
            if "segmenter" in stages and "duration" in stages["segmenter"]:
                segmenter_time = stages["segmenter"]["duration"]
                logger.info(f"任务 {task_id} 视频分段时间: {segmenter_time:.2f}秒")
            
            if "segments_processing" in stages and "duration" in stages["segments_processing"]:
                processing_time = stages["segments_processing"]["duration"]
                logger.info(f"任务 {task_id} 分段处理时间: {processing_time:.2f}秒")
            
            if "merge" in stages and "duration" in stages["merge"]:
                merge_time = stages["merge"]["duration"]
                logger.info(f"任务 {task_id} 视频合并时间: {merge_time:.2f}秒")
            
        return result, perf
    except Exception as e:
        logger.error(f"任务 {task_id} 处理失败: {str(e)}")
        return {"status": "error", "message": str(e)}, {}


async def run_parallel_tasks(video_path: str, num_tasks: int, target_language: str) -> Dict:
    """并行运行多个视频处理任务"""
    # 直接连接到已有的Ray集群
    import ray
    from ray import serve
    
    if not ray.is_initialized():
        ray.init(address="auto", namespace="videotrans", ignore_reinit_error=True)
        logger.info("已连接到Ray集群")
    
    # 检查核心服务是否已部署
    try:
        pipeline = serve.get_deployment_handle("VideoTransPipe", app_name="PipelineEngine")
        logger.info("成功获取Pipeline句柄")
    except Exception as e:
        logger.error(f"获取Pipeline句柄失败，请确保pipeline_scheduler已经启动: {e}")
        raise
    
    # 创建并行任务
    tasks = []
    task_ids = []
    
    logger.info(f"开始并行处理 {num_tasks} 个任务...")
    start_time = time.time()
    
    # 准备所有任务
    for i in range(num_tasks):
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        tasks.append(run_task(task_id, pipeline, video_path, target_language))
    
    # 使用asyncio.gather更高效地并行执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    task_results = {}
    for i, result in enumerate(results):
        task_id = task_ids[i]
        
        if isinstance(result, Exception):
            # 处理异常情况
            logger.error(f"任务 {task_id} 执行异常: {str(result)}")
            task_results[task_id] = ({"status": "error", "message": str(result)}, {})
        else:
            # 正常结果
            task_results[task_id] = result
    
    # 计算总体运行时间
    total_time = time.time() - start_time
    logger.info(f"所有任务处理完成，总耗时: {total_time:.2f}秒")
    
    return task_results


def write_results_to_file(results: Dict, video_path: str, num_tasks: int, target_language: str, output_file: str) -> None:
    """将测试结果写入文件"""
    with open(output_file, "w", encoding="utf-8") as f:
        # 写入测试基本信息
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"负载测试结果 - {timestamp}\n")
        f.write(f"视频: {video_path}\n")
        f.write(f"任务数量: {num_tasks}\n")
        f.write(f"目标语言: {target_language}\n\n")
        
        # 统计成功和失败任务
        successful_tasks = []
        failed_tasks = []
        
        for task_id, (result, perf) in results.items():
            if result.get("status") == "success":
                successful_tasks.append((task_id, result, perf))
            else:
                failed_tasks.append((task_id, result, perf))
        
        # 写入摘要
        f.write("摘要:\n")
        f.write(f"成功任务: {len(successful_tasks)}/{num_tasks}\n")
        f.write(f"失败任务: {len(failed_tasks)}/{num_tasks}\n\n")
        
        # 计算平均总处理时间（只考虑成功任务）
        if successful_tasks:
            total_times = []
            for _, _, perf in successful_tasks:
                if perf and "total" in perf and "duration" in perf["total"]:
                    total_times.append(perf["total"]["duration"])
            
            if total_times:
                avg_total = sum(total_times) / len(total_times)
                f.write(f"平均总处理时间: {avg_total:.2f}秒\n\n")
        
        # 为所有任务写入详细信息（合并"详细结果"和"分段处理时间详情"）
        f.write("各任务详细信息:\n")
        
        # 先处理成功任务
        for task_id, result, perf in successful_tasks:
            f.write(f"任务ID: {task_id}\n")
            f.write(f"状态: 成功\n")
            
            if "message" in result and result["message"]:
                f.write(f"消息: {result['message']}\n")
            
            # 总处理时间
            if "total" in perf and "duration" in perf["total"]:
                f.write(f"总处理时间: {perf['total']['duration']:.2f}秒\n")
            
            # 各分段处理时间
            if "segments_processing" in perf and perf["segments_processing"]:
                # 按分段索引排序
                sorted_segments = sorted(perf["segments_processing"], key=lambda x: x["segment_index"])
                f.write(f"分段数量: {len(sorted_segments)}\n")
                f.write("各分段处理时间:\n")
                
                for seg in sorted_segments:
                    f.write(f"  分段 {seg['segment_index']}: {seg['duration']:.2f}秒\n")
            
            f.write("\n")
        
        # 处理失败任务
        for task_id, result, perf in failed_tasks:
            f.write(f"任务ID: {task_id}\n")
            f.write(f"状态: 失败\n")
            
            if "message" in result and result["message"]:
                f.write(f"错误信息: {result['message']}\n")
            
            # 总处理时间(如果有)
            if perf and "total" in perf and "duration" in perf["total"]:
                f.write(f"处理时间: {perf['total']['duration']:.2f}秒\n")
            
            f.write("\n")
    
    logger.info(f"结果已写入文件: {output_file}")


async def main():
    """主函数：解析命令行参数并执行负载测试"""
    parser = argparse.ArgumentParser(description="视频翻译负载测试工具")
    parser.add_argument("-n", "--num_tasks", type=int, default=1, help="并行任务数量")
    parser.add_argument("-v", "--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("-l", "--language", type=str, default="en", help="目标语言 (zh, en, ja, ko)")
    parser.add_argument("-o", "--output", type=str, default="负载测试结果.txt", help="输出结果文件")
    
    args = parser.parse_args()
    
    # 确保视频文件路径正确
    video_path = args.video
    if not Path(video_path).is_absolute():
        # 尝试查找视频文件的绝对路径
        abs_path = Path(video_path).resolve()
        if abs_path.is_file():
            video_path = str(abs_path)
        else:
            # 尝试相对于测试目录查找
            test_dir = Path(__file__).parent
            project_video_path = test_dir / video_path
            if project_video_path.is_file():
                video_path = str(project_video_path.resolve())
            else:
                logger.error(f"找不到视频文件: {video_path}")
                return
    
    logger.info(f"使用视频文件: {video_path}")
    
    try:
        # 运行并行任务
        results = await run_parallel_tasks(
            video_path=video_path,
            num_tasks=args.num_tasks,
            target_language=args.language
        )
        
        # 写入结果
        write_results_to_file(
            results=results,
            video_path=video_path,
            num_tasks=args.num_tasks,
            target_language=args.language,
            output_file=args.output
        )
        
        logger.info("负载测试完成!")
    except Exception as e:
        logger.error(f"执行测试时发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
