import sys
import logging
import ray
from ray import serve
from config import get_config, init_logging

logger = logging.getLogger(__name__)

# 导入所有核心组件
from core.video_separator import VideoSeparator
from core.asr_model import ASRModel
from core.translation.simplifier import Simplifier
from core.translation.translator import Translator
from core.my_index_tts import MyIndexTTSDeployment
from core.timeadjust.duration_aligner import DurationAligner
from core.timeadjust.timestamp_adjuster import TimestampAdjuster
from core.media_mixer import MediaMixer
from core.hls_manager import HLSManager
from orchestrator import MainOrchestrator
from api import setup_server as setup_api_server

def deploy_core_services():
    """部署所有核心服务"""
    config = get_config()
    
    # 服务配置映射
    services = [
        ("VideoSeparatorApp", VideoSeparator.bind()),
        ("ASRApp", ASRModel.bind()),
        ("SimplifierApp", Simplifier.bind()),
        ("TranslatorApp", Translator.bind()),
        ("TTSApp", MyIndexTTSDeployment.bind(config)),
        ("DurationAlignerApp", DurationAligner.bind()),
        ("TimestampAdjusterApp", TimestampAdjuster.bind()),
        ("MediaMixerApp", MediaMixer.bind()),
        ("HLSManagerApp", HLSManager.bind()),
        ("MainOrchestratorApp", MainOrchestrator.bind())
    ]
    
    # 批量部署服务
    for app_name, app_instance in services:
        try:
            serve.run(app_instance, name=app_name, route_prefix=None)
            logger.info(f"{app_name} 部署成功")
        except Exception as e:
            logger.critical(f"{app_name} 部署失败: {e}", exc_info=True)
            raise

def main():
    # 初始化配置和日志
    init_logging()
    config = get_config()
    logger.info("配置和日志初始化完成")

    # 扩展系统路径
    if config.SYSTEM_PATHS:
        sys.path.extend(config.SYSTEM_PATHS)
        logger.info(f"系统路径已扩展: {config.SYSTEM_PATHS}")

    # 初始化Ray
    if not ray.is_initialized():
        ray.init(
            address="auto",
            namespace="videotrans",
            log_to_driver=True,
            ignore_reinit_error=True
        )
    logger.info(f"Ray初始化完成: {ray.get_runtime_context().gcs_address}")

    # 启动Ray Serve
    serve.start(
        detached=False,
        http_options={"host": config.SERVER_HOST, "port": config.SERVER_PORT}
    )
    logger.info(f"Ray Serve启动完成，端口: {config.SERVER_PORT}")

    # 部署核心服务
    try:
        deploy_core_services()
        logger.info("所有核心服务部署完成")
    except Exception as e:
        logger.critical(f"核心服务部署失败: {e}")
        return

    # 启动API服务器
    try:
        logger.info("启动API服务器...")
        setup_api_server()
        logger.info("API服务器启动完成")
    except Exception as e:
        logger.critical(f"API服务器启动失败: {e}")

if __name__ == "__main__":
    main() 