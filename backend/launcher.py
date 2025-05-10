import sys
import time
import logging

import ray
from ray import serve

from config import Config, init_logging

# Import necessary components from engine and api files
from pre_engine import PreEngine as PreEngineDef, video_separator_handle, asr_handle
from trans_engine import TransPipe as TransPipeDef, translator_handle, my_index_tts_handle, duration_aligner_handle, timestamp_adjuster_handle, media_mixer_handle, hls_manager_handle
from api import setup_server as setup_api_server

logger = logging.getLogger(__name__)

def main():
    # 1. Initialize logging and configuration
    init_logging()
    config = Config()
    config.init_directories() # Initialize directories once
    logger.info("Logging and configuration initialized.")

    # Add custom system paths if any are defined in config
    if hasattr(config, 'SYSTEM_PATHS') and config.SYSTEM_PATHS:
        sys.path.extend(config.SYSTEM_PATHS)
        logger.info(f"Extended system paths with: {config.SYSTEM_PATHS}")

    # 2. Initialize Ray and Ray Serve
    if not ray.is_initialized():
        ray.init(
            address="auto", 
            namespace="videotrans", 
            log_to_driver=True, 
            ignore_reinit_error=True
        )
    logger.info(f"Ray initialized: {ray.get_runtime_context().gcs_address if ray.is_initialized() else 'Failed'}")

    serve.start(
        detached=True,  # Run Serve in the background
        http_options={"host": "0.0.0.0", "port": 8000}
    )
    logger.info("Ray Serve started on port 8000.")

    # 3. Deploy Preprocessing Engine
    try:
        pre_pipe = PreEngineDef.bind(
            video_separator_handle=video_separator_handle,
            asr_model_handle=asr_handle
        )
        serve.run(pre_pipe, name="PreEngine", route_prefix=None)
        logger.info("PreEngine deployed successfully.")
    except Exception as e:
        logger.critical(f"Failed to deploy PreEngine: {e}", exc_info=True)
        return

    # 4. Deploy Translation Engine
    try:
        trans_pipe = TransPipeDef.bind(
            translator_handle=translator_handle,
            my_index_tts_handle=my_index_tts_handle,
            duration_aligner_handle=duration_aligner_handle,
            timestamp_adjuster_handle=timestamp_adjuster_handle,
            media_mixer_handle=media_mixer_handle,
            hls_manager_handle=hls_manager_handle
        )
        serve.run(trans_pipe, name="TransEngine", route_prefix=None)
        logger.info("TransEngine deployed successfully.")
    except Exception as e:
        logger.critical(f"Failed to deploy TransEngine: {e}", exc_info=True)
        return

    # 5. Deploy API Server
    # The api.py's setup_server will handle its own serve.run for the FastAPI app
    # It will connect to the already running Ray and Serve instance.
    logger.info("Attempting to deploy VideoAPI server...")
    try:
        # setup_api_server() is blocking, so this will be the last step for the foreground
        setup_api_server() 
        logger.info("VideoAPI server setup completed (likely blocking).")
    except Exception as e:
        logger.critical(f"Failed to setup/deploy VideoAPI server: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main() 