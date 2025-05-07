import sys
import time
import logging

import ray
from ray import serve

from config import Config, init_logging

# Import necessary components from engine and api files
from preprocessing_engine import PreprocessingPipe as PreprocessingPipeDef, video_separator_handle, asr_handle
from translation_engine import TranslationPipe as TranslationPipeDef, translator_handle, my_index_tts_handle, duration_aligner_handle, timestamp_adjuster_handle, media_mixer_handle, hls_manager_handle
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
        preprocessing_pipe_app = PreprocessingPipeDef.bind(
            video_separator_handle=video_separator_handle,
            asr_model_handle=asr_handle
        )
        serve.run(preprocessing_pipe_app, name="PreprocessingEngine", route_prefix=None)
        logger.info("PreprocessingEngine deployed successfully.")
    except Exception as e:
        logger.critical(f"Failed to deploy PreprocessingEngine: {e}", exc_info=True)
        return

    # 4. Deploy Translation Engine
    try:
        translation_pipe_app = TranslationPipeDef.bind(
            translator_handle=translator_handle,
            my_index_tts_handle=my_index_tts_handle,
            duration_aligner_handle=duration_aligner_handle,
            timestamp_adjuster_handle=timestamp_adjuster_handle,
            media_mixer_handle=media_mixer_handle,
            hls_manager_handle=hls_manager_handle
        )
        serve.run(translation_pipe_app, name="TranslationEngine", route_prefix=None)
        logger.info("TranslationEngine deployed successfully.")
    except Exception as e:
        logger.critical(f"Failed to deploy TranslationEngine: {e}", exc_info=True)
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