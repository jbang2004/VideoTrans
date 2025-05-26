import sys
import logging

import ray
from ray import serve

from config import Config, init_logging

# Import necessary core actor classes
from core.video_separator import VideoSeparator
from core.asr_model import ASRModel
from core.translation.simplifier import Simplifier
from core.my_index_tts import MyIndexTTSDeployment
from core.timeadjust.duration_aligner import DurationAligner
from core.timeadjust.timestamp_adjuster import TimestampAdjuster
from core.media_mixer import MediaMixer
from core.hls_manager import HLSManager
from orchestrator import MainOrchestrator # Import MainOrchestrator
# Import for API server (remains the same for now)
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
        detached=False,  # Run Serve in the background
        http_options={"host": "0.0.0.0", "port": 8000}
    )
    logger.info("Ray Serve started on port 8000.")

    # 3. Deploy Core Actor Applications
    try:
        # Video Separator
        video_separator_app = VideoSeparator.bind()
        serve.run(video_separator_app, name="VideoSeparatorApp", route_prefix=None)
        logger.info("VideoSeparatorApp deployed successfully.")

        # ASR Model
        asr_app = ASRModel.bind()
        serve.run(asr_app, name="ASRApp", route_prefix=None)
        logger.info("ASRApp deployed successfully.")

        # Simplifier
        simplifier_app = Simplifier.bind()
        serve.run(simplifier_app, name="SimplifierApp", route_prefix=None)
        logger.info("SimplifierApp deployed successfully.")

        # MyIndexTTS
        my_index_tts_app = MyIndexTTSDeployment.bind(config) # MyIndexTTSDeployment takes config
        serve.run(my_index_tts_app, name="TTSApp", route_prefix=None)
        logger.info("TTSApp deployed successfully.")

        # Duration Aligner
        # DurationAligner originally took simplifier_handle and my_index_tts_handle.
        # In the new model, it will fetch these handles internally.
        duration_aligner_app = DurationAligner.bind()
        serve.run(duration_aligner_app, name="DurationAlignerApp", route_prefix=None)
        logger.info("DurationAlignerApp deployed successfully.")

        # Timestamp Adjuster
        timestamp_adjuster_app = TimestampAdjuster.bind()
        serve.run(timestamp_adjuster_app, name="TimestampAdjusterApp", route_prefix=None)
        logger.info("TimestampAdjusterApp deployed successfully.")

        # Media Mixer
        media_mixer_app = MediaMixer.bind()
        serve.run(media_mixer_app, name="MediaMixerApp", route_prefix=None)
        logger.info("MediaMixerApp deployed successfully.")

        # HLS Manager
        hls_manager_app = HLSManager.bind()
        serve.run(hls_manager_app, name="HLSManagerApp", route_prefix=None)
        logger.info("HLSManagerApp deployed successfully.")

        logger.info("All core actor applications deployed.")

    except Exception as e:
        logger.critical(f"Failed to deploy one or more core actor applications: {e}", exc_info=True)
        return

    # 4. Deploy Main Orchestrator Application
    try:
        main_orchestrator_app = MainOrchestrator.bind()
        serve.run(main_orchestrator_app, name="MainOrchestratorApp", route_prefix=None)
        logger.info("MainOrchestratorApp deployed successfully.")
    except Exception as e:
        logger.critical(f"Failed to deploy MainOrchestratorApp: {e}", exc_info=True)
        return # If orchestrator fails, API server likely won't work correctly.

    # 5. Deploy API Server
    logger.info("Attempting to deploy VideoAPI server...")
    try:
        setup_api_server()
        logger.info("VideoAPI server setup completed (likely blocking).")
    except Exception as e:
        logger.critical(f"Failed to setup/deploy VideoAPI server: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main() 