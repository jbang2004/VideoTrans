import asyncio
import logging
from config import Config
from services.hls import serve

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    config = Config()
    server = serve(config)
    
    try:
        logger.info(f"启动HLS gRPC服务 - {config.HLS_GRPC_HOST}:{config.HLS_GRPC_PORT}")
        await server.start()
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("正在关闭服务器...")
        await server.stop(0)
        
if __name__ == "__main__":
    asyncio.run(main()) 