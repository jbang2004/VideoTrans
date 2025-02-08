import asyncio
import json
import aio_pika
from pathlib import Path

from shared.config import load_config
from shared.connection import RabbitMQConnection
from shared.sentence_tools import sentences_to_dict_list
from shared.asr_utils import SenseAutoModel
from shared.audio_separator import ClearVoiceSeparator
from shared.ffmpeg_utils import FFmpegTool
from shared.decorators import handle_errors
from shared.logger_utils import ServiceLogger

QUEUE_IN = "asr_queue"
QUEUE_OUT = "translation_queue"

class ASRWorker:
    def __init__(self):
        self.config = load_config()
        self.rabbit_conn = RabbitMQConnection()
        
        # 使用配置文件中的参数初始化ASR模型
        self.asr_model = SenseAutoModel(
            config=self.config,
            **self.config.ASR_MODEL
        )
        
        # 使用配置文件中的参数初始化音频分离器
        self.separator = ClearVoiceSeparator(**self.config.AUDIO_SEPARATOR)
        self.ffmpeg_tool = FFmpegTool()
        self.logger = ServiceLogger("ASRWorker")

    async def start(self):
        channel = await self.rabbit_conn.connect()
        await channel.declare_queue(QUEUE_IN, durable=True)
        queue_in = await channel.declare_queue(QUEUE_IN, durable=True)
        await queue_in.consume(self.handle_message)
        self.logger.info("Listening on asr_queue")

    @handle_errors()
    async def handle_message(self, message: aio_pika.IncomingMessage):
        async with message.process():
            data = json.loads(message.body.decode())
            task_id = data.get("task_id", "unknown")
            self.logger.set_task_id(task_id)

            audio_path = data.get("audio_path", "")
            target_language = data.get("target_language", "auto")
            self.logger.info(f"Processing task for audio: {audio_path}")

            try:
                task_dir = Path(audio_path).parent
                enhanced_path = str(task_dir / "enhanced_audio.wav")
                background_path = str(task_dir / "background_audio.wav")

                self.logger.info("Starting audio separation...")
                enhanced, background, sr = self.separator.separate_audio(audio_path)
                import soundfile as sf
                sf.write(enhanced_path, enhanced, sr)
                sf.write(background_path, background, sr)
                self.logger.info("Audio separation completed.")

                self.logger.info("Starting ASR...")
                sentences = await self.asr_model.generate_async(
                    input=enhanced_path,
                    language=target_language
                )
                self.logger.info(f"ASR produced {len(sentences)} sentences.")

                out_data = {
                    "task_id": task_id,
                    "target_language": target_language,
                    "sentences": sentences_to_dict_list(sentences),
                    "enhanced_path": enhanced_path,
                    "background_path": background_path
                }
            except Exception as e:
                self.logger.error(f"ASR processing failed: {e}")
                out_data = {
                    "task_id": task_id,
                    "error": str(e)
                }
                (Path(audio_path).parent / "error.txt").write_text(str(e))

            channel = await self.rabbit_conn.connect()
            await channel.declare_queue(QUEUE_OUT, durable=True)
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(out_data).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=QUEUE_OUT
            )
            self.logger.info("Message forwarded to translation_queue.")

async def main():
    worker = ASRWorker()
    await worker.start()
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())