import asyncio
import json
import aio_pika
from pathlib import Path

from shared.config import load_config
from shared.connection import RabbitMQConnection
from shared.sentence_tools import dict_list_to_sentences
from shared.ffmpeg_utils import FFmpegTool
from shared.mixer_utils import MediaMixer
from shared.decorators import handle_errors
from shared.logger_utils import ServiceLogger

QUEUE_IN = "mixer_queue"

class MixerWorker:
    def __init__(self):
        self.config = load_config()
        self.rabbit_conn = RabbitMQConnection()
        self.mixer = MediaMixer(self.config)
        self.ffmpeg_tool = FFmpegTool()
        self.logger = ServiceLogger("MixerWorker")

    async def start(self):
        channel = await self.rabbit_conn.connect()
        await channel.declare_queue(QUEUE_IN, durable=True)
        queue_in = await channel.declare_queue(QUEUE_IN, durable=True)
        await queue_in.consume(self.handle_message)
        self.logger.info("Listening on mixer_queue")

    @handle_errors()
    async def handle_message(self, message: aio_pika.IncomingMessage):
        async with message.process():
            data = json.loads(message.body.decode())
            task_id = data.get("task_id", "unknown")
            self.logger.set_task_id(task_id)
            sentence_dicts = data.get("sentences", [])
            enhanced_path = data.get("enhanced_path", "")
            background_path = data.get("background_path", "")

            sentences = dict_list_to_sentences(sentence_dicts)
            self.logger.info(f"Received {len(sentences)} sentences for mixing.")

            try:
                task_dir = Path(enhanced_path).parent
                output_dir = task_dir / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"final_{task_id}.mp4"

                self.logger.info("Starting media mixing...")
                await self.mixer.mix_sentences(
                    sentences=sentences,
                    output_path=str(output_path),
                    background_path=background_path,
                    vocals_volume=self.config.VOCALS_VOLUME,
                    background_volume=self.config.BACKGROUND_VOLUME
                )
                self.logger.info(f"Media mixing completed. Final video: {output_path}")
            except Exception as e:
                self.logger.error(f"Mixing failed: {e}")
                (Path(enhanced_path).parent / "error.txt").write_text(str(e))

async def main():
    worker = MixerWorker()
    await worker.start()
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
