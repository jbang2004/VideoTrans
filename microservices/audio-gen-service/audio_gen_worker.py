import asyncio
import json
import aio_pika
from pathlib import Path
from typing import List

from shared.config import load_config
from shared.connection import RabbitMQConnection
from shared.sentence_tools import dict_list_to_sentences, sentences_to_dict_list
from shared.audio_gen_utils import AudioGenerator
from shared.decorators import handle_errors
from shared.logger_utils import ServiceLogger

QUEUE_IN = "audio_gen_queue"
QUEUE_OUT = "mixer_queue"

class AudioGenWorker:
    def __init__(self):
        self.config = load_config()
        self.rabbit_conn = RabbitMQConnection()
        # 传入真实 TTS 模型时，请替换 cosyvoice_model 参数；此处示例用 None
        self.audio_generator = AudioGenerator(cosyvoice_model=None, sample_rate=self.config.SAMPLE_RATE)
        self.logger = ServiceLogger("AudioGenWorker")

    async def start(self):
        channel = await self.rabbit_conn.connect()
        await channel.declare_queue(QUEUE_IN, durable=True)
        queue_in = await channel.declare_queue(QUEUE_IN, durable=True)
        await queue_in.consume(self.handle_message)
        self.logger.info("Listening on audio_gen_queue")

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
            self.logger.info(f"Received {len(sentences)} sentences for audio generation.")

            try:
                self.logger.info("Starting audio generation...")
                await self.audio_generator.vocal_audio_maker(sentences)
                failed = [s for s in sentences if s.generated_audio is None]
                if failed:
                    raise RuntimeError(f"{len(failed)} sentences failed audio generation.")
                self.logger.info("Audio generation completed.")
                out_data = {
                    "task_id": task_id,
                    "sentences": sentences_to_dict_list(sentences),
                    "enhanced_path": enhanced_path,
                    "background_path": background_path
                }
            except Exception as e:
                self.logger.error(f"Audio generation failed: {e}")
                out_data = {
                    "task_id": task_id,
                    "error": str(e)
                }
                (Path(enhanced_path).parent / "error.txt").write_text(str(e))

            channel = await self.rabbit_conn.connect()
            await channel.declare_queue(QUEUE_OUT, durable=True)
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(out_data).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=QUEUE_OUT
            )
            self.logger.info("Message forwarded to mixer_queue.")

async def main():
    worker = AudioGenWorker()
    await worker.start()
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
