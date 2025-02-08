import asyncio
import json
import aio_pika
from pathlib import Path
from typing import List

from shared.config import load_config
from shared.connection import RabbitMQConnection
from shared.sentence_tools import dict_list_to_sentences, sentences_to_dict_list
from shared.audio_gen_utils import TTSTokenGenerator
from shared.decorators import handle_errors
from shared.logger_utils import ServiceLogger

QUEUE_IN = "tts_queue"
QUEUE_OUT = "duration_align_queue"

class TTSWorker:
    def __init__(self):
        self.config = load_config()
        self.rabbit_conn = RabbitMQConnection()
        # 此处请替换为实际 TTS 模型，如 cosyvoice_model 对象；示例中传 None
        self.tts_token_generator = TTSTokenGenerator(cosyvoice_model=None, Hz=25)
        self.logger = ServiceLogger("TTSWorker")

    async def start(self):
        channel = await self.rabbit_conn.connect()
        await channel.declare_queue(QUEUE_IN, durable=True)
        queue_in = await channel.declare_queue(QUEUE_IN, durable=True)
        await queue_in.consume(self.handle_message)
        self.logger.info("Listening on tts_queue")

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
            self.logger.info(f"Received {len(sentences)} sentences for TTS token generation.")

            try:
                self.logger.info("Starting TTS token generation...")
                await self.tts_token_generator.tts_token_maker(sentences, reuse_uuid=False)
                self.logger.info("TTS token generation completed.")
                out_data = {
                    "task_id": task_id,
                    "sentences": sentences_to_dict_list(sentences),
                    "enhanced_path": enhanced_path,
                    "background_path": background_path
                }
            except Exception as e:
                self.logger.error(f"TTS processing failed: {e}", exc_info=True)
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
            self.logger.info("Message forwarded to duration_align_queue.")

async def main():
    worker = TTSWorker()
    await worker.start()
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
