import asyncio
import json
import aio_pika
from pathlib import Path
from typing import List, Optional

from shared.config import load_config
from shared.connection import RabbitMQConnection
from shared.sentence_tools import dict_list_to_sentences, sentences_to_dict_list
from shared.model_in_utils import ModelInProcessor
from shared.decorators import handle_errors
from shared.logger_utils import ServiceLogger

QUEUE_IN = "modelin_queue"
QUEUE_OUT = "tts_queue"

class BatchProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = ServiceLogger("BatchProcessor")

    async def process_batch(self, items: List, process_func: callable, error_handler: Optional[callable] = None):
        if not items:
            return []
        results = []
        batch_size = self.config.MODELIN_BATCH_SIZE
        i = 0
        while i < len(items):
            batch = items[i:i+batch_size]
            try:
                processed = await process_func(batch)
                if processed:
                    results.extend(processed)
                i += len(batch)
            except Exception as e:
                self.logger.error(f"Batch processing failed: {e}", exc_info=True)
                if error_handler:
                    results.extend(error_handler(batch))
                i += len(batch)
        return results

class ModelInWorker:
    def __init__(self):
        self.config = load_config()
        self.rabbit_conn = RabbitMQConnection()
        self.model_in_processor = ModelInProcessor(max_concurrent_tasks=4)
        self.batch_processor = BatchProcessor(self.config)
        self.logger = ServiceLogger("ModelInWorker")

    async def start(self):
        channel = await self.rabbit_conn.connect()
        await channel.declare_queue(QUEUE_IN, durable=True)
        queue_in = await channel.declare_queue(QUEUE_IN, durable=True)
        await queue_in.consume(self.handle_message)
        self.logger.info("Listening on modelin_queue")

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
            self.logger.info(f"Received {len(sentences)} sentences for model input processing.")

            try:
                async def process_batch(batch: List) -> Optional[List]:
                    processed = []
                    async for result in self.model_in_processor.process_sentences(batch, reuse_speaker=True, reuse_uuid=False):
                        processed.extend(result)
                    return processed

                processed_sentences = await self.batch_processor.process_batch(sentences, process_batch)
                if processed_sentences:
                    processed_sentences[0].is_first = True
                    processed_sentences[-1].is_last = True

                out_data = {
                    "task_id": task_id,
                    "sentences": sentences_to_dict_list(processed_sentences),
                    "enhanced_path": enhanced_path,
                    "background_path": background_path
                }
            except Exception as e:
                self.logger.error(f"Model input processing failed: {e}", exc_info=True)
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
            self.logger.info("Message forwarded to tts_queue.")

async def main():
    worker = ModelInWorker()
    await worker.start()
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
