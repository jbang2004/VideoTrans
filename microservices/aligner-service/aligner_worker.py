import asyncio
import json
import aio_pika
from pathlib import Path
from typing import List

from shared.config import load_config
from shared.connection import RabbitMQConnection
from shared.sentence_tools import dict_list_to_sentences, sentences_to_dict_list
from shared.aligner_utils import DurationAligner
from shared.decorators import handle_errors
from shared.logger_utils import ServiceLogger

QUEUE_IN = "duration_align_queue"
QUEUE_OUT = "audio_gen_queue"

class AlignerWorker:
    def __init__(self):
        self.config = load_config()
        self.rabbit_conn = RabbitMQConnection()
        # 使用配置中的采样率来初始化时长对齐器
        self.duration_aligner = DurationAligner(sample_rate=self.config.SAMPLE_RATE, max_speed=1.1)
        self.logger = ServiceLogger("AlignerWorker")

    async def start(self):
        channel = await self.rabbit_conn.connect()
        await channel.declare_queue(QUEUE_IN, durable=True)
        queue_in = await channel.declare_queue(QUEUE_IN, durable=True)
        await queue_in.consume(self.handle_message)
        self.logger.info("Listening on queue: duration_align_queue")

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
            self.logger.info(f"Received {len(sentences)} sentences for duration alignment.")

            try:
                self.logger.info("Starting duration alignment...")
                await self.duration_aligner.align_durations(sentences)
                self.logger.info("Duration alignment completed.")

                out_data = {
                    "task_id": task_id,
                    "sentences": sentences_to_dict_list(sentences),
                    "enhanced_path": enhanced_path,
                    "background_path": background_path
                }
            except Exception as e:
                self.logger.error(f"Processing failed: {e}")
                out_data = {
                    "task_id": task_id,
                    "error": str(e)
                }
                # 写入错误文件
                task_dir = Path(enhanced_path).parent
                (task_dir / "error.txt").write_text(str(e))

            channel = await self.rabbit_conn.connect()
            await channel.declare_queue(QUEUE_OUT, durable=True)
            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(out_data).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=QUEUE_OUT
            )
            self.logger.info(f"Message forwarded to queue: {QUEUE_OUT}")

async def main():
    worker = AlignerWorker()
    await worker.start()
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())