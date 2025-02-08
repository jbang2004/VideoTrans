import asyncio
import json
import aio_pika
from typing import List, Dict, Optional
from pathlib import Path

from shared.config import load_config
from shared.connection import RabbitMQConnection
from shared.sentence_tools import dict_list_to_sentences, sentences_to_dict_list
from shared.translation_utils import DeepSeekClient, GeminiClient, Translator
from shared.decorators import handle_errors
from shared.logger_utils import ServiceLogger

QUEUE_IN = "translation_queue"
QUEUE_OUT = "modelin_queue"

class BatchProcessor:
    def __init__(self, config):
        self.config = config
        self.logger = ServiceLogger("TranslationBatchProcessor")

    async def process_batch(self, items: List, process_func: callable, error_handler: Optional[callable] = None):
        if not items:
            return []
        results = []
        batch_size = self.config.TRANSLATION_BATCH_SIZE
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

class TranslationWorker:
    def __init__(self):
        self.config = load_config()
        self.rabbit_conn = RabbitMQConnection()
        self.batch_processor = BatchProcessor(self.config)
        self.logger = ServiceLogger("TranslationWorker")
        if self.config.TRANSLATION_MODEL.lower() == "deepseek":
            client = DeepSeekClient(api_key=self.config.DEEPSEEK_API_KEY)
        else:
            client = GeminiClient(api_key=self.config.GEMINI_API_KEY)
        self.translator = Translator(client)

    async def start(self):
        channel = await self.rabbit_conn.connect()
        await channel.declare_queue(QUEUE_IN, durable=True)
        queue_in = await channel.declare_queue(QUEUE_IN, durable=True)
        await queue_in.consume(self.handle_message)
        self.logger.info("Listening on translation_queue")

    @handle_errors()
    async def handle_message(self, message: aio_pika.IncomingMessage):
        async with message.process():
            data = json.loads(message.body.decode())
            task_id = data.get("task_id", "unknown")
            self.logger.set_task_id(task_id)
            target_language = data.get("target_language", "zh")
            sentence_dicts = data.get("sentences", [])
            enhanced_path = data.get("enhanced_path", "")
            background_path = data.get("background_path", "")

            sentences = dict_list_to_sentences(sentence_dicts)
            self.logger.info(f"Received {len(sentences)} sentences for translation.")

            try:
                async def process_batch(batch: List) -> Optional[List]:
                    texts = {str(j): s.raw_text for j, s in enumerate(batch)}
                    self.logger.debug(f"Translating batch of {len(texts)} texts.")
                    translated = await self.translator.translate(texts, target_language=target_language)
                    if "output" not in translated:
                        self.logger.error("Translation result missing 'output' field.")
                        return None
                    translated_texts = translated["output"]
                    if len(translated_texts) == len(texts):
                        for j, s in enumerate(batch):
                            s.trans_text = translated_texts[str(j)]
                        return batch
                    return None

                def handle_error(batch: List) -> List:
                    self.logger.warning(f"Translation batch failed. Reverting to original text for {len(batch)} sentences.")
                    for s in batch:
                        s.trans_text = s.raw_text
                    return batch

                translated_sentences = await self.batch_processor.process_batch(
                    sentences, process_batch, error_handler=handle_error
                )

                # 简化：若某些句子太长则调用简化接口
                need_simplify = any(len(s.trans_text) > self.config.MAX_TOKENS_PER_SENTENCE for s in translated_sentences)
                if need_simplify:
                    self.logger.info("Simplification triggered for long sentences.")
                    texts = {str(i): s.trans_text for i, s in enumerate(translated_sentences)}
                    simplified = await self.translator.simplify(texts)
                    if "moderate" in simplified:
                        for i, s in enumerate(translated_sentences):
                            if len(s.trans_text) > self.config.MAX_TOKENS_PER_SENTENCE:
                                s.trans_text = simplified["moderate"].get(str(i), s.trans_text)

                out_data = {
                    "task_id": task_id,
                    "sentences": sentences_to_dict_list(translated_sentences),
                    "enhanced_path": enhanced_path,
                    "background_path": background_path
                }
            except Exception as e:
                self.logger.error(f"Translation processing failed: {e}", exc_info=True)
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
            self.logger.info("Message forwarded to modelin_queue.")

async def main():
    worker = TranslationWorker()
    await worker.start()
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
