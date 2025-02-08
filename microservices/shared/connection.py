import aio_pika

class RabbitMQConnection:
    def __init__(self, host='rabbitmq', port=5672):
        self.host = host
        self.port = port
        self._connection = None
        self._channel = None

    async def connect(self):
        if not self._connection:
            self._connection = await aio_pika.connect_robust(
                host=self.host,
                port=self.port,
                login="guest",
                password="guest"
            )
        if not self._channel:
            self._channel = await self._connection.channel()
        return self._channel

    async def close(self):
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._channel = None
