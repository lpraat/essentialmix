import asyncio
import math
from types import TracebackType
from typing import AsyncIterator, ClassVar, Optional, Protocol, Type, runtime_checkable

import click
from typing_extensions import Self
from websockets.server import WebSocketServer, WebSocketServerProtocol, serve

from essentialmix.cli.chat.common import Message
from essentialmix.core.log import Logger
from essentialmix.core.utils import asyncio_run

logger = Logger.from_name(__name__)


@runtime_checkable
class MessageHandler(Protocol):
    def __call__(self, message: Message) -> Self:
        ...

    def __aiter__(self) -> AsyncIterator[Message]:
        ...

    async def __anext__(self) -> Message:
        ...


class EchoHandler:
    def __call__(self, message: Message) -> Self:
        self.stream_messages = [Message(actor="EchoAssistant", content=f"Echoing {message.content}", end_stream=True)]
        return self

    def __aiter__(self) -> AsyncIterator[Message]:
        self.idx = 0
        return self

    async def __anext__(self) -> Message:
        if self.idx > 0:
            raise StopAsyncIteration
        out = self.stream_messages[self.idx]
        self.idx += 1
        return out


class StreamLongTextHandler:
    def __call__(self, message: Message) -> Self:
        self.md_text = """
# This is h1
## This is h2
```python
def f(x):
    return x
```
TODO list:
- a1
- a2
- a3
"""
        chunk_size = 3
        self.stream_messages = [
            Message(
                actor="Assistant {model=Dummy, price=0.000002$/token}",
                content=self.md_text[i * chunk_size : i * chunk_size + chunk_size],
                end_stream=False,
            )
            for i in range(math.ceil(len(self.md_text) / chunk_size))
        ]
        self.stream_messages[-1].end_stream = True
        return self

    def __aiter__(self) -> AsyncIterator[Message]:
        self.iter_stream = iter(self.stream_messages)
        return self

    async def __anext__(self) -> Message:
        try:
            await asyncio.sleep(0.02)
            return next(self.iter_stream)
        except StopIteration:
            raise StopAsyncIteration


class ChatServer:
    LOG_STATUS_EVERY: ClassVar[int] = 5

    def __init__(self, host: str = "0.0.0.0", port: int = 9999) -> None:
        self.host = host
        self.port = port
        self.connections: dict[str, WebSocketServerProtocol] = {}
        self.msg_handler: MessageHandler = StreamLongTextHandler()  # TODO
        self._ws_server: WebSocketServer
        self._log_status_task: asyncio.Task

    @property
    def ws_server(self) -> WebSocketServer:
        if self._ws_server is None:
            raise AttributeError()
        return self._ws_server

    async def __aenter__(self) -> Self:
        self._ws_server = await serve(ws_handler=self.ws_handle, host=self.host, port=self.port)
        self._log_status_task = asyncio.create_task(self.log_status())
        logger.info(f"Chat Server started listening (host={self.host}, port={self.port})")
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        self._log_status_task.cancel()
        self._ws_server.close()

    async def log_status(self) -> None:
        while True:
            if not self.connections:
                logger.info("Status: No clients are connected")
            else:
                logger.info("Status:")
                for conn_id, conn in self.connections.items():
                    logger.info(f"Conn={conn_id}, address={conn.remote_address}, open={not conn.closed}")
                # Remove closed connections
                self.connections = {conn_id: conn for conn_id, conn in self.connections.items() if not conn.closed}
            await asyncio.sleep(ChatServer.LOG_STATUS_EVERY)

    async def ws_handle(self, websocket: WebSocketServerProtocol) -> None:
        logger.info(f"Handling new client connection: {websocket}")
        self.connections[str(id(websocket))] = websocket
        async for msg in websocket:
            str_msg = msg.decode() if isinstance(msg, bytes) else msg  # typeguard
            async for stream_msg in self.msg_handler(Message.deserialize(str_msg)):
                await websocket.send(Message.serialize(stream_msg))


@click.command()
@click.option("--host", help="Host IP", required=True, default="0.0.0.0", type=str)
@click.option("--port", help="Host port", required=True, default=9999, type=int)
def run_server(host: str, port: int) -> None:
    async def main() -> None:
        async with ChatServer(host=host, port=port):
            await asyncio.Future()

    asyncio_run(main())


if __name__ == "__main__":
    import logging

    Logger.set_global_level(logging.INFO)
    run_server()
