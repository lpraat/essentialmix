import asyncio
from types import TracebackType
from typing import ClassVar, Optional, Type

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from typing_extensions import Self
from websockets.client import WebSocketClientProtocol, connect

from essentialmix.cli.chat.common import Message
from essentialmix.core.log import Logger
from essentialmix.core.utils import asyncio_run

logger = Logger.from_name(__name__)


# TODO generic interface to render terminal


class ChatClient:
    EXIT_STR: ClassVar[str] = "exit()"

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.chat_history: list[Message] = []
        self.console = Console()

    async def __aenter__(self) -> Self:
        self.websocket: WebSocketClientProtocol = await connect(self.uri)
        logger.info(f"Connected to remote server with address={self.websocket.remote_address}")
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]
    ) -> None:
        logger.info("Closing connection with remote server")
        await self.websocket.close()

    def update_terminal(self) -> None:
        self.console.rule("[bold red]Ignore everything above - terminal was cleared[/]")
        self.console.clear()
        for msg in self.chat_history:
            if msg.actor == "user":
                self.console.print(
                    Panel(f"{msg.content}", title=f"[green]{msg.actor.capitalize()}[/]", title_align="left")
                )
            else:
                self.console.print(
                    Panel(
                        Markdown(msg.content, code_theme="one-dark"),
                        title=f"[cyan]{msg.actor.capitalize()}",
                        title_align="left",
                    )
                )

    async def send_message(self, msg: Message) -> None:
        self.chat_history.append(msg)
        await self.websocket.send(Message.serialize(msg))

    async def recv_message(self) -> Message:
        msg_recv = await self.websocket.recv()
        str_msg = msg_recv.decode() if isinstance(msg_recv, bytes) else msg_recv  # typeguard
        return Message.deserialize(str_msg)

    async def run(self) -> None:
        while True:
            input_ = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: Prompt.ask(
                    f"[magenta]Send a message \[type {ChatClient.EXIT_STR} to close the chat] >\n", console=self.console
                ),
            )
            if input_ == ChatClient.EXIT_STR:
                return
            user_message = Message(actor="user", content=input_)
            await self.send_message(user_message)
            spinner = Status("Waiting for a response...", console=self.console)
            spinner.start()

            is_first_message = True
            while True:
                server_msg = await self.recv_message()
                spinner.stop()
                if is_first_message:
                    self.chat_history.append(server_msg)
                    is_first_message = False
                else:
                    self.chat_history[-1].content += server_msg.content
                self.update_terminal()
                if server_msg.end_stream:
                    break


@click.command()
@click.option(
    "--uri", help="Server URI (e.g., ws://localhost:9999)", required=True, default="ws://localhost:9999", type=str
)
def run_client(uri: str) -> None:
    async def main() -> None:
        async with ChatClient(uri=uri) as client:
            await client.run()

    asyncio_run(main())


if __name__ == "__main__":
    import logging

    Logger.set_global_level(logging.INFO)
    run_client()
