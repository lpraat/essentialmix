from essentialmix.cli.chat.common import Message
from essentialmix.cli.chat.server import EchoHandler, MessageHandler, StreamLongTextHandler


def test_message_serialization() -> None:
    msg_json = '{"actor": "user", "content": "foo", "end_stream": true}'
    assert Message(actor="user", content="foo", end_stream=True) == Message.deserialize(msg_json)
    assert Message.serialize(Message(actor="user", content="foo", end_stream=True)) == msg_json


def test_handler_interface() -> None:
    assert issubclass(EchoHandler, MessageHandler)
    assert isinstance(StreamLongTextHandler, MessageHandler)
