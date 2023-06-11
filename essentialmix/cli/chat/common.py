import json
from dataclasses import dataclass

import cattrs
from typing_extensions import Self


@dataclass
class Message:
    actor: str
    content: str

    # To signal whether this message is the last one from a stream of tokens
    end_stream: bool = True

    @classmethod
    def serialize(cls, msg: Self) -> str:
        return json.dumps(cattrs.unstructure(msg, cls))

    @classmethod
    def deserialize(cls, msg: str) -> Self:
        return cattrs.structure(json.loads(msg), Message)  # type: ignore
