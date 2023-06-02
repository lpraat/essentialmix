import logging
from typing import Any


class Logger:
    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s][%(module)s.py:%(lineno)s]" "[%(threadName)s][%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    @classmethod
    def from_name(cls, name: str) -> "Logger":
        return cls(name)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.logger, attr)
