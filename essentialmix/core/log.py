import logging
from typing import Any

from typing_extensions import ClassVar, Self


class Logger:
    instances: ClassVar[list[Self]] = []

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
    def from_name(cls, name: str) -> Self:
        logger = cls(name)
        cls.instances.append(logger)
        return logger

    @staticmethod
    def set_global_level(level: int) -> None:
        for logger in Logger.instances:
            logger.setLevel(level=level)

    def set_level(self, level: int) -> None:
        self.setLevel(level)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.logger, attr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name},level={logging.getLevelName(self.level)})"
