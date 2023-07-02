from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator


class TextCompletionModel(ABC):
    @abstractmethod
    def generate(self, prompts: list[str]) -> list[str]:
        ...

    @abstractmethod
    def stream_generate(self, prompts: list[str]) -> Iterator[list[str]]:
        ...

    @abstractmethod
    async def async_generate(self, prompts: list[str]) -> list[str]:
        ...

    @abstractmethod
    async def async_stream_generate(self, prompts: list[str]) -> AsyncIterator[list[str]]:
        ...
