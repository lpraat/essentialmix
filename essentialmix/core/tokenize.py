from abc import ABC, abstractmethod

import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        ...

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        ...

    @abstractmethod
    def decode_batch(self, token_ids: list[list[int]]) -> list[str]:
        ...

    @property
    @abstractmethod
    def eot_token_id(self) -> int:
        ...


class TiktokenTokenizer(Tokenizer):
    def __init__(self, name: str, n_threads: int = 4):
        self.tokenizer = tiktoken.get_encoding(name)
        self.n_threads = n_threads

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer.encode_batch(texts, num_threads=self.n_threads)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def decode_batch(self, token_ids: list[list[int]]) -> list[str]:
        return self.tokenizer.decode_batch(token_ids, num_threads=self.n_threads)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eot_token


class HFTokenizer(Tokenizer):
    def __init__(self, name: str):
        # NOTE: we use the base tokenizer to retrieve the eos token
        # (for some reason, it can't be found in fast tokenizer)
        self.base_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(name)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        raise NotImplementedError("encode_batch method not implemented in HFTokenizer")

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def decode_batch(self, token_ids: list[list[int]]) -> list[str]:
        raise NotImplementedError("decode_batch method not implemented in HFTokenizer")

    @property
    def eot_token_id(self) -> int:
        token_id = self.base_tokenizer.eos_token_id
        if token_id is None:
            raise RuntimeError(f"eot_token_id not found in {self.base_tokenizer}")
        return token_id
