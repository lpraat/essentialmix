import abc
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import tiktoken
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from essentialmix.core.log import Logger

logger = Logger.from_name(__name__)


class LMTokenizer(abc.ABC):
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


class TiktokenTokenizer(LMTokenizer):
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


class HFTokenizer(LMTokenizer):
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


@dataclass
class LanguageModelOutput:
    logits: torch.Tensor


# TODO improve
class LanguageModel(abc.ABC):
    @property
    @abstractmethod
    def ctx_len(self) -> int:
        ...

    @abstractmethod
    def __call__(self, token_indices: torch.Tensor, logits_indices: torch.Tensor) -> LanguageModelOutput:
        ...


class SamplingStrategy(abc.ABC):
    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        ...


class TopK(SamplingStrategy):
    def __init__(self, temperature: float, k: int):
        self.temperature = temperature
        self.k = k

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        scores = (logits / self.temperature).softmax(dim=1)
        top_k_scores, top_k_indices = torch.topk(scores, dim=1, sorted=False, k=self.k)
        next_tokens = top_k_indices[
            torch.arange(top_k_indices.shape[0]), torch.multinomial(top_k_scores, num_samples=1).squeeze()
        ]
        return next_tokens


class TextGeneration:
    def __init__(self, language_model: LanguageModel, tokenizer: LMTokenizer, sampling_strategy: SamplingStrategy):
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.sampling_strategy = sampling_strategy

    @torch.no_grad()
    def complete_prompts(self, prompts: list[str], max_seq_len: Optional[int] = None) -> list[str]:
        max_length = self.language_model.ctx_len if max_seq_len is None else max_seq_len

        is_prompt_complete = [False for _ in range(len(prompts))]

        def _update_is_prompt_complete(prompt_tokens: list[list[int]]) -> None:
            for i, tokens in enumerate(prompt_tokens):
                if len(tokens) >= max_length:
                    logger.debug(f"{i}th prompt complete with {len(tokens)}")
                    is_prompt_complete[i] = True

        # Init prompts
        # TODO encode batch, bench
        prompt_token_ids = [self.tokenizer.encode(prompt)[:max_length] for prompt in prompts]
        _update_is_prompt_complete(prompt_token_ids)

        while True:
            if sum(is_prompt_complete) == len(prompts):
                return [self.tokenizer.decode(token_ids) for token_ids in prompt_token_ids]

            # Only forward incomplete prompts
            incomplete_prompts = [
                token_ids for i, token_ids in enumerate(prompt_token_ids) if not is_prompt_complete[i]
            ]
            # Last logit index for each prompt
            last_indices = [len(token_ids) - 1 for token_ids in incomplete_prompts]
            # Padding
            max_len = max(len(token_ids) for token_ids in incomplete_prompts)
            padded_token_ids = [
                # Pad using the last token_id
                token_ids + [token_ids[-1] for _ in range(max_len - len(token_ids))]
                for token_ids in incomplete_prompts
            ]

            # Sample next tokens
            lm_out = self.language_model(
                torch.tensor(padded_token_ids, dtype=torch.int64), logits_indices=torch.tensor(last_indices)
            )
            next_tokens_ids = self.sampling_strategy.sample(lm_out.logits)

            # Update prompts
            next_tokens_idx = 0
            for i, token_ids in enumerate(prompt_token_ids):
                if not is_prompt_complete[i]:
                    new_token_id = next_tokens_ids[next_tokens_idx]
                    prompt_token_ids[i].append(int(new_token_id))
                    next_tokens_idx += 1

                    if new_token_id == self.tokenizer.eot_token_id:
                        is_prompt_complete[i] = True

            _update_is_prompt_complete(prompt_token_ids)
