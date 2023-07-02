from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Iterator, Optional

import torch

from essentialmix.core.lm.base import TextCompletionModel
from essentialmix.core.log import Logger
from essentialmix.core.tokenize import Tokenizer

logger = Logger.from_name(__name__)


@dataclass
class TorchLanguageModelOutput:
    logits: torch.Tensor


class TorchLanguageModel(ABC):
    """
    Torch-based left-to-right language model
    """

    @property
    @abstractmethod
    def ctx_len(self) -> int:
        ...

    @abstractmethod
    def __call__(self, token_indices: torch.Tensor, logits_indices: Optional[torch.Tensor]) -> TorchLanguageModelOutput:
        ...


class TorchSamplingStrategy(ABC):
    @abstractmethod
    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        ...


class TorchTopK(TorchSamplingStrategy):
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


@dataclass
class StopOptions:
    max_seq_len: Optional[int] = None
    extra_stop_tokens: Optional[list[str]] = None


class TorchTextCompletionModel(TextCompletionModel):
    def __init__(
        self,
        language_model: TorchLanguageModel,
        tokenizer: Tokenizer,
        sampling_strategy: TorchSamplingStrategy,
        stop_options: StopOptions = StopOptions(),
    ):
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.sampling_strategy = sampling_strategy
        self.stop_options = stop_options

    def stream_generate(self, prompts: list[str]) -> Iterator[list[str]]:
        max_length = (
            self.language_model.ctx_len if self.stop_options.max_seq_len is None else self.stop_options.max_seq_len
        )

        if self.stop_options.extra_stop_tokens is None:
            extra_stop_tokens_ids = []
        else:
            extra_stop_tokens_ids = [
                self.tokenizer.encode(extra_stop) for extra_stop in self.stop_options.extra_stop_tokens
            ]

        is_prompt_complete = [False for _ in range(len(prompts))]

        def _update_is_prompt_complete(prompt_tokens: list[list[int]]) -> None:
            for i, tokens in enumerate(prompt_tokens):
                if len(tokens) >= max_length:
                    logger.debug(f"{i}th prompt complete with {len(tokens)}")
                    is_prompt_complete[i] = True

        # Init prompts
        prompt_token_ids = [self.tokenizer.encode(prompt)[:max_length] for prompt in prompts]
        _update_is_prompt_complete(prompt_token_ids)

        while True:
            if sum(is_prompt_complete) == len(prompts):
                return

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
            for i in range(len(prompt_token_ids)):
                if not is_prompt_complete[i]:
                    new_token_id = next_tokens_ids[next_tokens_idx]
                    prompt_token_ids[i].append(int(new_token_id))
                    next_tokens_idx += 1

                    # Complete because of tokenizer eot token
                    if new_token_id == self.tokenizer.eot_token_id:
                        is_prompt_complete[i] = True

                    # Complete because of extra stop tokens
                    for extra_stop_token_id in extra_stop_tokens_ids:
                        if prompt_token_ids[i][-len(extra_stop_token_id) :] == extra_stop_token_id:
                            is_prompt_complete[i] = True
                            break

            _update_is_prompt_complete(prompt_token_ids)
            yield [self.tokenizer.decode(token_ids) for token_ids in prompt_token_ids]

    @torch.no_grad()
    def generate(self, prompts: list[str]) -> list[str]:
        return [complete_prompt for complete_prompt in self.stream_generate(prompts)][-1]

    async def async_stream_generate(self, prompts: list[str]) -> AsyncIterator[list[str]]:
        raise NotImplementedError()

    async def async_generate(self, prompts: list[str]) -> list[str]:
        raise NotImplementedError()


if __name__ == "__main__":
    import logging

    from essentialmix.core.tokenize import HFTokenizer
    from essentialmix.models.gpt2 import GPT2LanguageModel

    logger.set_global_level(logging.DEBUG)

    with torch.no_grad():
        name = "gpt2-medium"
        gpt2 = GPT2LanguageModel.from_pretrained("gpt2-medium")
        gpt2.model.eval()

        generator = TorchTextCompletionModel(
            language_model=GPT2LanguageModel.from_pretrained("gpt2-medium"),
            tokenizer=HFTokenizer("gpt2"),
            sampling_strategy=TorchTopK(temperature=0.6, k=100),
            stop_options=StopOptions(max_seq_len=256, extra_stop_tokens=["\n\n", "\n"]),
        )

        print(generator.generate(prompts=["It is known that"]))

        for i, e in enumerate(generator.stream_generate(prompts=["It is known that"])):
            print(f"{i=}, {e=}")
