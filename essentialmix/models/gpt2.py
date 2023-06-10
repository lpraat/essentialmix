"""
GPT2
https://github.com/openai/gpt-2/blob/master/src/model.py
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
https://github.com/karpathy/nanoGPT/blob/master/model.py
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from einops import einsum, rearrange
from typing_extensions import Self

from essentialmix.core.layers import left_to_right_attention_mask
from essentialmix.core.llm import LanguageModel, LanguageModelOutput
from essentialmix.core.log import Logger

logger = Logger.from_name(__name__)


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, bias: bool = True, p_dropout: float = 0.0, activation: Callable = nn.GELU) -> None:
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, 4 * emb_dim, bias=bias)
        self.act = activation()
        self.fc2 = nn.Linear(4 * emb_dim, emb_dim, bias=bias)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        return self.dropout(self.fc2(x))


class MultiHeadAttention(nn.Module):
    """
    Attention with d_q = d_k = d_v
    """

    def __init__(
        self, n_heads: int, emb_dim: int, bias: bool = True, p_dropout: float = 0.0, use_flash_attention: bool = True
    ) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.d = emb_dim // self.n_heads
        self.emb_dim = emb_dim
        self.use_flash_attention = use_flash_attention
        self.register_buffer("scale", torch.sqrt(torch.tensor([self.d])))
        self.proj_qkv = nn.Linear(emb_dim, 3 * self.n_heads * self.d)
        self.attn_dropout = nn.Dropout(p=p_dropout)
        self.resid_dropout = nn.Dropout(p=p_dropout)
        self.proj_o = nn.Linear(emb_dim, emb_dim, bias=bias)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        q, k, v = torch.split(self.proj_qkv(input_), self.emb_dim, dim=-1)
        rearrange_pattern = "batch time (heads dim) -> batch heads time dim"
        query = rearrange(q, rearrange_pattern, heads=self.n_heads)
        key = rearrange(k, rearrange_pattern, heads=self.n_heads)
        value = rearrange(v, rearrange_pattern, heads=self.n_heads)

        if self.use_flash_attention:
            attn = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            attn = (
                einsum(query, key, "batch heads time1 dim, batch heads time2 dim -> batch heads time1 time2")
                / self.scale
            )

            mask = left_to_right_attention_mask(input_.shape[1])
            assert mask.shape[-1] == mask.shape[-2] == q.shape[1]
            mask = mask.expand(query.shape[0], self.n_heads, mask.shape[-1], mask.shape[-1])
            attn.masked_fill_(mask == 0, float("-inf"))

            attn = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            attn = einsum(attn, value, "batch heads time1 time2, batch heads time2 dim -> batch heads time1 dim")
        return self.resid_dropout(
            self.proj_o(
                rearrange(attn, "batch heads time dim -> batch time (heads dim)", heads=self.n_heads, dim=self.d)
            )
        )


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        # https://github.com/huggingface/transformers/blob/fabe17a726bbf6081cfbcc975d8ac451a81f3e2d/src/transformers/models/gpt2/modeling_gpt2.py#LL470C37-L470C37
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class TransfomerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        attention_heads: int,
        p_dropout: float = 0.0,
        bias: bool = False,
        use_flash_attention: bool = True,
    ) -> None:
        super().__init__()
        self.lnorm1 = LayerNorm(emb_dim, bias=bias)
        self.attention = MultiHeadAttention(
            emb_dim=emb_dim,
            n_heads=attention_heads,
            bias=bias,
            p_dropout=p_dropout,
            use_flash_attention=use_flash_attention,
        )
        self.lnorm2 = LayerNorm(emb_dim, bias=bias)
        self.ff = FeedForward(emb_dim=emb_dim, bias=bias, p_dropout=p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.lnorm1(x))
        return x + self.ff(self.lnorm2(x))


class GPT2(nn.Module):
    def __init__(
        self,
        ctx_len: int,
        vocab_size: int,
        emb_dim: int,
        n_layers: int,
        n_attention_heads: int,
        p_dropout: float = 0.0,
        bias: bool = True,
        use_flash_attention: bool = True,
    ) -> None:
        super().__init__()
        self.ctx_len = ctx_len
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_attention_heads = n_attention_heads
        self.p_dropout = p_dropout
        self.bias = bias

        self.token_embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(ctx_len, emb_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.transformer_blocks = nn.ModuleList(
            [
                TransfomerBlock(
                    emb_dim=emb_dim,
                    attention_heads=n_attention_heads,
                    p_dropout=p_dropout,
                    bias=bias,
                    use_flash_attention=use_flash_attention,
                )
                for _ in range(n_layers)
            ]
        )
        self.l_norm = LayerNorm(emb_dim, bias=bias)
        self.head = nn.Linear(emb_dim, vocab_size)

        # https://paperswithcode.com/method/weight-tying
        self.token_embed.weight = self.head.weight

    def forward(self, x: torch.Tensor, logits_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert len(x.shape) == 2, "Input tensor must have shape (batch, time)"
        assert x.shape[1] <= self.ctx_len, f"Maximum context length is {self.ctx_len}"
        x = self.token_embed(x) + self.pos_embed(torch.arange(x.shape[1])).view(1, x.shape[1], self.emb_dim)
        x = self.dropout(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.l_norm(x)
        if logits_indices is not None:
            # Only compute logits for a sequence subset
            logits = self.head(x[torch.arange(x.shape[0]), logits_indices, :])
        else:
            logits = self.head(x)
        return logits

    @classmethod
    @torch.no_grad()
    def from_pretrained(cls, name: str) -> Self:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
        base_config = {"ctx_len": 1024, "vocab_size": 50257, "bias": True}
        model_config = {
            "distilgpt2": {"emb_dim": 768, "n_layers": 6, "n_attention_heads": 12},
            "gpt2": {"emb_dim": 768, "n_layers": 12, "n_attention_heads": 12},
            "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_attention_heads": 16},
            "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_attention_heads": 20},
            "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_attention_heads": 25},
        }
        weights = transformers.GPT2LMHeadModel.from_pretrained(name).state_dict()
        # Load weights
        logger.info(f"Loading HF gpt2 weights for model {name}")
        model = cls(**base_config, **model_config[name])  # type: ignore
        model.token_embed.weight.copy_(weights["transformer.wte.weight"])
        model.pos_embed.weight.copy_(weights["transformer.wpe.weight"])
        for i_block in range(len(model.transformer_blocks)):
            transfomer_block = model.transformer_blocks[i_block]
            # lnorm1
            transfomer_block.lnorm1.weight.copy_(weights[f"transformer.h.{i_block}.ln_1.weight"])
            if transfomer_block.lnorm1.bias is not None:
                transfomer_block.lnorm1.bias.copy_(weights[f"transformer.h.{i_block}.ln_1.bias"])
            # mha
            transfomer_block.attention.proj_qkv.weight.copy_(weights[f"transformer.h.{i_block}.attn.c_attn.weight"].t())
            transfomer_block.attention.proj_qkv.bias.copy_(weights[f"transformer.h.{i_block}.attn.c_attn.bias"])
            # proj_o
            transfomer_block.attention.proj_o.weight.copy_(weights[f"transformer.h.{i_block}.attn.c_proj.weight"].t())
            transfomer_block.attention.proj_o.bias.copy_(weights[f"transformer.h.{i_block}.attn.c_proj.bias"])
            transfomer_block.lnorm2.weight.copy_(weights[f"transformer.h.{i_block}.ln_2.weight"])
            if transfomer_block.lnorm2.bias is not None:
                transfomer_block.lnorm2.bias.copy_(weights[f"transformer.h.{i_block}.ln_2.bias"])
            # fc1
            transfomer_block.ff.fc1.weight.copy_(weights[f"transformer.h.{i_block}.mlp.c_fc.weight"].t())
            transfomer_block.ff.fc1.bias.copy_(weights[f"transformer.h.{i_block}.mlp.c_fc.bias"])
            # fc2
            transfomer_block.ff.fc2.weight.copy_(weights[f"transformer.h.{i_block}.mlp.c_proj.weight"].t())
            transfomer_block.ff.fc2.bias.copy_(weights[f"transformer.h.{i_block}.mlp.c_proj.bias"])

        model.l_norm.weight.copy_(weights["transformer.ln_f.weight"])
        if model.l_norm.bias is not None:
            model.l_norm.bias.copy_(weights["transformer.ln_f.bias"])
        model.head.weight.copy_(weights["lm_head.weight"])
        assert id(model.head.weight) == id(model.token_embed.weight)  # weight tying
        return model


class GPT2LanguageModel(LanguageModel):
    def __init__(self, model: GPT2) -> None:
        super().__init__()
        self.model: GPT2 = model

    @classmethod
    def from_pretrained(cls, name: str) -> Self:
        return cls(GPT2.from_pretrained(name))

    @property
    def ctx_len(self) -> int:
        return self.model.ctx_len

    def __call__(self, token_indices: torch.Tensor, logits_indices: torch.Tensor) -> LanguageModelOutput:
        return LanguageModelOutput(logits=self.model(token_indices, logits_indices))
