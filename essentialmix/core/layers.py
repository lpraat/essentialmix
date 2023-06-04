from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange


def left_to_right_attention_mask(seq_len: int) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len))


class MultiHeadAttention(nn.Module):
    """
    Attention with d_q = d_k = d_v
    """

    def __init__(
        self,
        n_heads: int,
        emb_dim: int,
        bias: bool = False,
        use_flash_attention: bool = True,
    ) -> None:
        super().__init__()
        assert emb_dim % n_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.n_heads = n_heads
        self.d = emb_dim // self.n_heads
        self.emb_dim = emb_dim
        self.use_flash_attention = use_flash_attention
        self.register_buffer("scale", torch.sqrt(torch.tensor([self.d])))
        self.proj_q = nn.Linear(emb_dim, self.n_heads * self.d, bias=bias)
        self.proj_k = nn.Linear(emb_dim, self.n_heads * self.d, bias=bias)
        self.proj_v = nn.Linear(emb_dim, self.n_heads * self.d, bias=bias)
        self.proj_o = nn.Linear(emb_dim, emb_dim, bias=bias)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert q.shape == k.shape == v.shape
        rearrange_pattern = "batch time (heads dim) -> batch heads time dim"
        query = rearrange(self.proj_q(q), rearrange_pattern, heads=self.n_heads)
        key = rearrange(self.proj_k(k), rearrange_pattern, heads=self.n_heads)
        value = rearrange(self.proj_v(v), rearrange_pattern, heads=self.n_heads)

        if self.use_flash_attention:
            attn = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        else:
            attn = (
                einsum(
                    query,
                    key,
                    "batch heads time1 dim, batch heads time2 dim -> batch heads time1 time2",
                )
                / self.scale
            )

            if mask is not None:
                assert mask.shape[-1] == mask.shape[-2] == q.shape[1]
                # (batch, heads, time, time)
                mask = mask.expand(query.shape[0], self.n_heads, mask.shape[-1], mask.shape[-1])
                attn.masked_fill_(mask == 0, float("-inf"))

            attn = attn.softmax(dim=-1)
            attn = einsum(
                attn,
                value,
                "batch heads time1 time2, batch heads time2 dim -> batch heads time1 dim",
            )
        return self.proj_o(
            rearrange(
                attn,
                "batch heads time dim -> batch time (heads dim)",
                heads=self.n_heads,
                dim=self.d,
            )
        )
