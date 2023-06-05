import pytest
import torch

from essentialmix.core.layers import MultiHeadAttention, left_to_right_attention_mask


@pytest.mark.parametrize(
    "batch_size,seq_len,emb_dim,n_heads,mask",
    [
        (1, 3, 16, 2, None),
        (1, 3, 16, 2, left_to_right_attention_mask(3)),
        (3, 2, 8, 1, None),
        (3, 2, 8, 1, left_to_right_attention_mask(2)),
        (3, 2, 9, 3, None),
        (3, 2, 9, 3, left_to_right_attention_mask(2)),
    ],
)
def test_multi_head_attention(batch_size: int, seq_len: int, emb_dim: int, n_heads: int, mask: torch.Tensor) -> None:
    input_ = torch.randn((batch_size, seq_len, emb_dim))
    mha = MultiHeadAttention(n_heads=n_heads, emb_dim=emb_dim)
    torch.testing.assert_close(mha.scale**2, torch.tensor([emb_dim / n_heads]))  # type: ignore
    attn = mha(q=input_, k=input_, v=input_, mask=mask)
    assert attn.shape == (batch_size, seq_len, emb_dim)


@pytest.mark.parametrize(
    "batch_size,seq_len,emb_dim,n_heads,mask",
    [
        (3, 2, 9, 2, None),
        (3, 2, 9, 2, left_to_right_attention_mask(2)),
        (3, 2, 9, 3, left_to_right_attention_mask(3)),
    ],
)
def test_multi_head_attention_with_err(
    batch_size: int, seq_len: int, emb_dim: int, n_heads: int, mask: torch.Tensor
) -> None:
    input_ = torch.randn((batch_size, seq_len, emb_dim))
    with pytest.raises(AssertionError):
        mha = MultiHeadAttention(n_heads=n_heads, emb_dim=emb_dim, use_flash_attention=False)
        mha(q=input_, k=input_, v=input_, mask=mask)
