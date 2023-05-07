import torch
import math


def positional_encoding(timesteps: torch.Tensor, embed_dim: int) -> torch.Tensor:
    assert embed_dim >= 1, "Embedding dimension must be atleast 1"
    assert embed_dim % 2 == 0, "Embedding dimension is not even"
    # (1, dim//2)
    indices = torch.arange(0, embed_dim // 2, dtype=torch.float32)
    # (n_ts, 1) * (1, dim//2) ---broadcast--> (n_ts, dim//2)
    freq = timesteps.view(-1, 1) / torch.exp(2 * indices / embed_dim * math.log(10000))
    emb = torch.zeros((timesteps.shape[0], embed_dim), dtype=torch.float32)
    emb[:, ::2] = torch.sin(freq)  # even -> sine
    emb[:, 1::2] = torch.cos(freq)  # odd -> cosine
    return emb


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt

    timesteps = torch.arange(100)
    encodings = positional_encoding(timesteps, 512)
    sns.heatmap(encodings)
    plt.show()
