import torch
import torch.nn as nn


class FFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        act: type[nn.Module],
        dropout: float = 0.0,
    ):
        super().__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim

        self._dropout = dropout

        self.layer = nn.Sequential(
            nn.Linear(self._embed_dim, self._hidden_dim),
            act(),
            nn.Dropout(self._dropout),
            nn.Linear(self._hidden_dim, self._embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
