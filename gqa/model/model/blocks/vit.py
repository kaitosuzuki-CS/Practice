import torch
import torch.nn as nn

from model.model.components import FFN, GroupQueryAttention


class ViTLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_head_groups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._num_head_groups = num_head_groups
        self._dropout = dropout

        self.norm1 = nn.RMSNorm(self._embed_dim)
        self.attn = GroupQueryAttention(
            embed_dim=self._embed_dim,
            num_heads=self._num_heads,
            num_head_groups=self._num_head_groups,
            dropout=self._dropout,
        )
        self.dropout1 = nn.Dropout(self._dropout)

        self.norm2 = nn.RMSNorm(self._embed_dim)
        self.ffn = FFN(
            embed_dim=self._embed_dim,
            hidden_dim=self._hidden_dim,
            act=nn.GELU,
            dropout=self._dropout,
        )
        self.dropout2 = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.norm1(x)
        _x, _ = self.attn(_x, _x, _x)
        x = x + self.dropout1(_x)

        _x = self.norm2(x)
        _x = self.ffn(_x)
        x = x + self.dropout2(_x)

        return x


class ViTBlock(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps

        self.layers = nn.ModuleList(
            [
                ViTLayer(
                    embed_dim=self._hps.embed_dim,
                    hidden_dim=self._hps.hidden_dim,
                    num_heads=self._hps.num_heads,
                    num_head_groups=getattr(self._hps, "num_head_groups", 1),
                    dropout=getattr(self._hps, "dropout", 0.0),
                )
                for _ in range(self._hps.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)

        return x


class ViT(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps

        self.in_conv = nn.Sequential(
            nn.Conv2d(
                self._hps.im_channels,
                self._hps.embed_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                self._hps.embed_dim,
                self._hps.embed_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        self.block = ViTBlock(self._hps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)  # (B, embed_dim, H//4, W//4)

        B, D, H, W = x.shape
        x = x.view(B, D, H * W).transpose(1, 2)  # (B, N, embed_dim)

        x = self.block(x)  # (B, N, embed_dim)

        return x
