import torch
import torch.nn as nn

from model.model.components import FFN, ConvBlock, GroupQueryAttention


class ConformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_head_groups: int,
        num_conv_groups: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self._embed_dim = embed_dim
        self._hidden_dim = hidden_dim
        self._num_heads = num_heads
        self._num_head_groups = num_head_groups
        self._num_conv_groups = num_conv_groups
        self._dropout = dropout

        self.norm_ffn1 = nn.RMSNorm(self._embed_dim)
        self.ffn1 = FFN(
            embed_dim=self._embed_dim,
            hidden_dim=self._hidden_dim,
            act=nn.SiLU,
            dropout=self._dropout,
        )
        self.dropout_ffn1 = nn.Dropout(self._dropout)

        self.norm_attn = nn.RMSNorm(self._embed_dim)
        self.attn = GroupQueryAttention(
            embed_dim=self._embed_dim,
            num_heads=self._num_heads,
            num_head_groups=self._num_head_groups,
            dropout=self._dropout,
        )
        self.dropout_attn = nn.Dropout(self._dropout)

        self.norm_conv = nn.RMSNorm(self._embed_dim)
        self.conv_block = ConvBlock(
            in_channels=self._embed_dim,
            hidden_channels=self._hidden_dim,
            num_groups=self._num_conv_groups,
            dropout=self._dropout,
        )
        self.dropout_conv = nn.Dropout(self._dropout)

        self.norm_ffn2 = nn.RMSNorm(self._embed_dim)
        self.ffn2 = FFN(
            embed_dim=self._embed_dim,
            hidden_dim=self._hidden_dim,
            act=nn.SiLU,
            dropout=self._dropout,
        )
        self.dropout_ffn2 = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        B, N, D = x.shape
        H, W = shape

        _x = self.norm_ffn1(x)
        _x = self.ffn1(_x)
        x = x + 0.5 * self.dropout_ffn1(_x)

        _x = self.norm_attn(x)
        _x, _ = self.attn(_x, _x, _x)
        x = x + self.dropout_attn(_x)

        _x = self.norm_conv(x)
        _x = _x.transpose(1, 2).reshape(B, D, H, W)  # (B, D, H, W)
        _x = self.conv_block(_x)  # (B, D, H, W)
        _x = _x.reshape(B, D, N).transpose(1, 2)  # (B, N, D)
        x = x + self.dropout_conv(_x)

        _x = self.norm_ffn2(x)
        _x = self.ffn2(_x)
        x = x + 0.5 * self.dropout_ffn2(_x)

        return x


class ConformerBlock(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps

        self.layers = nn.ModuleList(
            [
                ConformerLayer(
                    embed_dim=self._hps.embed_dim,
                    hidden_dim=self._hps.hidden_dim,
                    num_heads=self._hps.num_heads,
                    num_head_groups=getattr(self._hps, "num_head_groups", 1),
                    num_conv_groups=self._hps.num_conv_groups,
                    dropout=getattr(self._hps, "dropout", 0.0),
                )
                for _ in range(self._hps.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, shape)

        return x


class Conformer(nn.Module):
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
        self.block = ConformerBlock(self._hps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)  # (B, embed_dim, H//4, W//4)

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(1, 2)  # (B, N, embed_dim)
        x = self.block(x, shape=(H, W))  # (B, N, embed_dim)

        return x
