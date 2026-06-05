import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_groups: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._hidden_channels = hidden_channels
        self._num_groups = num_groups
        self._dropout = dropout

        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.depthwise_act = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            groups=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pointwise2_norm = nn.GroupNorm(
            num_groups=(
                num_groups if hidden_channels % num_groups == 0 else hidden_channels
            ),
            num_channels=hidden_channels,
        )
        self.pointwise2_act = nn.SiLU()
        self.pointwise2_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = self.pointwise_conv(x)  # (B, 2*hidden_channels, H, W)

        x = self.depthwise_act(x)  # (B, hidden_channels, H, W)
        x = self.depthwise_conv(x)  # (B, hidden_channels, H, W)

        x = self.pointwise2_norm(x)  # (B, hidden_channels, H, W)
        x = self.pointwise2_act(x)  # (B, hidden_channels, H, W)
        x = self.pointwise2_conv(x)  # (B, in_channels, H, W)

        return x
