import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupQueryAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, num_head_groups: int, dropout: float = 0.0
    ):
        """
        Using nn.Parameters instead of nn.Linear to practice using einsum.
        nn.Linear may be more efficient since einsum can't leverage a single large GEMM as easily as a fused linear layer.
        """
        super().__init__()

        assert embed_dim % num_heads == 0
        assert num_heads % num_head_groups == 0

        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._num_head_groups = num_head_groups
        self._dropout = dropout

        self._head_dim = self._embed_dim // self._num_heads
        self._num_heads_per_group = self._num_heads // self._num_head_groups
        self._scale = self._head_dim**0.5

        self.q = nn.Parameter(
            torch.empty(self._num_heads, self._embed_dim, self._head_dim)
        )  # (num_heads, embed_dim, head_dim)
        self.k = nn.Parameter(
            torch.empty(self._num_head_groups, self._embed_dim, self._head_dim)
        )  # (num_head_groups, embed_dim, head_dim)
        self.v = nn.Parameter(
            torch.empty(self._num_head_groups, self._embed_dim, self._head_dim)
        )  # (num_head_groups, embed_dim, head_dim)

        self.o = nn.Parameter(
            torch.empty(self._embed_dim, self._embed_dim)
        )  # (embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q)
        nn.init.xavier_uniform_(self.k)
        nn.init.xavier_uniform_(self.v)
        nn.init.xavier_uniform_(self.o)

    def _check_input(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        B_q, N_q, D_q = q.shape
        B_k, N_k, D_k = k.shape
        B_v, N_v, D_v = v.shape

        assert B_q == B_k == B_v
        assert N_k == N_v
        assert D_q == D_k == D_v == self._embed_dim

        if mask is not None:
            if mask.ndim == 3:  # (B, N_q, N_k)
                mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N_q, N_k)
            elif (
                mask.ndim == 4
            ):  # (B, num_head_groups, N_q, N_k) or (B, num_heads_per_group, N_q, N_k)
                if mask.shape[1] == 1 or mask.shape[1] == self._num_head_groups:
                    mask = mask.unsqueeze(
                        2
                    )  # (B, num_head_groups, 1, N_q, N_k) or (B, 1, 1, N_q, N_k)
                elif mask.shape[1] == self._num_heads_per_group:
                    mask = mask.unsqueeze(1)  # (B, 1, num_heads_per_group, N_q, N_k)
                else:
                    raise ValueError(f"Invalid mask shape: {mask.shape}")
            elif mask.ndim != 5:
                raise ValueError(f"Invalid mask shape: {mask.shape}")

            assert mask.shape == (
                B_q,
                self._num_head_groups,
                self._num_heads_per_group,
                N_q,
                N_k,
            ) or mask.shape == (B_q, 1, 1, N_q, N_k)

        return q, k, v, mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, mask = self._check_input(q, k, v, mask)

        B, N_q, D_q = q.shape
        _, N_k, D_k = k.shape
        _, N_v, D_v = v.shape

        q = torch.einsum("bnd,hdk->bhnk", q, self.q)  # (B, num_heads, N_q, head_dim)
        k = torch.einsum(
            "bmd,gdk->bgmk", k, self.k
        )  # (B, num_head_groups, N_k, head_dim)
        v = torch.einsum(
            "bmd,gdk->bgmk", v, self.v
        )  # (B, num_head_groups, N_v, head_dim)

        q = q.reshape(
            B, self._num_head_groups, self._num_heads_per_group, N_q, self._head_dim
        )  # (B, num_head_groups, num_heads_per_group, N_q, head_dim)

        logits = (
            torch.einsum("bgsnk,bgmk->bgsnm", q, k) / self._scale
        )  # (B, num_head_groups, num_heads_per_group, N_q, N_k)

        if mask is not None:
            if mask.dtype == torch.bool:
                logits = logits.masked_fill(
                    ~mask, torch.finfo(logits.dtype).min
                )  # (B, _, _, N_q, N_k)
            else:
                logits = logits + mask  # (B, _, _, N_q, N_k)

        weights = torch.softmax(logits.float(), dim=-1).to(
            logits.dtype
        )  # (B, num_head_groups, num_heads_per_group, N_q, N_k)
        weights = F.dropout(weights, p=self._dropout, training=self.training)

        o = torch.einsum(
            "bgsnm,bgmk->bgsnk", weights, v
        )  # (B, num_head_groups, num_heads_per_group, N_q, head_dim)
        o = o.permute(0, 3, 1, 2, 4).reshape(
            B, N_q, self._embed_dim
        )  # (B, N_q, embed_dim)

        out = torch.einsum("bnd,df->bnf", o, self.o)  # (B, N_q, embed_dim)

        return out, weights


class MultiHeadAttention(GroupQueryAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_head_groups=num_heads,
            dropout=dropout,
        )


class MultiQueryAttention(GroupQueryAttention):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_head_groups=1,
            dropout=dropout,
        )
