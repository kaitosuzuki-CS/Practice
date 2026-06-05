import torch
import torch.nn as nn

from model.blocks import Conformer


class ConformerClassifier(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps

        self.conformer = Conformer(self._hps)
        self.classifier = nn.Sequential(
            nn.SiLU(), nn.Linear(self._hps.embed_dim, self._hps.num_classes)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.RMSNorm):
                nn.init.ones_(m.weight)

            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def init_weights_with_ckpt(self, ckpt_path, freeze=False):
        ckpt = torch.load(ckpt_path)
        missing_keys, unexpected_keys = self.load_state_dict(ckpt["model_state_dict"])

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )
        print(f"Missing Keys: {missing_keys}")
        print(f"Unexpected Keys: {unexpected_keys}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conformer(x)  # (B, N, embed_dim)
        x = x.mean(dim=1)  # (B, embed_dim)

        logits = self.classifier(x)  # (B, num_classes)
        return logits
