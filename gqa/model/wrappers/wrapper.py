import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from model.model import ConformerClassifier, ViTClassifier
from utils import *

parent_dir = Path(__file__).resolve().parent.parent


def create_model_wrapper(
    model_config_path: str,
    train_config_path: str,
    device: str,
) -> ModelWrapper:
    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)

    mode = str(getattr(hps, "mode", "conformer")).lower()

    model = None
    if mode == "conformer":
        model = ConformerClassifier(hps)
    elif mode == "vit":
        model = ViTClassifier(hps)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    train_loader, val_loader = create_dataset(train_hps)

    wrapper = ModelWrapper(
        model=model,
        hps=hps,
        train_hps=train_hps,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    return wrapper


class ModelWrapper:
    def __init__(
        self,
        model,
        hps,
        train_hps,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
    ):
        self._hps = hps
        self._train_hps = train_hps
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device

        self._init_hyperparameters()

        self.model = model

    def _init_hyperparameters(self):
        self.optimizer_hps = self._train_hps.optimizer
        self.scheduler_hps = getattr(self._train_hps, "scheduler", None)
        self.early_stopping_hps = getattr(self._train_hps, "early_stopping", None)

        self.lr = float(self.optimizer_hps.lr)
        self.betas = tuple(
            map(float, getattr(self.optimizer_hps, "betas", (0.9, 0.999)))
        )
        self.weight_decay = float(getattr(self.optimizer_hps, "weight_decay", 0.0))

        if self.scheduler_hps is not None:
            self.warmup_epochs = int(self.scheduler_hps.warmup_epochs)

        if self.early_stopping_hps is not None:
            self.patience = int(self.early_stopping_hps.patience)
            self.min_delta = float(self.early_stopping_hps.min_delta)

        self.num_epochs = int(self._train_hps.num_epochs)
        self.accum_steps = int(getattr(self._train_hps, "accum_steps", 1))

        self.checkpoint_dir = os.path.join(
            parent_dir, str(getattr(self._train_hps, "checkpoint_dir", "checkpoints"))
        )
        self.checkpoint_freq = int(getattr(self._train_hps, "checkpoint_freq", 10))

        self.seed = int(getattr(self._train_hps, "seed", 42))

    def _init_training_scheme(self):
        self.optim = Adam(
            params=self.model.parameters(),
            lr=self.lr,
            betas=self.betas,  # type: ignore
            weight_decay=self.weight_decay,
        )

        self.scheduler = None
        if self.scheduler_hps is not None:
            num_warmup_steps = self.warmup_epochs * np.ceil(
                len(self._train_loader) / self.accum_steps
            )
            num_training_steps = self.num_epochs * np.ceil(
                len(self._train_loader) / self.accum_steps
            )
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        self.early_stopping = None
        if self.early_stopping_hps is not None:
            self.early_stopping = EarlyStopping(
                patience=self.patience, min_delta=self.min_delta
            )

    def _init_weights(self):
        self.model.init_weights()

    def _init_weights_with_ckpt(self, ckpt_path, freeze=False):
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_path)
        self.model.init_weights_with_ckpt(ckpt_path, freeze)

    def move_to_device(self, device):
        self.model = self.model.to(device)

        print(f"Moved to {device}")

    def train(self):
        set_seeds(self.seed)
        self._init_weights()
        self._init_training_scheme()
        self.move_to_device(self._device)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            self.optim.zero_grad(set_to_none=True)

            total_loss = 0.0
            num_batches = 0
            for x, label in tqdm(self._train_loader, leave=False):
                num_batches += 1

                x, label = x.to(self._device), label.to(self._device)
                logits = self.model(x)

                loss = F.cross_entropy(logits, label, reduction="mean")
                loss = loss / self.accum_steps

                loss.backward()

                if num_batches % self.accum_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item() * self.accum_steps

            if num_batches % self.accum_steps != 0:
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, label in tqdm(self._val_loader, leave=False):
                    x, label = x.to(self._device), label.to(self._device)
                    logits = self.model(x)

                    loss = F.cross_entropy(logits, label, reduction="mean")
                    val_loss += loss.item()

            total_loss /= len(self._train_loader)
            val_loss /= len(self._val_loader)

            print(f"----Epoch {epoch}----")
            print(f"Train Loss: {total_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"LR: {self.optim.param_groups[0]['lr']:.6f}")

            if epoch % self.checkpoint_freq == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "scheduler_state_dict": (
                            self.scheduler.state_dict()
                            if self.scheduler is not None
                            else None
                        ),
                        "early_stopping": (
                            self.early_stopping.get_state_dict()
                            if self.early_stopping is not None
                            else None
                        ),
                        "loss": val_loss,
                    },
                    os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}.pt"),
                )

            if self.early_stopping is not None:
                self.early_stopping(self.model, val_loss)

                if self.early_stopping.stop:
                    if self.early_stopping.best_model is not None:
                        self.model = self.early_stopping.best_model

                    break

        torch.save(
            {"model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoint_dir, "best_model.pt"),
        )

        print("Training Complete")

    def evaluate(self, ckpt_path):
        set_seeds(self.seed)
        self._init_weights_with_ckpt(ckpt_path, freeze=True)
        self.move_to_device(self._device)

        self.model.eval()
        with torch.no_grad():
            for i, loader in enumerate((self._train_loader, self._val_loader)):
                num_correct, num_samples = 0, 0

                for x, label in tqdm(loader, leave=False):
                    x = x.to(self._device)
                    logits = self.model(x)

                    pred = logits.argmax(dim=-1).to(label.device)
                    num_correct += (pred == label).sum()
                    num_samples += label.shape[0]

                print(f"----Results ({'Train' if i == 0 else 'Val'})----")
                print(f"Result: {num_correct}/{num_samples}")
                print(f"Accuracy: {(num_correct / num_samples):6f}")

        print("Evaluation Complete")
