from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch.nn as nn


class MIRAGEClassifier(nn.Module):
    """Official MIRAGE B-scan encoder with a classification head."""

    def __init__(
        self,
        mirage_repo: str,
        weights: str,
        model_size: str,
        num_classes: int,
        mode: str = "finetune",
        input_size: int = 512,
        pool: str = "global",
    ):
        super().__init__()

        repo = Path(mirage_repo).expanduser().resolve()
        if not (repo / "mirage_wrapper.py").is_file():
            raise FileNotFoundError(
                f"Official MIRAGE repository not found at {repo}. "
                "Expected mirage_wrapper.py in that directory."
            )

        weights_path = Path(weights).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"MIRAGE weights not found: {weights_path}")

        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        wrapper = importlib.import_module("mirage_wrapper")
        try:
            model_cls = wrapper.miragecls_factory[pool]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported MIRAGE pooling: {pool}. "
                "Choose global, cls, or token_mix."
            ) from exc

        self.backbone = model_cls(
            input_size=input_size,
            patch_size=32,
            num_classes=num_classes,
            modalities="bscan",
            weights=str(weights_path),
            device="cpu",
        )

        checkpoint_model = str(
            getattr(self.backbone.args, "model", "")
        ).lower()
        if model_size.lower() not in checkpoint_model:
            raise ValueError(
                f"Requested MIRAGE-{model_size}, but checkpoint reports "
                f"model={checkpoint_model!r}."
            )

        if mode == "linear_probe":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.head.parameters():
                parameter.requires_grad = True
        elif mode == "finetune":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, images):
        return self.backbone(images)
