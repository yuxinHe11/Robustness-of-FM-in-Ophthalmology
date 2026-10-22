from __future__ import annotations

import os
import sys
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class RetiZeroVisionBackbone(nn.Module):
    """RetiZero vision backbone only.

    This mirrors the vision part in the official model.py:
        VisionModel -> self.model = lora(pretrained=pretrained, R=R)

    It intentionally does not define TextModel, tokenizer, AutoModel, or CLIPRModel.
    """

    def __init__(
        self,
        retizero_repo: Optional[str] = None,
        vision_type: str = "lora",
        pretrained: bool = False,
        r: int = 8,
    ) -> None:
        super().__init__()

        if vision_type not in {"lora", "RETFound"}:
            raise ValueError(f"Unsupported vision_type={vision_type}. Expected 'lora' or 'RETFound'.")

        if retizero_repo is not None:
            retizero_repo = os.path.abspath(retizero_repo)
            if retizero_repo not in sys.path:
                sys.path.insert(0, retizero_repo)

        try:
            from src.RetiZero.clip_modules.modeling.LoraRETFound import lora
        except Exception as e:
            raise ImportError(
                "Cannot import `clip_modules.modeling.LoraRETFound.lora`. "
                "Pass --retizero_repo explicitly and point it to the RetiZero repo root."
            ) from e

        self.model = lora(pretrained=pretrained, R=r)
        self.out_dim = 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> OrderedDict:
    out = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
    return out


def _load_torch_state(path: str, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint at {path} is not a state_dict-like dict.")

    cleaned = OrderedDict()
    for k, v in obj.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    return cleaned


class RetiZeroClassifier(nn.Module):
    """RetiZero vision-only classifier.

    Parameters
    ----------
    num_classes:
        Number of local classes after label remapping.
    weight_path:
        Official RetiZero pretrained .pth. Only vision_model.model.* keys are loaded.
    retizero_repo:
        Path to cloned RetiZero repo root. Used only to import LoRA-RETFound backbone implementation.
    freeze_backbone:
        If True, train only classifier head. For full fine-tuning, use False.
    """

    def __init__(
        self,
        num_classes: int,
        weight_path: Optional[str] = None,
        retizero_repo: Optional[str] = None,
        vision_type: str = "lora",
        r: int = 8,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = RetiZeroVisionBackbone(
            retizero_repo=retizero_repo,
            vision_type=vision_type,
            pretrained=False,
            r=r,
        )
        self.feature_dim = self.backbone.out_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.feature_dim, num_classes),
        )

        if weight_path is not None and weight_path != "":
            self.load_retizero_pretrained(weight_path)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def load_retizero_pretrained(self, weight_path: str) -> Tuple[list, list]:
        state = _load_torch_state(weight_path, map_location="cpu")

        # Official RetiZero CLIPRModel stores the raw image encoder under this prefix.
        backbone_state = _strip_prefix_if_present(state, "vision_model.model.")

        # Fallbacks for already-extracted checkpoints.
        if len(backbone_state) == 0:
            backbone_state = _strip_prefix_if_present(state, "backbone.model.")
        if len(backbone_state) == 0:
            backbone_state = _strip_prefix_if_present(state, "backbone.")
        if len(backbone_state) == 0:
            # Last fallback: try loading whole dict into self.backbone.model.
            backbone_state = state

        incompatible = self.backbone.model.load_state_dict(backbone_state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        print(f"Loaded RetiZero vision backbone from: {weight_path}")
        print(f"Loaded keys into backbone: {len(backbone_state)}")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        if missing[:10]:
            print("First missing keys:")
            for k in missing[:10]:
                print("  ", k)
        if unexpected[:10]:
            print("First unexpected keys:")
            for k in unexpected[:10]:
                print("  ", k)
        return missing, unexpected

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.classifier(feat)
