import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class CsvImageDataset(Dataset):
    """Dataset for CSVs with columns: image,label,dataset,split.

    The `image` column can be absolute, or relative to `image_root`.
    Labels can be remapped by passing `label_map`, e.g. {0: 0, 1: 1}.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str = "",
        label_map: Optional[Dict[int, int]] = None,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root) if image_root else None
        self.df = pd.read_csv(self.csv_path)
        required = {"image", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

        self.label_map = label_map or self._auto_label_map(self.df["label"].tolist())
        self.transform = transform or default_retizero_transform()

    @staticmethod
    def _auto_label_map(labels: List[int]) -> Dict[int, int]:
        unique = sorted(pd.Series(labels).dropna().astype(int).unique().tolist())
        return {old: new for new, old in enumerate(unique)}

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, p: str) -> Path:
        path = Path(str(p))
        if path.is_absolute():
            return path
        if self.image_root is None:
            return path
        return self.image_root / path

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {img_path}") from e
        image = self.transform(image)
        raw_label = int(row["label"])
        label = self.label_map[raw_label]
        return image, torch.tensor(label, dtype=torch.long), str(img_path)


def default_retizero_transform(image_size: int = 224):
    # RetiZero examples use ViT 224 input. ImageNet normalization is safer than raw ToTensor.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def import_retizero_cliprmodel(retizero_repo: str, module_name: str = "iden_modules"):
    """Import CLIPRModel from the RetiZero repo without importing its utils package.

    For fine-tuning, the provided official script imports `from iden_modules import CLIPRModel`.
    For retrieval/zero-shot they use other wrappers.  This keeps the only repo-level import explicit.
    """
    repo = str(Path(retizero_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        module = __import__(module_name, fromlist=["CLIPRModel"])
        return getattr(module, "CLIPRModel")
    except Exception as e:
        raise ImportError(
            f"Could not import CLIPRModel from module '{module_name}' under {repo}. "
            "Check --retizero_repo and whether the repo contains iden_modules.py/package."
        ) from e


class RetiZeroClassifier(nn.Module):
    """RetiZero visual encoder + linear classification head."""

    def __init__(
        self,
        retizero_repo: str,
        weight_path: str,
        num_classes: int,
        vision_type: str = "lora",
        module_name: str = "iden_modules",
        freeze_encoder: bool = False,
        r: int = 8,
    ):
        super().__init__()
        CLIPRModel = import_retizero_cliprmodel(retizero_repo, module_name)

        # Official fine-tuning script uses from_checkpoint=True and then takes vision_model.model.
        pretrained = CLIPRModel(
            vision_type=vision_type,
            from_checkpoint=True,
            weights_path=weight_path,
            R=r,
        )
        self.encoder = pretrained.vision_model.model

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.feature_dim = self._infer_feature_dim()
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _infer_feature_dim(self) -> int:
        # Avoid hard-coding 1024/2048 where possible. Some RetiZero wrappers return [B,D],
        # others may return [B,N,D]; we handle both in forward_features.
        device = next(self.encoder.parameters()).device
        was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            feat = self.encoder(dummy)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if feat.ndim == 3:
                feat = feat.mean(dim=1)
            dim = int(feat.shape[-1])
        self.encoder.train(was_training)
        return dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if feat.ndim == 3:
            feat = feat.mean(dim=1)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def build_label_map(*csv_paths: str) -> Dict[int, int]:
    labels = []
    for p in csv_paths:
        df = pd.read_csv(p)
        labels.extend(df["label"].astype(int).tolist())
    unique = sorted(set(labels))
    return {old: new for new, old in enumerate(unique)}


def save_checkpoint(path: str, model: nn.Module, label_map: Dict[int, int], epoch: int, metrics: dict, args_dict: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "label_map": label_map,
        "metrics": metrics,
        "args": args_dict,
    }, path)


def load_checkpoint(path: str, model: nn.Module, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    return ckpt


def write_metrics_csv(path: str, row: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
