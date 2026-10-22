import argparse
import csv
import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from timm.models.layers import trunc_normal_

import src.model.models_vit as models_vit
from functools import partial
import timm.models.vision_transformer


class CSVVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """RETFound/MAE-style ViT classifier compatible with newer timm forward()."""
    def __init__(self, global_pool=False, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            if hasattr(self, "norm"):
                del self.norm

    def forward_features(self, x, *args, **kwargs):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        return x

    def forward(self, x):
        return self.head(self.forward_features(x))


def make_vit(arch: str, num_classes: int, drop_path_rate: float, global_pool: bool):
    if arch == "vit_base_patch16":
        return CSVVisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes, drop_path_rate=drop_path_rate, global_pool=global_pool,
        )
    if arch == "vit_large_patch16":
        return CSVVisionTransformer(
            img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes, drop_path_rate=drop_path_rate, global_pool=global_pool,
        )
    raise ValueError(f"Unsupported --model {arch}; use vit_base_patch16, vit_large_patch16, or auto")


def torch_load_compat(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_vit_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> str:
    w = sd.get("patch_embed.proj.weight", None)
    if w is None:
        for k, v in sd.items():
            if k.endswith("patch_embed.proj.weight"):
                w = v
                break
    if w is None:
        print("Could not infer architecture from checkpoint; fallback to vit_large_patch16")
        return "vit_large_patch16"
    dim = int(w.shape[0])
    if dim == 768:
        return "vit_base_patch16"
    if dim == 1024:
        return "vit_large_patch16"
    raise ValueError(f"Unknown ViT embed dim in checkpoint patch_embed.proj.weight: {dim}")


ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class EyeCsvBinaryDataset(Dataset):
    def __init__(self, csv_path: str, image_root: str, task: str, split: str,
                 input_size: int = 224, is_train: bool = False):
        self.csv_path = csv_path
        self.image_root = Path(image_root).expanduser()
        self.task = task.lower()
        self.split = split.lower()
        self.input_size = input_size
        self.is_train = is_train

        df = pd.read_csv(csv_path)
        required = {"image", "label", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}. Required columns: {required}")

        df["split"] = df["split"].astype(str).str.lower()
        df = df[df["split"] == self.split].copy()
        df["label"] = df["label"].astype(int)

        if self.task == "amd":
            # original labels: non-AMD=0, AMD=1
            df = df[df["label"].isin([0, 1])].copy()
            df["target"] = df["label"].map({0: 0, 1: 1}).astype(int)
            self.positive_name = "AMD"
            self.negative_name = "non-AMD"
        elif self.task == "glaucoma":
            # original labels: non-Glaucoma=2, Glaucoma=3
            df = df[df["label"].isin([2, 3])].copy()
            df["target"] = df["label"].map({2: 0, 3: 1}).astype(int)
            self.positive_name = "Glaucoma"
            self.negative_name = "non-Glaucoma"
        else:
            raise ValueError("--task must be one of: amd, glaucoma")

        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No rows left for task={task}, split={split}. Check labels/split names.")

        csv_paths = df["image"].astype(str).str.strip().tolist()
        self.paths = [
            str(path if path.is_absolute() else self.image_root / path)
            for path in map(lambda value: Path(value).expanduser(), csv_paths)
        ]
        self.targets = df["target"].astype(int).tolist()
        self.original_labels = df["label"].astype(int).tolist()
        self.datasets = df["dataset"].tolist() if "dataset" in df.columns else [None] * len(df)

        # RETFound/MAE fine-tuning convention: ImageNet normalization after RGB resize/crop.
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(input_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {path}. Error: {e}")
        img = self.transform(img)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return img, target, path


def interpolate_pos_embed_if_needed(model: nn.Module, checkpoint_model: Dict[str, torch.Tensor]) -> None:
    """Minimal MAE/ViT pos_embed interpolation. For 224x224 ViT-L/16 it usually does nothing."""
    if "pos_embed" not in checkpoint_model:
        return
    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size == new_size:
        return

    print(f"Interpolating pos_embed: {orig_size}x{orig_size} -> {new_size}x{new_size}")
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)


def get_state_dict_from_checkpoint(ckpt: Dict) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "module", "teacher"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = {}
    for k, v in state_dict.items():
        for prefix in ["module.", "backbone."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        new_state[k] = v
    return new_state


def build_model(args) -> nn.Module:
    checkpoint_model = None
    checkpoint = None

    if args.pretrained:
        print(f"Loading RETFound checkpoint metadata: {args.pretrained}")
        checkpoint = torch_load_compat(args.pretrained)
        checkpoint_model = strip_prefix_if_present(get_state_dict_from_checkpoint(checkpoint))

    arch = args.model
    if arch == "auto":
        if checkpoint_model is None:
            raise ValueError("--model auto requires --pretrained")
        arch = infer_vit_arch_from_state_dict(checkpoint_model)
    print(f"Building model: {arch}")

    model = make_vit(
        arch=arch,
        num_classes=2,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if checkpoint_model is not None:
        print(f"Loading RETFound base checkpoint: {args.pretrained}")
        state_dict = model.state_dict()

        # remove classifier and any same-name tensors whose shape differs.
        # strict=False does NOT ignore shape mismatch, so we must delete them explicitly.
        remove_keys = []
        for k, v in checkpoint_model.items():
            if k in state_dict and tuple(v.shape) != tuple(state_dict[k].shape):
                remove_keys.append(k)
        for k in remove_keys:
            print(f"Removing mismatched key: {k}, ckpt={tuple(checkpoint_model[k].shape)}, model={tuple(state_dict[k].shape)}")
            del checkpoint_model[k]

        # MAE checkpoints may contain decoder keys; classification ViT reports them as unexpected, OK.
        interpolate_pos_embed_if_needed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("Load message:", msg)

        trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.constant_(model.head.bias, 0)

    return model


def compute_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> Dict[str, float]:
    y_pred = (prob_pos >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "n": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, prob_pos))
        out["auprc"] = float(average_precision_score(y_true, prob_pos))
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses: List[float] = []
    all_targets: List[int] = []
    all_probs: List[float] = []
    rows: List[Dict] = []

    for images, targets, paths in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
            probs = F.softmax(logits.float(), dim=1)

        losses.append(loss.item() * images.size(0))
        prob_pos = probs[:, 1].detach().cpu().numpy()
        prob_neg = probs[:, 0].detach().cpu().numpy()
        pred = probs.argmax(dim=1).detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()

        all_probs.extend(prob_pos.tolist())
        all_targets.extend(target_np.tolist())
        for p, y, yh, p0, p1 in zip(paths, target_np, pred, prob_neg, prob_pos):
            rows.append({"image": p, "target": int(y), "pred": int(yh), "prob_negative": float(p0), "prob_positive": float(p1)})

    y_true = np.array(all_targets, dtype=int)
    prob_pos = np.array(all_probs, dtype=float)
    metrics = compute_metrics(y_true, prob_pos)
    mean_loss = float(np.sum(losses) / max(1, len(loader.dataset)))
    metrics["loss"] = mean_loss
    return mean_loss, metrics, pd.DataFrame(rows)


def train_one_epoch(model, loader, optimizer, device, scaler, epoch: int, training_mode: str,
                    amp: bool = True, max_norm: float = 0.0) -> float:
    model.train()
    if training_mode == "linear_probe":
        # Freeze stochastic/stateful backbone behavior as well as its gradients.
        model.eval()
        model.head.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0

    pbar = tqdm(loader, desc=f"train epoch {epoch}")
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if max_norm and max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), max_norm
            )
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        pbar.set_postfix(loss=total_loss / max(1, total_n), lr=optimizer.param_groups[0]["lr"])

    return total_loss / max(1, total_n)


def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int, best_metric: float, args, val_metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "args": vars(args),
        "val_metrics": val_metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV with image,label,dataset,split")
    parser.add_argument(
        "--image_root",
        required=True,
        help="Root directory prepended to relative image paths in --csv",
    )
    parser.add_argument("--task", default="glaucoma", choices=["amd", "glaucoma"], help="Binary task")
    parser.add_argument("--pretrained", required=True, help="RETFound base .pth checkpoint")
    parser.add_argument("--output_dir", required=True, help="Directory to save best.pt, logs, predictions")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--model", default="vit_large_patch16", choices=["auto", "vit_base_patch16", "vit_large_patch16"], help="Use auto to infer Base/Large from checkpoint")
    parser.add_argument("--lr", type=float, default=5e-5, help="Conservative full fine-tuning LR")
    parser.add_argument("--training_mode", default="linear_probe", choices=["finetune", "linear_probe"],)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--drop_path", type=float, default=0.2)
    parser.add_argument("--global_pool", action="store_true", default=True)
    parser.add_argument("--cls_token", action="store_false", dest="global_pool")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--metric", default="auroc", choices=["auroc", "auprc", "bacc", "macro_f1", "weighted_f1", "acc"])
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", default="", help="Fine-tuned checkpoint to evaluate or resume from")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = EyeCsvBinaryDataset(args.csv, args.image_root, args.task, "train", args.input_size, is_train=True)
    val_ds = EyeCsvBinaryDataset(args.csv, args.image_root, args.task, "val", args.input_size, is_train=False)
    test_ds = EyeCsvBinaryDataset(args.csv, args.image_root, args.task, "test", args.input_size, is_train=False)

    print(f"Task: {args.task}")
    print(f"Train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"Train positives/negatives: {sum(train_ds.targets)} / {len(train_ds) - sum(train_ds.targets)}")
    print(f"Val positives/negatives:   {sum(val_ds.targets)} / {len(val_ds) - sum(val_ds.targets)}")
    print(f"Test positives/negatives:  {sum(test_ds.targets)} / {len(test_ds) - sum(test_ds.targets)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_model(args).to(device)
    if args.training_mode == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Training mode: {args.training_mode} | trainable params: {trainable:,}/{total:,}")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if args.resume:
        print(f"Loading fine-tuned checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        state = get_state_dict_from_checkpoint(ckpt)
        msg = model.load_state_dict(strip_prefix_if_present(state), strict=False)
        print("Resume load message:", msg)
        if not args.eval_only and isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

    if args.eval_only:
        _, metrics, pred_df = evaluate(model, test_loader, device, amp=args.amp)
        print("Test metrics:", json.dumps(metrics, indent=2))
        pd.DataFrame([metrics]).to_csv(out_dir / f"{args.task}_test_metrics.csv", index=False)
        pred_df.to_csv(out_dir / f"{args.task}_test_predictions.csv", index=False)
        return

    best_metric = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: List[Dict] = []

    start = time.time()
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler, epoch,
            args.training_mode, amp=args.amp,
        )
        _, val_metrics, _ = evaluate(model, val_loader, device, amp=args.amp)
        current = val_metrics[args.metric]

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / f"{args.task}_train_log.csv", index=False)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_{args.metric}={current:.4f}, val_metrics={val_metrics}")

        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            bad_epochs = 0
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_metric, args, val_metrics)
            print(f"Saved best.pt at epoch {epoch}, best val_{args.metric}={best_metric:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best val_{args.metric}={best_metric:.4f}")
                break

    # evaluate best checkpoint on test
    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model"], strict=True)
    _, test_metrics, test_pred = evaluate(model, test_loader, device, amp=args.amp)
    print("Best epoch:", best_epoch)
    print("Test metrics:", json.dumps(test_metrics, indent=2))

    pd.DataFrame([test_metrics]).to_csv(out_dir / f"{args.task}_test_metrics.csv", index=False)
    test_pred.to_csv(out_dir / f"{args.task}_test_predictions.csv", index=False)

    elapsed = time.time() - start
    print(f"Done. Training time: {elapsed/60:.1f} min. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
