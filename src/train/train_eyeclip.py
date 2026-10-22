import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedShuffleSplit

import src.eyeclip as eyeclip


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EyeCsvDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["image"]
        y = int(row["class"])
        try:
            img = Image.open(path).convert("RGB")
            img = self.preprocess(img)
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {path}. Error: {e}")
        return img, y, path


def load_and_prepare_csv(csv_path: str, image_root: str, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, int]]:
    df = pd.read_csv(csv_path)
    required = {"image", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Required columns: image,label,dataset,split")

    df = df.copy()
    root = Path(image_root).expanduser()
    csv_paths = df["image"].astype(str).str.strip()
    df["image"] = [
        str(path if path.is_absolute() else root / path)
        for path in map(lambda value: Path(value).expanduser(), csv_paths)
    ]
    df["label"] = df["label"].astype(int)
    df["split"] = df["split"].astype(str).str.lower()

    bad_paths = df.loc[~df["image"].map(os.path.exists), "image"].head(20).tolist()
    if bad_paths:
        raise FileNotFoundError("Some image paths do not exist, first examples:\n" + "\n".join(bad_paths))

    unique_labels = sorted(df["label"].unique().tolist())
    label_map = {old: new for new, old in enumerate(unique_labels)}
    df["class"] = df["label"].map(label_map).astype(int)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    val_df = df[df["split"].isin(["val", "valid", "validation"])].copy()

    if len(train_df) == 0:
        raise ValueError("No training samples found: split must contain 'train'.")
    if len(test_df) == 0:
        raise ValueError("No test samples found: split must contain 'test'.")

    if len(val_df) == 0:
        # Carve validation set from training split, stratified by local class.
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(splitter.split(train_df, train_df["class"]))
        val_df = train_df.iloc[val_idx].copy()
        train_df = train_df.iloc[train_idx].copy()

    return train_df, val_df, test_df, label_map


def build_model(clip_model_type: str, checkpoint_path: str, num_classes: int, device: torch.device):
    model, preprocess = eyeclip.load(clip_model_type, device=device, jit=False,
                                     download_root="models_weights")
    model.float()

    if not hasattr(model, "visual"):
        raise RuntimeError("Loaded EyeCLIP model has no .visual encoder.")

    visual_dim = getattr(model.visual, "output_dim", None)
    if visual_dim is None:
        raise RuntimeError("Cannot infer model.visual.output_dim; please inspect the EyeCLIP visual encoder.")

    model.visual.head = nn.Linear(visual_dim, num_classes).to(device)

    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt

        # Remove possible classifier head from other downstream tasks.
        state = {k: v for k, v in state.items()
                 if not k.endswith("head.weight") and not k.endswith("head.bias")}

        msg = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Missing keys: {len(msg.missing_keys)}")
        print(f"Unexpected keys: {len(msg.unexpected_keys)}")
        if msg.missing_keys[:20]:
            print("First missing keys:", msg.missing_keys[:20])
        if msg.unexpected_keys[:20]:
            print("First unexpected keys:", msg.unexpected_keys[:20])

    return model.to(device), preprocess


def forward_logits(model, images):
    feats = model.visual(images)
    return model.visual.head(feats)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, out_dir: Path, split_name: str, epoch: int, label_map: Dict[int, int]) -> Dict[str, float]:
    model.eval()
    y_true, y_pred, y_prob, paths = [], [], [], []

    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for images, targets, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = forward_logits(model, images)
        loss = loss_fn(logits, targets)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)

        losses.append(float(loss.item()))
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
        paths.extend(list(batch_paths))

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_prob_np = np.array(y_prob)

    metrics = {
        "epoch": epoch,
        "loss": float(np.mean(losses)) if losses else np.nan,
        "acc": accuracy_score(y_true_np, y_pred_np),
        "macro_f1": f1_score(y_true_np, y_pred_np, average="macro"),
    }

    if num_classes == 2:
        metrics["auc_roc"] = roc_auc_score(y_true_np, y_prob_np[:, 1])
        metrics["auc_pr"] = average_precision_score(y_true_np, y_prob_np[:, 1])
        cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics["sensitivity"] = tp / (tp + fn + 1e-12)
        metrics["specificity"] = tn / (tn + fp + 1e-12)
    else:
        metrics["auc_roc"] = roc_auc_score(y_true_np, y_prob_np, multi_class="ovr", average="macro")
        metrics["auc_pr"] = average_precision_score(np.eye(num_classes)[y_true_np], y_prob_np, average="macro")

    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "image": paths,
        "true_class": y_true,
        "pred_class": y_pred,
        "prob_pred": y_prob_np.max(axis=1),
    })
    for c in range(num_classes):
        pred_df[f"prob_class_{c}"] = y_prob_np[:, c]
    pred_df.to_csv(out_dir / f"{split_name}_pred_epoch{epoch}.csv", index=False)

    with open(out_dir / f"{split_name}_classification_report_epoch{epoch}.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_true_np, y_pred_np, digits=4))
        f.write("\n\noriginal_label_to_local_class = ")
        f.write(json.dumps(label_map, ensure_ascii=False))

    return metrics


def train_one_epoch(model, loader, optimizer, scaler, device, epoch: int, accum_iter: int,
                    max_norm: float, training_mode: str):
    model.train()
    if training_mode == "linear_probe":
        # Keep the frozen encoder deterministic while training only the new head.
        model.visual.eval()
        model.visual.head.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = forward_logits(model, images)
            loss = loss_fn(logits, targets) / accum_iter

        scaler.scale(loss).backward()

        if (step + 1) % accum_iter == 0:
            if max_norm and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), max_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * accum_iter

    return {"train_loss": total_loss / max(1, len(loader))}


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_metric: float, label_map: Dict[int, int], args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "label_map": label_map,
        "args": vars(args),
    }, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", required=True)
    p.add_argument(
        "--image_root",
        required=True,
        help="Root directory prepended to relative image paths in --csv_path",
    )
    p.add_argument("--task_name", default="Glaucoma", help="e.g. AMD or Glaucoma")
    p.add_argument("--finetune", required=True, help="Downloaded EyeCLIP .pt checkpoint")
    p.add_argument("--clip_model_type", default="ViT-B/32")
    p.add_argument("--output_dir", required=True)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--training_mode", default="linear_probe", choices=["finetune", "linear_probe"],
        help="finetune updates the visual encoder; linear_probe freezes it and trains only visual.head",
    )
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--accum_iter", type=int, default=1)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--freeze_text", action="store_true", default=True)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--resume", default="", help="Path to best.pt or last.pt for evaluation/resume")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir) / args.task_name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df, label_map = load_and_prepare_csv(
        args.csv_path, args.image_root, args.val_frac, args.seed
    )
    num_classes = len(label_map)
    print(f"Task: {args.task_name}")
    print(f"Original label -> local class: {label_map}")
    print(f"Train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_df.to_csv(out_dir / "train_used.csv", index=False)
    val_df.to_csv(out_dir / "val_used.csv", index=False)
    test_df.to_csv(out_dir / "test_used.csv", index=False)
    with open(out_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    model, preprocess = build_model(args.clip_model_type, args.finetune, num_classes, device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"Loaded resume/eval checkpoint: {args.resume}")

    if args.training_mode == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.visual.head.parameters():
            p.requires_grad = True
    elif args.freeze_text:
        for name, p in model.named_parameters():
            if not name.startswith("visual."):
                p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Training mode: {args.training_mode} | trainable params: {trainable:,}/{total:,}")

    train_loader = DataLoader(
        EyeCsvDataset(train_df, preprocess),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        EyeCsvDataset(val_df, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        EyeCsvDataset(test_df, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    if args.eval_only:
        metrics = evaluate(model, test_loader, device, num_classes, out_dir / "test", "test", 0, label_map)
        pd.DataFrame([metrics]).to_csv(out_dir / "metrics_eval_only.csv", index=False)
        print(metrics)
        return

    best_auc = -1.0
    all_logs = []

    for epoch in range(args.epochs):
        train_log = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            args.accum_iter, args.clip_grad, args.training_mode,
        )
        val_metrics = evaluate(model, val_loader, device, num_classes, out_dir / "val", "val", epoch, label_map)
        row = {"epoch": epoch, **train_log, **{f"val_{k}": v for k, v in val_metrics.items() if k != "epoch"}}
        all_logs.append(row)
        pd.DataFrame(all_logs).to_csv(out_dir / "log.csv", index=False)

        print(row)

        current_auc = val_metrics["auc_roc"]
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, current_auc, label_map, args)

        if current_auc > best_auc:
            best_auc = current_auc
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_auc, label_map, args)
            print(f"Saved best.pt at epoch {epoch}, val_auc={best_auc:.4f}")

    # Final test using best checkpoint
    best_ckpt = torch.load(out_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    test_metrics = evaluate(model, test_loader, device, num_classes, out_dir / "test", "test", int(best_ckpt["epoch"]), label_map)
    pd.DataFrame([test_metrics]).to_csv(out_dir / "metrics_test.csv", index=False)
    print("Final test:", test_metrics)


if __name__ == "__main__":
    main()
