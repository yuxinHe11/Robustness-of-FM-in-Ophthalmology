import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils_retizero import (
    CsvImageDataset,
    RetiZeroClassifier,
    build_label_map,
    default_retizero_transform,
    save_checkpoint,
    seed_everything,
    write_metrics_csv,
)


def split_csv_if_needed(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise ValueError("A single --csv must contain column: split")
    out = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "valid", "test"]:
        part = df[df["split"].astype(str).str.lower() == split]
        if len(part):
            name = "val" if split == "valid" else split
            p = Path(out_dir) / f"{name}.csv"
            part.to_csv(p, index=False)
            out[name] = str(p)
    if "train" not in out or "val" not in out or "test" not in out:
        raise ValueError(f"Need train/val/test splits, got {sorted(out)}")
    return out["train"], out["val"], out["test"]


def binary_auc(y_true, prob):
    classes = sorted(set(y_true))
    if len(classes) != 2:
        return float("nan")
    # after label_map, positive class is local class 1
    return roc_auc_score(y_true, prob[:, 1])


def multiclass_auc(y_true, prob):
    if len(set(y_true)) < 2:
        return float("nan")
    try:
        if prob.shape[1] == 2:
            return binary_auc(y_true, prob)
        return roc_auc_score(y_true, prob, multi_class="ovr", average="macro")
    except Exception:
        return float("nan")


@torch.no_grad()
def evaluate(model, loader, device, num_classes, desc="eval"):
    model.eval()
    all_y, all_pred, all_prob = [], [], []
    for images, labels, _paths in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        all_y.extend(labels.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_prob.append(prob.cpu().numpy())
    y = np.asarray(all_y)
    pred = np.asarray(all_pred)
    prob = np.concatenate(all_prob, axis=0) if all_prob else np.empty((0, num_classes))
    return {
        "acc": float(accuracy_score(y, pred)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "auc": float(multiclass_auc(y, prob)),
    }


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, training_mode, grad_clip=0.0):
    model.train()
    if training_mode == "linear_probe":
        # model.train() would otherwise re-enable dropout/BN updates in the frozen encoder.
        model.encoder.eval()
        model.classifier.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"train epoch {epoch}")
    for images, labels, _paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), grad_clip
            )
        optimizer.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{total_loss / max(n,1):.5f}")
    return total_loss / max(n, 1)


def main(args):
    seed_everything(args.seed)
    if args.freeze_encoder:
        # Backward-compatible alias for older launch commands.
        args.training_mode = "linear_probe"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        train_csv, val_csv, test_csv = split_csv_if_needed(args.csv, out_dir / "split_csv")
    else:
        train_csv, val_csv, test_csv = args.train_csv, args.val_csv, args.test_csv

    label_map = build_label_map(train_csv, val_csv, test_csv)
    num_classes = len(label_map)
    print(f"Original label -> local class: {label_map}")
    print(f"Num classes: {num_classes}")

    transform = default_retizero_transform(args.image_size)
    train_ds = CsvImageDataset(train_csv, args.image_root, label_map, transform)
    val_ds = CsvImageDataset(val_csv, args.image_root, label_map, transform)
    test_ds = CsvImageDataset(test_csv, args.image_root, label_map, transform)
    print(f"Train/val/test: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=args.drop_last,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = RetiZeroClassifier(
        retizero_repo=args.retizero_repo,
        weight_path=args.weight_path,
        num_classes=num_classes,
        vision_type=args.vision_type,
        module_name=args.module_name,
        freeze_encoder=(args.training_mode == "linear_probe"),
        r=args.r,
    ).to(device)
    print(f"Feature dim: {model.feature_dim}")

    if args.training_mode == "linear_probe":
        params = model.classifier.parameters()
    else:
        params = model.parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Training mode: {args.training_mode} | trainable params: {trainable:,}/{total:,}")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = -1.0
    best_epoch = -1
    no_improve = 0
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    metrics_path = out_dir / "metrics.csv"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            args.training_mode, args.grad_clip,
        )
        val_m = evaluate(model, val_loader, device, num_classes, desc=f"val epoch {epoch}")
        test_m = evaluate(model, test_loader, device, num_classes, desc=f"test epoch {epoch}")

        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}, **{f"test_{k}": v for k, v in test_m.items()}}
        write_metrics_csv(str(metrics_path), row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

        score = val_m[args.monitor]
        save_checkpoint(str(last_path), model, label_map, epoch, {"val": val_m, "test": test_m}, vars(args))
        if score > best_val:
            best_val = score
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(str(best_path), model, label_map, epoch, {"val": val_m, "test": test_m}, vars(args))
            print(f"Saved best: {best_path} | best_val_{args.monitor}={best_val:.6f}")
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stop at epoch {epoch}; best epoch={best_epoch}, best val {args.monitor}={best_val:.6f}")
            break

    print(f"Done. Best checkpoint: {best_path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--retizero_repo", required=True, help="Path to cloned RetiZero repo")
    p.add_argument("--weight_path", required=True, help="Path to RetiZero.pth")
    p.add_argument("--csv", required=True, help="One CSV with split column")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--image_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vision_type", default="lora")
    p.add_argument("--module_name", default="iden_modules")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument(
        "--training_mode", default="linear_probe", choices=["finetune", "linear_probe"],
        help="finetune updates the encoder and classifier; linear_probe trains only the classifier",
    )
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--monitor", default="auc", choices=["auc", "bacc", "acc", "f1_macro"])
    p.add_argument("--freeze_encoder", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--drop_last", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not args.csv and not (args.train_csv and args.val_csv and args.test_csv):
        raise SystemExit("Provide either --csv or --train_csv/--val_csv/--test_csv")
    main(args)
