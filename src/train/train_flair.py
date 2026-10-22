import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from src.metrics import compute_metrics
    from src.model.models_flair_v2 import FLAIRRetinaClassifier
    from src.utils import ensure_dir, save_json
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from metrics import compute_metrics
    from model.models_flair_v2 import FLAIRRetinaClassifier
    from utils import ensure_dir, save_json


ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class FlairV2FundusDataset(Dataset):
    def __init__(self, csv_path: str, split: str, transform, image_root: str = ""):
        df_all = pd.read_csv(csv_path)
        required = ["image", "label", "split"]
        missing = [c for c in required if c not in df_all.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Current columns: {list(df_all.columns)}")

        labels = sorted(df_all["label"].unique().tolist())
        self.label_map = {label: idx for idx, label in enumerate(labels)}
        self.num_classes = len(labels)
        self.df = df_all[df_all["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No samples found for split={split}")

        self.transform = transform
        self.image_root = image_root
        print(f"[{split}] n={len(self.df)} labels={labels} label_map={self.label_map}")

    def __len__(self):
        return len(self.df)

    def resolve_path(self, image_path):
        image_path = str(image_path)
        if os.path.isabs(image_path):
            return image_path
        return os.path.join(self.image_root, image_path) if self.image_root else image_path

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.resolve_path(row["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {img_path}") from e
        return {
            "image": self.transform(image),
            "label": torch.tensor(self.label_map[row["label"]], dtype=torch.long),
            "path": img_path,
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--num_classes", type=int, default=2)

    p.add_argument("--flair_repo", type=str, required=True, help="Local path to the official jusiro/FLAIR repo.")
    p.add_argument("--pretrained", type=str, required=True, help="Local official FLAIR weights, usually model.safetensors; no online download is used.")
    p.add_argument("--vision_type", type=str, default="resnet_v1", choices=["resnet_v1", "resnet_v2", "efficientnet"])
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--projection", action="store_true", default=False)
    p.add_argument("--norm_features", action="store_true", default=False)
    p.add_argument("--mode", type=str, choices=["linear_probe", "partial_finetune", "finetune"], default="linear_probe")
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=15)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--cuda", type=str, default="3")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="amp")
    return p.parse_args()


def build_loaders(args, preprocess_train, preprocess_val):
    train_set = FlairV2FundusDataset(args.csv, "train", preprocess_train, args.image_root)
    val_set = FlairV2FundusDataset(args.csv, "val", preprocess_val, args.image_root)
    test_set = FlairV2FundusDataset(args.csv, "test", preprocess_val, args.image_root)
    if args.num_classes != train_set.num_classes:
        print(f"[WARN] args.num_classes={args.num_classes}, dataset num_classes={train_set.num_classes}; using dataset value.")
        args.num_classes = train_set.num_classes

    return (
        DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True),
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    )


def build_optimizer_and_scheduler(args, model):
    head_params = []
    backbone_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    optimizer = AdamW(param_groups)
    warmup_epochs = min(args.warmup_epochs, args.epochs)
    if warmup_epochs > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup_epochs, 1), eta_min=args.min_lr),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    return optimizer, scheduler


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, amp):
    model.train()
    total_loss = 0.0
    total_num = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_num += bs
    return total_loss / max(total_num, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp, desc="Eval"):
    model.eval()
    total_loss = 0.0
    total_num = 0
    all_labels, all_probs, all_paths = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits.float(), dim=1)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_num += bs
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_paths.extend(batch["path"])
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = float(total_loss / max(total_num, 1))
    return metrics, all_labels, all_probs, all_paths


def write_log_header(path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_auroc", "val_auprc", "val_f1_macro", "val_acc", "val_bacc", "val_selection_score", "head_lr", "backbone_lr"])


def append_log(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def save_predictions(path, labels, probs, paths):
    ensure_dir(os.path.dirname(path))
    num_classes = len(probs[0])
    fieldnames = ["image", "label", "pred"] + [f"prob_{i}" for i in range(num_classes)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for image_path, label, prob in zip(paths, labels, probs):
            row = {"image": image_path, "label": int(label), "pred": int(np.argmax(prob))}
            row.update({f"prob_{i}": float(p) for i, p in enumerate(prob)})
            writer.writerow(row)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, args):
    ensure_dir(os.path.dirname(path))
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "args": vars(args),
            "model_family": "official_jusiro_flair",
        },
        path,
    )


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = FLAIRRetinaClassifier(
        pretrained=args.pretrained,
        flair_repo=args.flair_repo,
        num_classes=args.num_classes,
        vision_type=args.vision_type,
        image_size=args.image_size,
        projection=args.projection,
        norm_features=args.norm_features,
        mode=args.mode,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks,
        dropout=args.dropout,
    ).to(device)
    train_loader, val_loader, test_loader = build_loaders(args, model.preprocess_train, model.preprocess_val)
    save_json(vars(args), os.path.join(args.output_dir, "config.json"))

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total / 1e6:.2f}M, trainable={trainable / 1e6:.2f}M, feature_dim={model.feature_dim}")

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_score = -1.0
    bad_epochs = 0
    best_path = os.path.join(args.output_dir, "best.pt")
    log_path = os.path.join(args.output_dir, "log.csv")
    write_log_header(log_path)

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, args.amp)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device, args.amp, desc="Val")
        scheduler.step()
        head_lr = optimizer.param_groups[0]["lr"]
        backbone_lr = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0
        score = val_metrics["selection_score"]
        append_log(log_path, [epoch, train_loss, val_metrics["loss"], val_metrics["auroc"], val_metrics["auprc"], val_metrics["f1_macro"], val_metrics["acc"], val_metrics["bacc"], score, head_lr, backbone_lr])
        print(f"Epoch [{epoch}/{args.epochs}] train_loss={train_loss:.4f} val_auroc={val_metrics['auroc']:.4f} val_f1={val_metrics['f1_macro']:.4f} score={score:.4f}")

        save_checkpoint(os.path.join(args.output_dir, "last.pt"), model, optimizer, scheduler, epoch, best_score, args)
        if score > best_score:
            best_score = score
            bad_epochs = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, args)
            save_json({"epoch": epoch, "best_score": best_score, "val_metrics": val_metrics}, os.path.join(args.output_dir, "best_val_metrics.json"))
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    test_metrics, labels, probs, paths = evaluate(model, test_loader, criterion, device, args.amp, desc="Test")
    save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))
    save_predictions(os.path.join(args.output_dir, "test_predictions.csv"), labels, probs, paths)
    print(f"Best score={best_score:.4f}; test_metrics={json.dumps(test_metrics, indent=2)}")
    print(f"Done. Training time: {(time.time() - start) / 60:.1f} min. Outputs saved to: {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
