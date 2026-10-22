import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import csv
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from src.datasets import FundusDataset
from src.model.models_retfound_dinov2 import RETFoundDINOv2Classifier
from src.metrics import compute_metrics
from src.utils import (
    set_seed,
    ensure_dir,
    save_json,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)

    # model
    parser.add_argument(
        "--arch",
        type=str,
        default="vit_large_patch14_dinov2",
    )
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default="retfound_dinov2",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["linear_probe", "finetune"],
        default="linear_probe",
    )
    parser.add_argument("--dropout", type=float, default=0.0)

    # training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)

    # system
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def build_loaders(args):
    train_set = FundusDataset(
        csv_path=args.csv,
        split="train",
        image_root=args.image_root,
        image_size=args.image_size,
    )

    val_set = FundusDataset(
        csv_path=args.csv,
        split="val",
        image_root=args.image_root,
        image_size=args.image_size,
    )

    test_set = FundusDataset(
        csv_path=args.csv,
        split="test",
        image_root=args.image_root,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
):
    model.train()

    total_loss = 0.0
    total_num = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_num += bs

    return total_loss / max(total_num, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Eval"):
    model.eval()

    total_loss = 0.0
    total_num = 0

    all_labels = []
    all_probs = []
    all_paths = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_num += bs

        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_paths.extend(batch["path"])

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = float(total_loss / max(total_num, 1))

    return metrics, all_labels, all_probs, all_paths


def build_optimizer_and_scheduler(args, model):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("classifier."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []

    if len(backbone_params) > 0:
        param_groups.append({
            "params": backbone_params,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        })

    if len(head_params) > 0:
        param_groups.append({
            "params": head_params,
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
        })

    optimizer = AdamW(param_groups)

    warmup_epochs = min(args.warmup_epochs, args.epochs)

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - warmup_epochs, 1),
            eta_min=args.min_lr,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )

    return optimizer, scheduler


def write_log_header(log_path):
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_auroc",
            "val_auprc",
            "val_f1_macro",
            "val_f1_weighted",
            "val_acc",
            "val_bacc",
            "val_selection_score",
            "lr_backbone",
            "lr_head",
        ])


def append_log(log_path, row):
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_predictions(path, labels, probs, paths):
    ensure_dir(os.path.dirname(path))

    num_classes = len(probs[0])

    fieldnames = ["image", "label", "pred"] + [
        f"prob_{i}" for i in range(num_classes)
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_path, label, prob in zip(paths, labels, probs):
            pred = int(max(range(num_classes), key=lambda i: prob[i]))

            row = {
                "image": image_path,
                "label": int(label),
                "pred": pred,
            }

            for i, p in enumerate(prob):
                row[f"prob_{i}"] = float(p)

            writer.writerow(row)


def get_current_lrs(optimizer):
    if len(optimizer.param_groups) == 1:
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[0]["lr"]
    else:
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

    return lr_backbone, lr_head


def main():
    args = parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    save_json(vars(args), os.path.join(args.output_dir, "config.json"))

    train_loader, val_loader, test_loader = build_loaders(args)

    model = RETFoundDINOv2Classifier(
        arch=args.arch,
        pretrained_path=args.pretrained_path,
        num_classes=args.num_classes,
        mode=args.mode,
        dropout=args.dropout,
        image_size=args.image_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer, scheduler = build_optimizer_and_scheduler(args, model)

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    best_score = -1.0
    best_path = os.path.join(args.output_dir, "best.pt")
    last_path = os.path.join(args.output_dir, "last.pt")
    log_path = os.path.join(args.output_dir, "log.csv")

    write_log_header(log_path)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
        )

        val_metrics, _, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            desc="Val",
        )

        scheduler.step()

        lr_backbone, lr_head = get_current_lrs(optimizer)
        current_score = val_metrics["selection_score"]

        append_log(
            log_path,
            [
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["auroc"],
                val_metrics["auprc"],
                val_metrics["f1_macro"],
                val_metrics["f1_weighted"],
                val_metrics["acc"],
                val_metrics["bacc"],
                val_metrics["selection_score"],
                lr_backbone,
                lr_head,
            ],
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_auprc={val_metrics['auprc']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"val_bacc={val_metrics['bacc']:.4f} "
            f"score={current_score:.4f} "
            f"lr_backbone={lr_backbone:.2e} "
            f"lr_head={lr_head:.2e}"
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=best_score,
            args=args,
        )

        if current_score > best_score:
            best_score = current_score

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_score,
                args=args,
            )

            save_json(
                {
                    "epoch": epoch,
                    "best_score": best_score,
                    "val_metrics": val_metrics,
                },
                os.path.join(args.output_dir, "best_val_metrics.json"),
            )

            print(f"Saved best checkpoint: {best_path}")

    print("Loading best checkpoint for final test evaluation...")
    load_checkpoint(best_path, model, device)

    test_metrics, test_labels, test_probs, test_paths = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        desc="Test",
    )

    save_json(
        test_metrics,
        os.path.join(args.output_dir, "test_metrics.json"),
    )

    save_predictions(
        path=os.path.join(args.output_dir, "test_predictions.csv"),
        labels=test_labels,
        probs=test_probs,
        paths=test_paths,
    )

    print("Final Test Metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()