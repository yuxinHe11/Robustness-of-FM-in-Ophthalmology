import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import csv
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

import src.visionfm as visionfm
import src.visionfm.utils_visionfm as utils
from src.visionfm.head import ClsHead


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class EyeCSVDataset(Dataset):
    """
    CSV columns:
        image,label,dataset,split

    Relative image paths are resolved under image_root. Absolute paths are
    used as-is.

    task:
        AMD: labels 0/1 -> local 0/1
        Glaucoma: labels 2/3 -> local 0/1
    """
    def __init__(self, csv_path, image_root, split, task, transform=None):
        self.csv_path = csv_path
        self.image_root = Path(image_root).expanduser()
        self.split = split
        self.task = task
        self.transform = transform
        self.samples = []

        task = task.lower()
        if task == "amd":
            valid_labels = {0: 0, 1: 1}
        elif task == "glaucoma":
            valid_labels = {2: 0, 3: 1}
        else:
            raise ValueError("task must be AMD or Glaucoma")

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue

                raw_label = int(row["label"])
                if raw_label not in valid_labels:
                    continue

                img_path = Path(row["image"].strip()).expanduser()
                if not img_path.is_absolute():
                    img_path = self.image_root / img_path
                local_label = valid_labels[raw_label]
                self.samples.append((str(img_path), local_label, raw_label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={split}, task={task}, csv={csv_path}")

        counts = {}
        for _, y, _ in self.samples:
            counts[y] = counts.get(y, 0) + 1
        print(f"[{task}][{split}] n={len(self.samples)}, local label counts={counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, raw_label = self.samples[idx]
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), img_path, raw_label


def build_transforms(input_size, modality):
    mean, std = utils.get_stats(modality)
    print(f"use {modality} mean={mean}, std={std}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf


def build_model(args, device):
    model = visionfm.__dict__[args.arch](
        img_size=[args.input_size],
        patch_size=args.patch_size,
        num_classes=0,
        use_mean_pooling=(args.avgpool_patchtokens == 1),
    )
    embed_dim = model.embed_dim

    model.to(device)
    print(f"Model {args.arch}, patch_size={args.patch_size}, embed_dim={embed_dim}")

    utils.load_pretrained_weights(
        model,
        args.pretrained_weights,
        args.checkpoint_key,
        args.arch,
        args.patch_size,
    )

    if args.avgpool_patchtokens == 2:
        head_dim = embed_dim * (args.n_last_blocks + 1)
    else:
        head_dim = embed_dim * args.n_last_blocks

    head_layers = 1 if args.training_mode == "linear_probe" else args.head_layers
    if args.training_mode == "linear_probe" and args.head_layers != 1:
        print(f"Linear probe uses a one-layer classifier (ignoring --head_layers={args.head_layers}).")
    classifier = ClsHead(embed_dim=head_dim, num_classes=args.num_labels, layers=head_layers)
    classifier.to(device)

    return model, classifier


def extract_features(model, x, n_last_blocks, avgpool_patchtokens):
    intermediate_output = model.get_intermediate_layers(x, n_last_blocks)

    if avgpool_patchtokens == 0:
        feats = [o[:, 0] for o in intermediate_output]
    elif avgpool_patchtokens == 1:
        feats = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
    elif avgpool_patchtokens == 2:
        feats = [o[:, 0] for o in intermediate_output]
        feats.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
    else:
        raise ValueError(f"Unknown avgpool_patchtokens={avgpool_patchtokens}")

    return torch.cat(feats, dim=-1)


def compute_metrics(y_true, prob, loss):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)

    if prob.shape[1] == 2:
        pred = prob.argmax(axis=1)
        pos_prob = prob[:, 1]
        auc = roc_auc_score(y_true, pos_prob)
        aupr = average_precision_score(y_true, pos_prob)
    else:
        pred = prob.argmax(axis=1)
        auc = roc_auc_score(y_true, prob, average="macro", multi_class="ovr")
        aupr = average_precision_score(np.eye(prob.shape[1])[y_true], prob, average="macro")

    return {
        "loss": float(loss),
        "auc": float(auc),
        "aupr": float(aupr),
        "acc": float(accuracy_score(y_true, pred)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def train_one_epoch(model, classifier, loader, optimizer, device, args, epoch):
    model.train()
    classifier.train()
    if args.training_mode == "linear_probe":
        # Keep dropout and other training-time backbone behavior disabled.
        model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0

    for step, batch in enumerate(loader):
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        feats = extract_features(model, x, args.n_last_blocks, args.avgpool_patchtokens)
        logits = classifier(feats)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(model.parameters()) + list(classifier.parameters()) if p.requires_grad],
                args.grad_clip,
            )

        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        if step % args.print_freq == 0:
            print(f"Epoch {epoch} [{step}/{len(loader)}] loss={loss.item():.4f}")

    return {"loss": total_loss / max(total_n, 1)}


@torch.no_grad()
def evaluate(model, classifier, loader, device, args, split):
    model.eval()
    classifier.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0

    all_y = []
    all_prob = []
    all_paths = []
    all_raw_labels = []

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        paths = batch[2]
        raw_labels = batch[3]

        feats = extract_features(model, x, args.n_last_blocks, args.avgpool_patchtokens)
        logits = classifier(feats)
        loss = criterion(logits, y)

        prob = torch.softmax(logits, dim=1)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        all_y.extend(y.cpu().numpy().tolist())
        all_prob.append(prob.cpu().numpy())
        all_paths.extend(paths)
        all_raw_labels.extend([int(v) for v in raw_labels])

    all_prob = np.concatenate(all_prob, axis=0)
    avg_loss = total_loss / max(total_n, 1)

    metrics = compute_metrics(all_y, all_prob, avg_loss)
    print(
        f"[{split}] loss={metrics['loss']:.4f} "
        f"auc={metrics['auc']:.4f} aupr={metrics['aupr']:.4f} "
        f"acc={metrics['acc']:.4f} f1={metrics['f1_macro']:.4f}"
    )
    print(f"[{split}] confusion_matrix={metrics['confusion_matrix']}")

    return metrics, np.asarray(all_y), all_prob, all_paths, np.asarray(all_raw_labels)


def save_predictions(path, y_true, prob, img_paths, raw_labels):
    pred = prob.argmax(axis=1)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        cols = ["image", "raw_label", "local_label", "pred", "prob_0", "prob_1"]
        writer.writerow(cols)
        for p, raw, y, pr, pb in zip(img_paths, raw_labels, y_true, pred, prob):
            writer.writerow([p, int(raw), int(y), int(pr), float(pb[0]), float(pb[1])])


def main(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.fix_random_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_tf, eval_tf = build_transforms(args.input_size, args.modality)

    train_set = EyeCSVDataset(args.csv_path, args.image_root, "train", args.task, train_tf)
    val_set = EyeCSVDataset(args.csv_path, args.image_root, "val", args.task, eval_tf)
    test_set = EyeCSVDataset(args.csv_path, args.image_root, "test", args.task, eval_tf)

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

    model, classifier = build_model(args, device)

    if args.training_mode == "linear_probe":
        for p in model.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable += sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in classifier.parameters())
    print(f"Training mode: {args.training_mode} | trainable params: {trainable:,}/{total:,}")

    param_groups = [
        {
            "params": classifier.parameters(),
            "lr": args.lr_head,
            "group_name": "head",
        }
    ]
    if args.training_mode == "finetune":
        param_groups.insert(
            0,
            {
                "params": model.parameters(),
                "lr": args.lr_backbone,
                "group_name": "backbone",
            },
        )
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    start_epoch = 0
    best_auc = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_mode = ckpt.get("args", {}).get("training_mode", "finetune")
        if saved_mode != args.training_mode:
            raise ValueError(
                f"Checkpoint training_mode={saved_mode!r}, but current mode={args.training_mode!r}. "
                "Resume with the same --training_mode."
            )
        model.load_state_dict(ckpt["visionfm_state_dict"], strict=False)
        classifier.load_state_dict(ckpt["classifier_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        best_auc = ckpt.get("best_auc", -1.0)
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_auc={best_auc:.4f}")

    best_path = os.path.join(args.output_dir, "best.pt")
    last_path = os.path.join(args.output_dir, "last.pt")
    log_path = os.path.join(args.output_dir, "log.txt")

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, classifier, train_loader, optimizer, device, args, epoch)
        scheduler.step()

        val_metrics, _, _, _, _ = evaluate(model, classifier, val_loader, device, args, "val")

        log = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr_backbone": optimizer.param_groups[0]["lr"] if args.training_mode == "finetune" else 0.0,
            "lr_head": optimizer.param_groups[-1]["lr"],
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

        save_dict = {
            "epoch": epoch + 1,
            "visionfm_state_dict": model.state_dict(),
            "classifier_state_dict": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_auc": best_auc,
            "args": vars(args),
        }
        torch.save(save_dict, last_path)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            save_dict["best_auc"] = best_auc
            torch.save(save_dict, best_path)
            print(f"Saved best checkpoint: {best_path}, best_auc={best_auc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}, best_auc={best_auc:.4f}")
            break

    print("Training finished. Loading best checkpoint for final test...")
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["visionfm_state_dict"], strict=False)
    classifier.load_state_dict(ckpt["classifier_state_dict"], strict=False)

    test_metrics, y_true, prob, img_paths, raw_labels = evaluate(
        model, classifier, test_loader, device, args, "test"
    )

    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(args.output_dir, "test_prob.npy"), prob)
    np.save(os.path.join(args.output_dir, "test_target.npy"), y_true)
    save_predictions(os.path.join(args.output_dir, "test_predictions.csv"), y_true, prob, img_paths, raw_labels)

    print("Final test metrics:")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetune VisionFM on eye CSV")

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Root directory prepended to relative image paths in --csv_path",
    )
    parser.add_argument("--task", type=str, default="Glaucoma", choices=["AMD", "Glaucoma"])
    parser.add_argument("--pretrained_weights", type=str, required=True)
    parser.add_argument("--checkpoint_key", type=str, default="teacher")

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--arch", type=str, default="vit_base", choices=["vit_tiny", "vit_small", "vit_base", "vit_large"])
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--modality", type=str, default="OCT")

    parser.add_argument("--n_last_blocks", type=int, default=4)
    parser.add_argument("--avgpool_patchtokens", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--head_layers", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument(
        "--training_mode", default="linear_probe", choices=["finetune", "linear_probe"],
        help="finetune updates backbone and head; linear_probe freezes backbone and trains a linear head",
    )
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()
    main(args)
