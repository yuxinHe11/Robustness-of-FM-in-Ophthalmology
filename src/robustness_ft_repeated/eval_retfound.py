import os
import json
import argparse
from pathlib import Path
from functools import partial
from itertools import combinations
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm.models.vision_transformer

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CSVVisionTransformer(timm.models.vision_transformer.VisionTransformer):
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
        bsz = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
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
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
        )

    if arch == "vit_large_patch16":
        return CSVVisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            global_pool=global_pool,
        )

    raise ValueError(f"Unsupported arch: {arch}")


def torch_load_compat(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_state_dict_from_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "module", "teacher"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor]):
    new_state = {}
    for k, v in state_dict.items():
        for prefix in ["module.", "backbone."]:
            if k.startswith(prefix):
                k = k[len(prefix):]
        new_state[k] = v
    return new_state


def infer_vit_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> str:
    w = None
    for k, v in sd.items():
        if k.endswith("patch_embed.proj.weight"):
            w = v
            break

    if w is None:
        print("[WARN] Cannot infer arch from checkpoint. Fallback to vit_large_patch16.")
        return "vit_large_patch16"

    dim = int(w.shape[0])
    if dim == 768:
        return "vit_base_patch16"
    if dim == 1024:
        return "vit_large_patch16"

    raise ValueError(f"Unknown ViT embed dim: {dim}")


def infer_num_classes_from_state_dict(sd: Dict[str, torch.Tensor], default: int):
    for k, v in sd.items():
        if k.endswith("head.weight") and len(v.shape) == 2:
            return int(v.shape[0])
    return default


def interpolate_pos_embed_if_needed(model: nn.Module, checkpoint_model: Dict[str, torch.Tensor]) -> None:
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
        pos_tokens,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False,
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)


def load_state_flexible(model: nn.Module, state_dict: Dict[str, torch.Tensor], desc: str):
    state_dict = strip_prefix_if_present(state_dict)
    model_state = model.state_dict()

    remove_keys = []
    for k, v in state_dict.items():
        if k in model_state and tuple(v.shape) != tuple(model_state[k].shape):
            remove_keys.append(k)

    for k in remove_keys:
        print(f"[{desc}] remove mismatched key: {k}, ckpt={tuple(state_dict[k].shape)}, model={tuple(model_state[k].shape)}")
        del state_dict[k]

    interpolate_pos_embed_if_needed(model, state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[{desc}] load message: {msg}")


def build_retfound_model(args):
    if not args.checkpoint:
        raise ValueError("External validation requires --checkpoint pointing to finetuned best.pt")

    ckpt = torch_load_compat(args.checkpoint)
    ft_state = strip_prefix_if_present(get_state_dict_from_checkpoint(ckpt))

    arch = args.model
    if arch == "auto":
        arch = infer_vit_arch_from_state_dict(ft_state)

    num_classes = infer_num_classes_from_state_dict(ft_state, args.num_classes)
    print(f"Building RETFound: arch={arch}, num_classes={num_classes}, global_pool={args.global_pool}")

    model = make_vit(
        arch=arch,
        num_classes=num_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.pretrained:
        print(f"Loading base RETFound checkpoint first: {args.pretrained}")
        base_ckpt = torch_load_compat(args.pretrained)
        base_state = get_state_dict_from_checkpoint(base_ckpt)
        load_state_flexible(model, base_state, "base")

    print(f"Loading finetuned checkpoint: {args.checkpoint}")
    load_state_flexible(model, ft_state, "finetuned")

    return model, num_classes


def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def standardize_csv(csv_path, image_root, diseases):
    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "label": "disease",
        "patientID": "patient_id",
        "relative_path": "image",
    })

    if "image" not in df.columns:
        if "filename" not in df.columns:
            raise ValueError("CSV must contain image/relative_path or filename.")
        df["image"] = df["filename"]

    required = ["image", "disease", "patient_id", "eye", "equipment"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}. Current columns: {list(df.columns)}")

    keep = [x.strip() for x in diseases.split(",") if x.strip()]
    if keep:
        df = df[df["disease"].astype(str).isin(keep)].copy()

    if len(df) == 0:
        raise ValueError(f"No samples after disease filtering: {keep}")

    for c in [
        "image", "disease", "patient_id", "eye", "equipment",
        "shooting_date", "sex", "gender_code", "date_of_birth",
        "age_at_shooting", "filename",
    ]:
        if c in df.columns:
            df[c] = df[c].map(norm_str)

    def resolve(p):
        p = str(p)
        if os.path.isabs(p):
            return p
        return os.path.join(image_root, p) if image_root else p

    df["image_path"] = df["image"].apply(resolve)

    for c in ["sex", "gender_code", "date_of_birth", "age_at_shooting", "shooting_date"]:
        if c not in df.columns:
            df[c] = ""

    return df.reset_index(drop=True)


def infer_class_order(diseases, positive_label, class_order):
    if class_order:
        classes = [x.strip() for x in class_order.split(",") if x.strip()]
    else:
        labels = [x.strip() for x in diseases.split(",") if x.strip()]
        health_like = [x for x in labels if x.lower() in ["health", "healthy", "normal", "control"]]
        others = [x for x in labels if x not in health_like]
        classes = health_like + sorted(others)

    if positive_label not in classes:
        raise ValueError(f"positive_label={positive_label} not in classes={classes}")

    return classes, classes.index(positive_label)


class HospitalFundusDataset(Dataset):
    def __init__(self, df, image_size=224):
        self.df = df.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["image_path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {path}") from e

        return {
            "image": self.transform(img),
            "idx": idx,
        }


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "idx": [b["idx"] for b in batch],
    }


def build_loader(args, df):
    dataset = HospitalFundusDataset(df, image_size=args.image_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def attach_predictions(df, logits_all, probs_all, classes, positive_index):
    pred_idx = probs_all.argmax(axis=1)
    pred_label = [classes[i] if i < len(classes) else str(i) for i in pred_idx]

    out = df.copy()

    for k in range(probs_all.shape[1]):
        name = classes[k] if k < len(classes) else f"class{k}"
        out[f"logit_{name}"] = logits_all[:, k]
        out[f"prob_{name}"] = probs_all[:, k]

    out["pred_index"] = pred_idx
    out["pred_label"] = pred_label
    out["positive_label"] = classes[positive_index]
    out["prob_positive"] = probs_all[:, positive_index]
    out["target_positive"] = (out["disease"] == classes[positive_index]).astype(int)

    return out


def run_inference(args, df, classes, positive_index):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader = build_loader(args, df)

    model, actual_num_classes = build_retfound_model(args)
    model = model.to(device)
    model.eval()

    if actual_num_classes != len(classes):
        raise ValueError(
            f"Checkpoint head has {actual_num_classes} classes, but class_order has {len(classes)} classes: {classes}. "
            f"Use --class_order to match training order."
        )

    logits_all = np.zeros((len(df), actual_num_classes), dtype=np.float32)
    probs_all = np.zeros((len(df), actual_num_classes), dtype=np.float32)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Infer {args.run_name}"):
            images = batch["image"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(images)
                probs = F.softmax(logits.float(), dim=1)

            idxs = np.array(batch["idx"], dtype=int)
            logits_all[idxs] = logits.float().cpu().numpy()
            probs_all[idxs] = probs.float().cpu().numpy()

    return attach_predictions(df, logits_all, probs_all, classes, positive_index)


def summarize(vals):
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan}

    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
    }


def build_prediction_pairs(pred):
    rows = []
    keys = ["patient_id", "eye", "disease", "equipment", "shooting_date"]
    for group_key, g in pred.groupby(keys, sort=False, dropna=False):
        if len(g) < 2:
            continue
        if len(g) != 2:
            raise ValueError(f"Expected two repeat images for {group_key}, found {len(g)}")

        i, j = sorted(g.index.tolist(), key=lambda k: str(pred.at[k, "image_path"]))
        if pred.at[i, "image_path"] == pred.at[j, "image_path"]:
            raise ValueError(f"Duplicate image in repeat pair: {group_key}")
        p_i = float(pred.at[i, "prob_positive"])
        p_j = float(pred.at[j, "prob_positive"])
        pred_pos_i = int(pred.at[i, "pred_label"] == pred.at[i, "positive_label"])
        pred_pos_j = int(pred.at[j, "pred_label"] == pred.at[j, "positive_label"])
        rows.append({
            "disease": pred.at[i, "disease"],
            "patient_id": pred.at[i, "patient_id"],
            "eye": pred.at[i, "eye"],
            "shooting_date": pred.at[i, "shooting_date"],
            "equipment_i": pred.at[i, "equipment"],
            "equipment_j": pred.at[j, "equipment"],
            "equipment_pair": f"{pred.at[i, 'equipment']} -> {pred.at[j, 'equipment']}",
            "image_i": pred.at[i, "image_path"],
            "image_j": pred.at[j, "image_path"],
            "prob_positive_i": p_i,
            "prob_positive_j": p_j,
            "probability_shift_signed": p_j - p_i,
            "probability_shift_abs": abs(p_j - p_i),
            "pred_i": pred.at[i, "pred_label"],
            "pred_j": pred.at[j, "pred_label"],
            "prediction_consistent": float(pred.at[i, "pred_label"] == pred.at[j, "pred_label"]),
            "target_positive_i": int(pred.at[i, "target_positive"]),
            "target_positive_j": int(pred.at[j, "target_positive"]),
            "pred_positive_i": pred_pos_i,
            "pred_positive_j": pred_pos_j,
            "both_correct_binary": float(
                int(pred.at[i, "target_positive"]) == pred_pos_i
                and int(pred.at[j, "target_positive"]) == pred_pos_j
            ),
        })
    if not rows:
        raise ValueError("No same-eye, same-scanner repeat pairs found")
    return pd.DataFrame(rows)


def evaluate_prediction_robustness(pred):
    detail = build_prediction_pairs(pred)
    metrics = {
        "probability_shift_abs": "probability_shift_abs",
        "prediction_consistency": "prediction_consistent",
        "probability_shift_signed": "probability_shift_signed",
    }
    rows = []
    for disease in sorted(pred["disease"].unique().tolist()) + ["ALL"]:
        group = detail if disease == "ALL" else detail[detail["disease"] == disease] if len(detail) else detail
        for metric, column in metrics.items():
            rows.append({
                "metric": metric,
                "disease": disease,
                **summarize(group[column].values if len(group) else []),
            })

    pair_rows = []
    if len(detail):
        for (disease, pair), group in detail.groupby(["disease", "equipment_pair"]):
            for metric, column in metrics.items():
                pair_rows.append({
                    "disease": disease,
                    "equipment_pair": pair,
                    "metric": metric,
                    **summarize(group[column].values),
                })
    return pd.DataFrame(rows), detail, pd.DataFrame(pair_rows)


def evaluate_classification_metrics(pred, classes, positive_index):
    y_true = pred["target_positive"].astype(int).values
    y_prob = pred["prob_positive"].astype(float).values
    y_pred = (pred["pred_index"].astype(int).values == int(positive_index)).astype(int)

    out = {
        "n": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "positive_label": classes[positive_index],
        "negative_label": classes[1 - positive_index] if len(classes) == 2 else "",
        "auroc": np.nan,
        "auprc": np.nan,
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if len(np.unique(y_true)) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out["tn"] = int(cm[0, 0])
    out["fp"] = int(cm[0, 1])
    out["fn"] = int(cm[1, 0])
    out["tp"] = int(cm[1, 1])

    return pd.DataFrame([out])


def save_outputs(args, pred, classes, positive_index):
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pred.to_csv(out_dir / "predictions.csv", index=False)

    cls_metrics = evaluate_classification_metrics(pred, classes, positive_index)
    cls_metrics.insert(0, "run_name", args.run_name)
    cls_metrics.to_csv(out_dir / "classification_metrics.csv", index=False)

    summary, detail, by_equipment = evaluate_prediction_robustness(pred)
    summary.insert(0, "run_name", args.run_name)

    summary.to_csv(out_dir / "prediction_robustness_summary.csv", index=False)
    detail.to_csv(out_dir / "prediction_robustness_pair_detail.csv", index=False)
    by_equipment.to_csv(out_dir / "prediction_robustness_by_equipment_pair.csv", index=False)

    cfg = vars(args).copy()
    cfg["class_order_final"] = classes
    cfg["positive_index"] = positive_index

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print("\nClassification metrics:")
    print(cls_metrics.to_string(index=False))

    print("\nPrediction robustness summary:")
    print(summary.to_string(index=False))

    print(f"\nSaved to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--cuda", type=str, default="2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--diseases", type=str, default="health,AMD")
    p.add_argument("--positive_label", type=str, default="AMD")
    p.add_argument("--class_order", type=str, default="", help="Default: health-like first, disease second.")

    p.add_argument("--pretrained", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, default="vit_large_patch16", choices=["auto", "vit_base_patch16", "vit_large_patch16"])
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--drop_path", type=float, default=0.0)
    p.add_argument("--global_pool", action="store_true", default=True)
    p.add_argument("--cls_token", action="store_false", dest="global_pool")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default="amd")

    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    classes, positive_index = infer_class_order(
        args.diseases,
        args.positive_label,
        args.class_order,
    )

    df = standardize_csv(args.csv, args.image_root, args.diseases)

    print(f"Loaded hospital samples: {len(df)}")
    print(df["disease"].value_counts().to_string())
    print(f"Class order: {classes}, positive_index={positive_index}")

    pred = run_inference(args, df, classes, positive_index)
    save_outputs(args, pred, classes, positive_index)


if __name__ == "__main__":
    main()