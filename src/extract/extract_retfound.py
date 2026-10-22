import os
import json
import argparse
from pathlib import Path
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm.models.vision_transformer

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CSVVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    RETFound/MAE-style ViT classifier.

    Important:
    - Do not use transformers.
    - Do not call original timm forward_features with unexpected newer-version args.
    - This custom forward_features is stable across timm versions used in RETFound-style code.
    """

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
        feat = self.forward_features(x)
        return self.head(feat)


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
    state_for_arch = None

    if args.checkpoint:
        ckpt = torch_load_compat(args.checkpoint)
        state_for_arch = strip_prefix_if_present(get_state_dict_from_checkpoint(ckpt))
    elif args.pretrained:
        ckpt = torch_load_compat(args.pretrained)
        state_for_arch = strip_prefix_if_present(get_state_dict_from_checkpoint(ckpt))

    arch = args.model
    if arch == "auto":
        if state_for_arch is None:
            raise ValueError("--model auto requires --pretrained or --checkpoint")
        arch = infer_vit_arch_from_state_dict(state_for_arch)

    num_classes = args.num_classes
    if args.checkpoint:
        num_classes = infer_num_classes_from_state_dict(state_for_arch, args.num_classes)

    print(f"Building RETFound: arch={arch}, num_classes={num_classes}, global_pool={args.global_pool}")

    model = make_vit(
        arch=arch,
        num_classes=num_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.pretrained:
        print(f"Loading base RETFound checkpoint: {args.pretrained}")
        ckpt = torch_load_compat(args.pretrained)
        base_state = get_state_dict_from_checkpoint(ckpt)
        load_state_flexible(model, base_state, "base")

    if args.checkpoint:
        print(f"Loading finetuned checkpoint: {args.checkpoint}")
        ckpt = torch_load_compat(args.checkpoint)
        ft_state = get_state_dict_from_checkpoint(ckpt)
        load_state_flexible(model, ft_state, "finetuned")

    return model


def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def standardize_hospital_csv(csv_path, image_root, diseases):
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

    if diseases:
        keep = [x.strip() for x in diseases.split(",") if x.strip()]
        df = df[df["disease"].astype(str).isin(keep)].copy()

    if len(df) == 0:
        raise ValueError(f"No samples after disease filtering: {diseases}")

    for c in [
        "image", "disease", "patient_id", "eye", "equipment",
        "shooting_date", "sex", "gender_code", "date_of_birth",
        "age_at_shooting", "filename",
    ]:
        if c in df.columns:
            df[c] = df[c].map(norm_str)

    def resolve_path(p):
        p = str(p)
        if os.path.isabs(p):
            return p
        return os.path.join(image_root, p) if image_root else p

    df["image_path"] = df["image"].apply(resolve_path)

    for c in ["shooting_date", "sex", "gender_code", "date_of_birth", "age_at_shooting"]:
        if c not in df.columns:
            df[c] = ""

    return df.reset_index(drop=True)


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
        row = self.df.iloc[idx]
        path = row["image_path"]
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


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--cuda", type=str, default="2")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--diseases", type=str, default="health,AMD")

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

    p.add_argument("--run_name", type=str, default="retfound_base")
    p.add_argument("--output_dir", type=str, required=True)

    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, CUDA_VISIBLE_DEVICES={args.cuda}")

    df = standardize_hospital_csv(args.csv, args.image_root, args.diseases)
    print(f"Loaded hospital samples: {len(df)}")
    print(df["disease"].value_counts().to_string())

    dataset = HospitalFundusDataset(df, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_retfound_model(args).to(device)
    model.eval()

    feats = np.zeros((len(df), model.num_features), dtype=np.float32)

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract {args.run_name}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                feat = model.forward_features(images)

            idxs = np.array(batch["idx"], dtype=int)
            feats[idxs] = feat.float().cpu().numpy()

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = [
        "image_path", "patient_id", "eye", "equipment", "disease",
        "shooting_date", "sex", "gender_code", "date_of_birth", "age_at_shooting",
    ]
    meta = df[[c for c in meta_cols if c in df.columns]].copy()

    np.savez_compressed(
        out_dir / "embeddings.npz",
        features=feats,
        image_path=meta["image_path"].astype(str).values,
        patient_id=meta["patient_id"].astype(str).values,
        eye=meta["eye"].astype(str).values,
        equipment=meta["equipment"].astype(str).values,
        disease=meta["disease"].astype(str).values,
        shooting_date=meta["shooting_date"].astype(str).values if "shooting_date" in meta.columns else np.array([""] * len(meta), dtype=object),
    )

    meta.to_csv(out_dir / "meta.csv", index=False)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"Saved embeddings: {out_dir / 'embeddings.npz'}")
    print(f"Saved meta:       {out_dir / 'meta.csv'}")
    print(f"features shape:   {feats.shape}")


if __name__ == "__main__":
    main()