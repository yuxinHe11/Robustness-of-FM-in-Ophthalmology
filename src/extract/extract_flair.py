import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from src.model.models_flair_v2 import FLAIRRetinaClassifier
    from src.utils import ensure_dir, load_checkpoint
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from model.models_flair_v2 import FLAIRRetinaClassifier
    from utils import ensure_dir, load_checkpoint


ImageFile.LOAD_TRUNCATED_IMAGES = True


def norm_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def standardize_hospital_csv(csv_path, image_root, diseases):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"label": "disease", "patientID": "patient_id", "relative_path": "image"})
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

    for c in ["image", "disease", "patient_id", "eye", "equipment", "shooting_date", "sex", "gender_code", "date_of_birth", "age_at_shooting"]:
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


class HospitalFlairV2Dataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {path}") from e
        return {"image": self.transform(image), "idx": idx}


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "idx": [b["idx"] for b in batch],
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", type=str, default="6")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--diseases", type=str, default="health,AMD,Glaucoma")

    p.add_argument("--flair_repo", type=str, required=True, help="Local path to the official jusiro/FLAIR repo.")
    p.add_argument("--pretrained", type=str, required=True, help="Local official FLAIR weights, usually model.safetensors; no online download is used.")
    p.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned best.pt. Empty means base FLAIR v2.")
    p.add_argument("--vision_type", type=str, default="resnet_v1", choices=["resnet_v1", "resnet_v2", "efficientnet"])
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--projection", action="store_true", default=False)
    p.add_argument("--norm_features", action="store_true", default=False)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--mode", type=str, default="finetune", choices=["linear_probe", "partial_finetune", "finetune"])
    p.add_argument("--unfreeze_last_n_blocks", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--run_name", type=str, default="flair_base")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, CUDA_VISIBLE_DEVICES={args.cuda}")

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
    if args.checkpoint:
        print(f"Loading fine-tuned checkpoint: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model, device)
    else:
        print("No checkpoint provided. Using base official FLAIR v2 backbone.")

    df = standardize_hospital_csv(args.csv, args.image_root, args.diseases)
    print(f"Loaded hospital samples: {len(df)}")
    print(df["disease"].value_counts().to_string())

    loader = DataLoader(
        HospitalFlairV2Dataset(df, model.preprocess_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model.eval()
    feats = np.zeros((len(df), model.feature_dim), dtype=np.float32)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract {args.run_name}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                feat = model.encode_image_feature(images)
            idxs = np.array(batch["idx"], dtype=int)
            feats[idxs] = feat.float().cpu().numpy()

    out_dir = Path(args.output_dir) / args.run_name
    ensure_dir(str(out_dir))
    meta_cols = ["image_path", "patient_id", "eye", "equipment", "disease", "shooting_date", "sex", "gender_code", "date_of_birth", "age_at_shooting"]
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
