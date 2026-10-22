import os
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Extract EyeCLIP embeddings")
    p.add_argument("--cuda", type=str, default="2")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--diseases", type=str, default="health,AMD", help="Optional comma-separated subset, e.g. Health,AMD,Glaucoma")

    p.add_argument("--clip_model_type", type=str, default="ViT-B/32")
    p.add_argument("--download_root", type=str, required=True)
    p.add_argument("--base_checkpoint", type=str, required=True, help="EyeCLIP pretrained .pt; passed to eyeclip.load/load state if needed")
    p.add_argument("--checkpoint", type=str, required=True, help="Finetuned best.pt. Empty means base EyeCLIP.")
    p.add_argument("--num_classes", type=int, default=2, help="Only needed to rebuild finetuned classifier head shape.")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--run_name", type=str, default="eyeclip_base")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import src.eyeclip as eyeclip

    class HospitalFundusDataset(Dataset):
        def __init__(self, csv_path, image_root="", diseases="", preprocess=None):
            df = pd.read_csv(csv_path)
            required = ["image", "patient_id", "eye", "equipment", "disease"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}. Current columns: {list(df.columns)}")
            if diseases:
                keep = [x.strip() for x in diseases.split(",") if x.strip()]
                df = df[df["disease"].astype(str).isin(keep)].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError("No samples after filtering.")
            self.df = df.reset_index(drop=True)
            self.image_root = image_root
            self.preprocess = preprocess

        def __len__(self):
            return len(self.df)

        def _resolve(self, path):
            path = str(path)
            return path if os.path.isabs(path) else os.path.join(self.image_root, path)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = self._resolve(row["image"])
            try:
                image = Image.open(path).convert("RGB")
                image = self.preprocess(image)
            except Exception as e:
                raise RuntimeError(f"Failed to read image: {path}") from e
            return {
                "image": image,
                "image_path": path,
                "patient_id": str(row["patient_id"]),
                "eye": str(row["eye"]),
                "equipment": str(row["equipment"]),
                "disease": str(row["disease"]),
                "shooting_date": str(row["shooting_date"]) if "shooting_date" in self.df.columns else "",
            }

    def collate_fn(batch):
        images = torch.stack([b["image"] for b in batch], dim=0)
        out = {"image": images}
        for k in ["image_path", "patient_id", "eye", "equipment", "disease", "shooting_date"]:
            out[k] = [b[k] for b in batch]
        return out

    def get_state_dict(ckpt):
        if isinstance(ckpt, dict):
            for key in ["model_state_dict", "state_dict", "model", "module"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]
        return ckpt

    def strip_module_prefix(state):
        return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, CUDA_VISIBLE_DEVICES={args.cuda}")

    model, preprocess = eyeclip.load(args.clip_model_type, device=device, jit=False, download_root=args.download_root)
    model.float()
    if not hasattr(model, "visual"):
        raise RuntimeError("Loaded EyeCLIP model has no .visual encoder.")

    visual_dim = getattr(model.visual, "output_dim", None)
    if visual_dim is None:
        raise RuntimeError("Cannot infer model.visual.output_dim. Please inspect EyeCLIP visual encoder.")
    model.visual.head = nn.Linear(visual_dim, args.num_classes).to(device)

    # Some EyeCLIP repos load the base checkpoint through eyeclip.load(download_root=...),
    # while the finetuning code also supports an explicit .pt state. Keep this fallback robust.
    if args.base_checkpoint and os.path.exists(args.base_checkpoint) and not args.checkpoint:
        try:
            base_ckpt = torch.load(args.base_checkpoint, map_location="cpu")
            base_state = strip_module_prefix(get_state_dict(base_ckpt))
            base_state = {k: v for k, v in base_state.items() if not k.endswith("head.weight") and not k.endswith("head.bias")}
            msg = model.load_state_dict(base_state, strict=False)
            print(f"Loaded base checkpoint: {args.base_checkpoint}; missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        except Exception as e:
            print(f"[WARN] Explicit base checkpoint load failed, continuing with eyeclip.load weights: {e}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = strip_module_prefix(get_state_dict(ckpt))
        msg = model.load_state_dict(state, strict=False)
        print(f"Loaded finetuned checkpoint: {args.checkpoint}; missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    else:
        print("No finetuned checkpoint provided. Using base EyeCLIP visual encoder.")

    dataset = HospitalFundusDataset(args.csv, args.image_root, args.diseases, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    model.eval()
    feats = []
    meta = {k: [] for k in ["image_path", "patient_id", "eye", "equipment", "disease", "shooting_date"]}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract {args.run_name}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                feat = model.visual(images)
            feats.append(feat.float().cpu().numpy())
            for k in meta:
                meta[k].extend(batch[k])

    features = np.concatenate(feats, axis=0).astype("float32")
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "embeddings.npz",
        features=features,
        image_path=np.array(meta["image_path"], dtype=object),
        patient_id=np.array(meta["patient_id"], dtype=object),
        eye=np.array(meta["eye"], dtype=object),
        equipment=np.array(meta["equipment"], dtype=object),
        disease=np.array(meta["disease"], dtype=object),
        shooting_date=np.array(meta["shooting_date"], dtype=object),
    )
    pd.DataFrame(meta).to_csv(out_dir / "meta.csv", index=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_dir / 'embeddings.npz'}")
    print(f"features shape: {features.shape}")


if __name__ == "__main__":
    main()
