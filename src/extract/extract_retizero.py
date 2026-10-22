import os
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Extract RetiZero embeddings")
    p.add_argument("--cuda", type=str, default="2")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--diseases", type=str, default="health,AMD,Glaucoma", help="Optional comma-separated subset, e.g. Health,AMD,Glaucoma")
    p.add_argument("--retizero_repo", type=str, required=True)
    p.add_argument("--weight_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Finetuned best.pt. Empty means base RetiZero.")
    p.add_argument("--vision_type", type=str, default="lora")
    p.add_argument("--module_name", type=str, default="iden_modules")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--freeze_encoder", action="store_true")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--run_name", type=str, default="retizero_base")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    from src.utils_retizero import RetiZeroClassifier, default_retizero_transform

    class HospitalFundusDataset(Dataset):
        def __init__(self, csv_path, image_root="", diseases="", transform=None):
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
            self.transform = transform

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
                image = self.transform(image)
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

    def extract_by_hook(model, images):
        """Capture the tensor entering model.classifier, i.e. pre-classifier embedding."""
        captured = {}
        if not hasattr(model, "classifier"):
            raise RuntimeError("RetiZeroClassifier has no .classifier. Please inspect src/utils_retizero.py for its feature method.")

        def hook(_module, input_tuple, _output):
            captured["feat"] = input_tuple[0]

        handle = model.classifier.register_forward_hook(hook)
        try:
            _ = model(images)
        finally:
            handle.remove()
        if "feat" not in captured:
            raise RuntimeError("Failed to capture classifier input. Please inspect RetiZeroClassifier.forward().")
        return captured["feat"]

    def extract_features(model, images):
        # Prefer explicit feature methods if your local utils_retizero.py defines them.
        for name in ["extract_features", "forward_features", "encode_image", "get_features"]:
            fn = getattr(model, name, None)
            if callable(fn):
                out = fn(images)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out
        return extract_by_hook(model, images)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, CUDA_VISIBLE_DEVICES={args.cuda}")

    transform = default_retizero_transform(args.image_size)
    dataset = HospitalFundusDataset(args.csv, args.image_root, args.diseases, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    model = RetiZeroClassifier(
        retizero_repo=args.retizero_repo,
        weight_path=args.weight_path,
        num_classes=args.num_classes,
        vision_type=args.vision_type,
        module_name=args.module_name,
        freeze_encoder=args.freeze_encoder,
        r=args.r,
    ).to(device)
    print(f"Feature dim reported by model: {getattr(model, 'feature_dim', 'unknown')}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = strip_module_prefix(get_state_dict(ckpt))
        msg = model.load_state_dict(state, strict=False)
        print(f"Loaded finetuned checkpoint: {args.checkpoint}; missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    else:
        print("No finetuned checkpoint provided. Using base RetiZero encoder from --weight_path.")

    model.eval()
    feats_all = []
    meta = {k: [] for k in ["image_path", "patient_id", "eye", "equipment", "disease", "shooting_date"]}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract {args.run_name}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                feat = extract_features(model, images)
            feats_all.append(feat.float().cpu().numpy())
            for k in meta:
                meta[k].extend(batch[k])

    features = np.concatenate(feats_all, axis=0).astype("float32")
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
