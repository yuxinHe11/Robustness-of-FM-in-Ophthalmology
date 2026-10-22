import os
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", type=str, default="2", help="CUDA_VISIBLE_DEVICES value, e.g. 0 or 1")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--diseases", type=str, default="health,AMD", help="Optional comma-separated subset, e.g. Health,AMD,Glaucoma")
    p.add_argument("--image_size", type=int, default=224)

    p.add_argument("--model_dir", type=str, required=True, help="DINOv3 model directory")
    p.add_argument("--checkpoint", type=str, required=True, help="Finetuned best.pt. Empty means base model.")
    p.add_argument("--num_classes", type=int, default=2, help="Must match the finetuned checkpoint head.")
    p.add_argument("--mode", type=str, default="finetune", choices=["finetune", "linear_probe"])
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--run_name", type=str, default="dinov3_vitb16", help="Subdirectory name under output_dir")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from tqdm import tqdm
    from src.model.models_dinov3 import DINOv3Classifier
    # from model.models_dinov3 import DINOv3Classifier
    from src.utils import ensure_dir, load_dinov3_checkpoint_flexible

    class HospitalFundusDataset(Dataset):
        def __init__(self, csv_path, image_root="", diseases="", image_size=224):
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
            self.transform = transforms.Compose([
                transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

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
            except Exception as e:
                raise RuntimeError(f"Failed to read image: {path}") from e
            return {
                "image": self.transform(image),
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, CUDA_VISIBLE_DEVICES={args.cuda}")

    dataset = HospitalFundusDataset(args.csv, args.image_root, args.diseases, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    model = DINOv3Classifier(
        model_dir=args.model_dir,
        num_classes=args.num_classes,
        mode=args.mode,
        dropout=args.dropout,
    ).to(device)

    if args.checkpoint:
        print(f"Loading finetuned checkpoint: {args.checkpoint}")
        load_dinov3_checkpoint_flexible(args.checkpoint, model, device)
    else:
        print("No checkpoint provided. Using base DINOv3 backbone.")

    model.eval()
    feats = []
    meta = {k: [] for k in ["image_path", "patient_id", "eye", "equipment", "disease", "shooting_date"]}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract {args.run_name}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                outputs = model.backbone(pixel_values=images)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    feat = outputs.pooler_output
                else:
                    feat = outputs.last_hidden_state[:, 0]
            feats.append(feat.float().cpu().numpy())
            for k in meta:
                meta[k].extend(batch[k])

    features = np.concatenate(feats, axis=0).astype("float32")
    out_dir = Path(args.output_dir) / args.run_name
    ensure_dir(str(out_dir))

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
