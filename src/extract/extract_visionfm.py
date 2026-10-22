import os
import csv
import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("Extract VisionFM embeddings")
    p.add_argument("--cuda", type=str, default="2")
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--diseases", type=str, default="health,AMD", help="Optional comma-separated subset, e.g. Health,AMD,Glaucoma")
    p.add_argument("--pretrained_weights", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True, help="Finetuned best.pt. Empty means base VisionFM.")
    p.add_argument("--checkpoint_key", type=str, default="teacher")
    p.add_argument("--arch", type=str, default="vit_base", choices=["vit_tiny", "vit_small", "vit_base", "vit_large"])
    p.add_argument("--input_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--modality", type=str, default="OCT", choices=["Fundus", "OCT"])
    p.add_argument("--n_last_blocks", type=int, default=4)
    p.add_argument("--avgpool_patchtokens", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--head_layers", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--num_labels", type=int, default=3)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--run_name", type=str, default="visionfm_base")
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
    from torchvision import transforms
    from tqdm import tqdm
    import src.visionfm as visionfm
    import src.visionfm.utils_visionfm as utils
    from src.visionfm.head import ClsHead

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
                with open(path, "rb") as f:
                    image = Image.open(f).convert("RGB")
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

    def build_eval_transform(input_size, modality):
        mean, std = utils.get_stats(modality)
        print(f"use {modality} mean={mean}, std={std}")
        return transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

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

    def build_model(device):
        model = visionfm.__dict__[args.arch](
            img_size=[args.input_size],
            patch_size=args.patch_size,
            num_classes=0,
            use_mean_pooling=(args.avgpool_patchtokens == 1),
        )
        embed_dim = model.embed_dim
        model.to(device)
        print(f"Model {args.arch}, patch_size={args.patch_size}, embed_dim={embed_dim}")
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        if args.avgpool_patchtokens == 2:
            head_dim = embed_dim * (args.n_last_blocks + 1)
        else:
            head_dim = embed_dim * args.n_last_blocks
        classifier = ClsHead(embed_dim=head_dim, num_classes=args.num_labels, layers=args.head_layers).to(device)
        return model, classifier, head_dim

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, CUDA_VISIBLE_DEVICES={args.cuda}")

    eval_tf = build_eval_transform(args.input_size, args.modality)
    dataset = HospitalFundusDataset(args.csv, args.image_root, args.diseases, eval_tf)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)

    model, classifier, head_dim = build_model(device)
    print(f"Feature dim from extraction setting: {head_dim}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "visionfm_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint lacks 'visionfm_state_dict': {args.checkpoint}")
        msg1 = model.load_state_dict(ckpt["visionfm_state_dict"], strict=False)
        print(f"Loaded finetuned VisionFM: {args.checkpoint}; missing={len(msg1.missing_keys)}, unexpected={len(msg1.unexpected_keys)}")
        if "classifier_state_dict" in ckpt:
            msg2 = classifier.load_state_dict(ckpt["classifier_state_dict"], strict=False)
            print(f"Loaded classifier head too; missing={len(msg2.missing_keys)}, unexpected={len(msg2.unexpected_keys)}")
    else:
        print("No finetuned checkpoint provided. Using base VisionFM backbone.")

    model.eval()
    classifier.eval()
    feats_all = []
    meta = {k: [] for k in ["image_path", "patient_id", "eye", "equipment", "disease", "shooting_date"]}

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract {args.run_name}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                feat = extract_features(model, images, args.n_last_blocks, args.avgpool_patchtokens)
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
