from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Extract MIRAGE embeddings")
    parser.add_argument("--cuda", type=str, default="2")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument(
        "--image_root",
        type=str,
        required=True,
        help="Used when the image column contains relative paths.",
    )
    parser.add_argument("--diseases", type=str, default="health,AMD")
    parser.add_argument("--image_size", type=int, default=512)

    parser.add_argument(
        "--mirage_repo",
        type=str,
        required=True,
        help="Path to the official j-morano/MIRAGE repository.",
    )
    parser.add_argument("--model_size", choices=["base", "large"], default="base")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Official MIRAGE pretrained weights.",
    )
    parser.add_argument(
        "--pool",
        choices=["global", "cls", "token_mix"],
        default="global",
    )
    parser.add_argument(
        "--mode",
        choices=["linear_probe", "finetune"],
        default="finetune",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Optional finetuned MIRAGE checkpoint, e.g. best.pt.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Classifier classes. Defaults to the number of labels in --csv.",
    )

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--run_name", type=str, default="mirage_hospital")
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def import_mirage_components():
    try:
        from src.datasets_mirage import build_mirage_transforms
        from src.model.models_mirage import MIRAGEClassifier

        return build_mirage_transforms, MIRAGEClassifier
    except ImportError:
        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        from datasets_mirage import build_mirage_transforms
        from model.models_mirage import MIRAGEClassifier

        return build_mirage_transforms, MIRAGEClassifier


def torch_load_compat(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in [
            "model",
            "state_dict",
            "model_state_dict",
            "module",
            "network",
            "net",
        ]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def normalize_checkpoint_key(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def load_model_checkpoint(model, checkpoint_path: str, device):
    if not checkpoint_path:
        return {"loaded": False, "checkpoint": ""}

    try:
        from src.utils import load_checkpoint

        load_checkpoint(checkpoint_path, model, device)
        return {"loaded": True, "checkpoint": checkpoint_path, "loader": "src.utils"}
    except Exception as src_error:
        checkpoint = torch_load_compat(checkpoint_path)
        state = get_state_dict(checkpoint)
        if not isinstance(state, dict):
            raise TypeError(
                f"Unsupported checkpoint state type: {type(state)}"
            ) from src_error

        normalized = {
            normalize_checkpoint_key(key): value
            for key, value in state.items()
        }
        message = model.load_state_dict(normalized, strict=True)
        return {
            "loaded": True,
            "checkpoint": checkpoint_path,
            "loader": "fallback_load_state_dict",
            "missing_keys": list(message.missing_keys),
            "unexpected_keys": list(message.unexpected_keys),
        }


def read_metadata(csv_path: str, image_root: str, diseases: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "disease" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "disease"})
    if "image" not in df.columns and "relative_path" in df.columns:
        df = df.rename(columns={"relative_path": "image"})
    if "image" not in df.columns and "filename" in df.columns:
        df = df.rename(columns={"filename": "image"})
    if "patient_id" not in df.columns and "patientID" in df.columns:
        df = df.rename(columns={"patientID": "patient_id"})
    if "disease" not in df.columns and "diagnosis_label" in df.columns:
        df = df.rename(columns={"diagnosis_label": "disease"})

    required = ["image", "patient_id", "eye", "equipment", "disease"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}; columns={list(df.columns)}"
        )

    keep = [value.strip() for value in diseases.split(",") if value.strip()]
    if keep:
        df = df[df["disease"].astype(str).isin(keep)].copy()

    if df.empty:
        raise ValueError(f"No rows matched --diseases={diseases!r}.")

    def resolve(path):
        path = str(path).strip()
        if os.path.isabs(path):
            return path
        return os.path.join(image_root, path) if image_root else path

    df["image_path"] = df["image"].map(resolve)
    for column in ["patient_id", "eye", "equipment", "disease", "shooting_date"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()
    if "shooting_date" not in df.columns:
        df["shooting_date"] = ""
    return df.reset_index(drop=True)


class MirageEmbeddingDataset:
    def __init__(self, metadata, transform):
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        import torch
        from PIL import Image

        row = self.metadata.iloc[index]
        path = row["image_path"]
        try:
            image = Image.open(path)
            image.load()
        except Exception as error:
            raise RuntimeError(f"Failed to read image: {path}") from error

        return {
            "image": self.transform(image),
            "index": torch.tensor(index, dtype=torch.long),
        }


def collate_batch(batch):
    import torch

    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "index": torch.stack([item["index"] for item in batch], dim=0),
    }


def find_head_module(model):
    if hasattr(model, "backbone") and hasattr(model.backbone, "head"):
        return model.backbone.head
    if hasattr(model, "head"):
        return model.head
    raise AttributeError("Could not find a MIRAGE classification head.")


def forward_with_head_input(model, images):
    holder = {}
    head = find_head_module(model)

    def capture_head_input(_module, inputs):
        features = inputs[0]
        if isinstance(features, (tuple, list)):
            features = features[0]
        holder["features"] = features

    handle = head.register_forward_pre_hook(capture_head_input)
    try:
        logits = model(images)
    finally:
        handle.remove()

    if "features" not in holder:
        raise RuntimeError("MIRAGE head was not called; cannot extract features.")
    features = holder["features"]
    if features.ndim > 2:
        features = features.flatten(start_dim=1)
    return features, logits


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    build_mirage_transforms, MIRAGEClassifier = import_mirage_components()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, CUDA_VISIBLE_DEVICES={args.cuda}")

    metadata = read_metadata(args.csv, args.image_root, args.diseases)
    num_classes = args.num_classes or metadata["disease"].nunique()

    dataset = MirageEmbeddingDataset(
        metadata=metadata,
        transform=build_mirage_transforms("test", image_size=args.image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    model = MIRAGEClassifier(
        mirage_repo=args.mirage_repo,
        weights=args.weights,
        model_size=args.model_size,
        num_classes=num_classes,
        mode=args.mode,
        input_size=args.image_size,
        pool=args.pool,
    ).to(device)
    load_report = load_model_checkpoint(model, args.checkpoint, device)
    model.eval()

    feature_batches = []
    index_batches = []
    raw_norm_sum = 0.0
    raw_norm_count = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Extract {args.run_name}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(
                enabled=args.amp and device.type == "cuda"
            ):
                features, _ = forward_with_head_input(model, images)
                if features.ndim != 2:
                    raise ValueError(
                        f"Expected [batch, dim] features, got {features.shape}"
                    )
                norms = features.float().norm(p=2, dim=1)
                raw_norm_sum += norms.sum().item()
                raw_norm_count += len(norms)
                if args.l2_normalize:
                    features = F.normalize(features, p=2, dim=1)
            feature_batches.append(features.float().cpu().numpy())
            index_batches.append(batch["index"].cpu().numpy())

    unordered_features = np.concatenate(feature_batches, axis=0).astype(np.float32)
    indices = np.concatenate(index_batches, axis=0)
    features = np.empty_like(unordered_features)
    features[indices] = unordered_features

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = ["image_path", "patient_id", "eye", "equipment", "disease", "shooting_date"]
    payload = {"features": features}
    for column in fields:
        payload[column] = np.asarray(metadata[column].astype(str), dtype=object)
    np.savez_compressed(output_dir / "embeddings.npz", **payload)
    metadata[fields].to_csv(output_dir / "meta.csv", index=False)

    config = vars(args).copy()
    config["num_classes"] = num_classes
    config["feature_definition"] = "input tensor to MIRAGE classification head"
    config["extra_l2_normalize"] = bool(args.l2_normalize)
    config["mean_raw_feature_l2_norm"] = raw_norm_sum / max(raw_norm_count, 1)
    config["checkpoint_load_report"] = load_report
    config["n_samples"] = len(metadata)
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print(f"Saved: {output_dir / 'embeddings.npz'}")
    print(f"Saved metadata: {output_dir / 'meta.csv'}")
    print(f"Feature shape: {features.shape}")
    print(f"Mean raw L2 norm: {config['mean_raw_feature_l2_norm']:.6f}")


if __name__ == "__main__":
    main()
