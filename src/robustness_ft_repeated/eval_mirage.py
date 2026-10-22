from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Repeated OCT metadata.csv")
    p.add_argument("--image_root", required=True, help="Root prepended to relative_path; auto-detected from CSV when omitted")
    p.add_argument("--diseases", default="health,AMD")
    p.add_argument("--positive_label", default="AMD")
    p.add_argument("--class_order", default="health,AMD", help="Checkpoint class order; default: sorted disease labels")
    p.add_argument("--mirage_repo", required=True)
    p.add_argument("--model_size", choices=["base", "large"], default="large")
    p.add_argument("--weights", required=True)
    p.add_argument("--pool", choices=["global", "cls", "token_mix"], default="global")
    p.add_argument("--mode", choices=["linear_probe", "finetune"], default="linear_probe")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_classes", type=int, default=2)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cuda", default="2")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--run_name", default="amd")
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def read_metadata(csv_path, image_root, diseases):
    import pandas as pd

    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"patient_id": str, "pair_id": str})
    required = ["disease", "image", "pair_id", "patient_id", "eye", "equipment", "shooting_date"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Repeated OCT CSV missing columns: {missing}")
    df = df.rename(columns={"patient_id": "patient_id", "disease": "disease"})
    selected = [label.strip() for label in diseases.split(",") if label.strip()]
    if selected:
        df = df[df["disease"].astype(str).isin(selected)].copy()
    if df.empty:
        raise ValueError(f"No rows matched --diseases={diseases!r}")
    for name in ["image", "pair_id", "patient_id", "eye", "equipment", "shooting_date", "disease"]:
        if df[name].isna().any() or df[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing values in required column: {name}")
        df[name] = df[name].astype(str).str.strip()
    paths = [Path(value.replace("\\", "/")) for value in df["image"]]
    if image_root:
        root = Path(image_root).expanduser().resolve()
    else:
        csv_dir = Path(csv_path).resolve().parent
        sample = next((path for path in paths if not path.is_absolute()), None)
        root = next((candidate for candidate in (csv_dir, *csv_dir.parents)
                     if sample is not None and (candidate / sample).is_file()), None)
        if sample is not None and root is None:
            raise FileNotFoundError(
                f"Cannot resolve image={str(sample)!r} from CSV location; pass --image_root"
            )
    df["image_path"] = [str(path if path.is_absolute() else root / path) for path in paths]
    missing_paths = [path for path in df["image_path"] if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"{len(missing_paths)} OCT images not found; examples: {missing_paths[:3]}")
    sizes = df.groupby("pair_id").size()
    invalid = sizes[sizes != 2]
    if not invalid.empty:
        raise ValueError(f"Each pair_id needs two images; examples: {invalid.head().to_dict()}")
    if df["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths in repeated OCT CSV")
    return df.reset_index(drop=True)


def import_mirage():
    try:
        from src.datasets_mirage import build_mirage_transforms
        from src.model.models_mirage import MIRAGEClassifier
    except ModuleNotFoundError as exc:
        if exc.name != "src":
            raise
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from datasets_mirage import build_mirage_transforms
        from model.models_mirage import MIRAGEClassifier
    return build_mirage_transforms, MIRAGEClassifier


def load_checkpoint(model, path):
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dict")
    normalized = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.", "model.")):
            key = key.split(".", 1)[1]
        normalized[key] = value
    model.load_state_dict(normalized, strict=True)


def make_loader(metadata, transform, args, pin_memory):
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    class OCTDataset(Dataset):
        def __len__(self):
            return len(metadata)

        def __getitem__(self, index):
            path = metadata.at[index, "image_path"]
            try:
                with Image.open(path) as image:
                    tensor = transform(image)
            except Exception as exc:
                raise RuntimeError(f"Failed to read image: {path}") from exc
            return tensor, index

    return DataLoader(OCTDataset(), batch_size=args.batch_size, shuffle=False,
                      num_workers=args.num_workers, pin_memory=pin_memory)


def attach_predictions(metadata, logits, classes, positive_index):
    import numpy as np

    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    pred_index = probs.argmax(axis=1)
    out = metadata.copy()
    for index, label in enumerate(classes):
        out[f"logit_{label}"] = logits[:, index]
        out[f"prob_{label}"] = probs[:, index]
    out["pred_index"] = pred_index
    out["pred_label"] = [classes[index] for index in pred_index]
    out["positive_label"] = classes[positive_index]
    out["prob_positive"] = probs[:, positive_index]
    out["target_positive"] = (out["disease"] == classes[positive_index]).astype(int)
    return out


def summarize(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return {"n": int(len(values)),
            "mean": float(values.mean()) if len(values) else np.nan,
            "std": float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else np.nan)}


def build_repeat_pairs(pred):
    import pandas as pd

    rows = []
    for pair_id, group in pred.groupby("pair_id", sort=False):
        if len(group) != 2:
            raise ValueError(f"pair_id={pair_id!r} has {len(group)} images; expected two")
        for field in ("patient_id", "eye", "disease", "equipment", "shooting_date"):
            if group[field].nunique(dropna=False) != 1:
                raise ValueError(f"pair_id={pair_id!r} has inconsistent {field}")
        i, j = sorted(group.index, key=lambda index: str(pred.at[index, "image_path"]))
        pi, pj = float(pred.at[i, "prob_positive"]), float(pred.at[j, "prob_positive"])
        yi, yj = int(pred.at[i, "target_positive"]), int(pred.at[j, "target_positive"])
        qi = int(pred.at[i, "pred_label"] == pred.at[i, "positive_label"])
        qj = int(pred.at[j, "pred_label"] == pred.at[j, "positive_label"])
        rows.append({
            "pair_id": pair_id, "disease": pred.at[i, "disease"],
            "patient_id": pred.at[i, "patient_id"], "eye": pred.at[i, "eye"],
            "shooting_date": pred.at[i, "shooting_date"],
            "equipment_i": pred.at[i, "equipment"], "equipment_j": pred.at[j, "equipment"],
            "equipment_pair": f"{pred.at[i, 'equipment']} -> {pred.at[j, 'equipment']}",
            "image_i": pred.at[i, "image_path"], "image_j": pred.at[j, "image_path"],
            "prob_positive_i": pi, "prob_positive_j": pj,
            "probability_shift_signed": pj - pi, "probability_shift_abs": abs(pj - pi),
            "pred_i": pred.at[i, "pred_label"], "pred_j": pred.at[j, "pred_label"],
            "prediction_consistent": float(pred.at[i, "pred_label"] == pred.at[j, "pred_label"]),
            "target_positive_i": yi, "target_positive_j": yj,
            "pred_positive_i": qi, "pred_positive_j": qj,
            "both_correct_binary": float(yi == qi and yj == qj),
        })
    return pd.DataFrame(rows)


def classification_metrics(pred, classes, positive_index):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, roc_auc_score)

    y_true = pred["target_positive"].to_numpy(dtype=int)
    y_pred = (pred["pred_index"].to_numpy(dtype=int) == positive_index).astype(int)
    y_prob = pred["prob_positive"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": len(y_true), "n_positive": int(y_true.sum()), "n_negative": int(len(y_true) - y_true.sum()),
        "positive_label": classes[positive_index], "negative_label": classes[1 - positive_index],
        "auroc": np.nan, "auprc": np.nan,
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "acc": accuracy_score(y_true, y_pred), "bacc": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        out["auroc"] = roc_auc_score(y_true, y_prob)
        out["auprc"] = average_precision_score(y_true, y_prob)
    return pd.DataFrame([out])


def robustness_metrics(pred):
    import pandas as pd

    detail = build_repeat_pairs(pred)
    metrics = {"probability_shift_abs": "probability_shift_abs",
               "prediction_consistency": "prediction_consistent",
               "probability_shift_signed": "probability_shift_signed"}
    summary, by_equipment = [], []
    for disease in sorted(pred["disease"].unique().tolist()) + ["ALL"]:
        group = detail if disease == "ALL" else detail[detail["disease"] == disease]
        for metric, column in metrics.items():
            summary.append({"metric": metric, "disease": disease, **summarize(group[column])})
    for (disease, pair), group in detail.groupby(["disease", "equipment_pair"]):
        for metric, column in metrics.items():
            by_equipment.append({"disease": disease, "equipment_pair": pair,
                                 "metric": metric, **summarize(group[column])})
    return pd.DataFrame(summary), detail, pd.DataFrame(by_equipment)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    import numpy as np
    import torch
    from tqdm import tqdm

    build_mirage_transforms, MIRAGEClassifier = import_mirage()
    classes = ([label.strip() for label in args.class_order.split(",") if label.strip()]
               if args.class_order else sorted(label.strip() for label in args.diseases.split(",") if label.strip()))
    if len(classes) != 2 or len(set(classes)) != 2 or args.num_classes != 2:
        raise ValueError("Repeated-image evaluation requires two distinct classes and --num_classes=2")
    if args.positive_label not in classes:
        raise ValueError(f"--positive_label={args.positive_label!r} is absent from {classes}")
    positive_index = classes.index(args.positive_label)
    metadata = read_metadata(args.csv, args.image_root, args.diseases)
    unknown = set(metadata["disease"]) - set(classes)
    if unknown:
        raise ValueError(f"CSV labels absent from class order: {sorted(unknown)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    transform = build_mirage_transforms("test", image_size=args.image_size)
    loader = make_loader(metadata, transform, args, device.type == "cuda")
    model = MIRAGEClassifier(
        mirage_repo=args.mirage_repo, weights=args.weights, model_size=args.model_size,
        num_classes=args.num_classes, mode=args.mode, input_size=args.image_size, pool=args.pool,
    ).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    logits = np.empty((len(metadata), len(classes)), dtype=np.float32)
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc=f"Infer {args.run_name}"):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                output = model(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            logits[indices.numpy()] = output.float().cpu().numpy()

    pred = attach_predictions(metadata, logits, classes, positive_index)
    classification = classification_metrics(pred, classes, positive_index)
    classification.insert(0, "run_name", args.run_name)
    summary, detail, by_equipment = robustness_metrics(pred)
    summary.insert(0, "run_name", args.run_name)
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output_dir / "predictions.csv", index=False)
    classification.to_csv(output_dir / "classification_metrics.csv", index=False)
    summary.to_csv(output_dir / "prediction_robustness_summary.csv", index=False)
    detail.to_csv(output_dir / "prediction_robustness_pair_detail.csv", index=False)
    by_equipment.to_csv(output_dir / "prediction_robustness_by_equipment_pair.csv", index=False)
    config = vars(args).copy()
    config.update({"class_order_final": classes, "positive_index": positive_index,
                   "n_images": len(metadata), "n_pairs": len(detail)})
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
    print(classification.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
