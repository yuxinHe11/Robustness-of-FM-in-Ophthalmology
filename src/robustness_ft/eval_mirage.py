from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate MIRAGE by hospital")
    parser.add_argument("--cuda", type=str, default="2")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--diseases", type=str, default="health,AMD")
    parser.add_argument("--class_order", type=str, default="")
    parser.add_argument("--image_size", type=int, default=512)

    parser.add_argument("--mirage_repo",type=str,required=True,help="Path to the official j-morano/MIRAGE repository.",)
    parser.add_argument("--model_size", choices=["base", "large"], default="base")
    parser.add_argument("--weights", type=str,required=True,)
    parser.add_argument("--pool",choices=["global", "cls", "token_mix"],default="global")
    parser.add_argument("--mode",choices=["linear_probe", "finetune"],default="linear_probe")
    parser.add_argument("--checkpoint",type=str,required=True,help="Finetuned MIRAGE checkpoint, e.g. best.pt.")
    parser.add_argument("--num_classes",type=int,default=2)
    parser.add_argument("--positive_label",type=str,default="AMD")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
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
        print("No checkpoint provided. Evaluating the initialized MIRAGE head.")
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


def infer_class_order(diseases: str, positive_label: str, class_order: str):
    if class_order:
        classes = [value.strip() for value in class_order.split(",") if value.strip()]
    else:
        labels = [value.strip() for value in diseases.split(",") if value.strip()]
        healthy = [value for value in labels if value.lower() in {"health", "healthy", "normal", "control"}]
        classes = healthy + sorted(value for value in labels if value not in healthy)
    if len(classes) != len(set(classes)) or positive_label not in classes:
        raise ValueError(f"Invalid class order or positive label: {classes}, {positive_label}")
    return classes, classes.index(positive_label)


class MirageEvalDataset:
    def __init__(self, metadata, label_map, transform):
        self.metadata = metadata.reset_index(drop=True)
        self.label_map = label_map
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
            "label": torch.tensor(self.label_map[row["disease"]], dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


def collate_batch(batch):
    import torch

    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "index": torch.stack([item["index"] for item in batch], dim=0),
    }


def safe_metric(default=float("nan")):
    def wrapper(function, *args, **kwargs):
        try:
            return float(function(*args, **kwargs))
        except Exception:
            return default

    return wrapper


def compute_classification_metrics(y_true, probs, positive_index: int):
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    safe = safe_metric()
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(axis=1)
    n_classes = probs.shape[1]

    metrics = {
        "n": int(len(y_true)),
        "accuracy": safe(accuracy_score, y_true, y_pred),
        "balanced_accuracy": safe(balanced_accuracy_score, y_true, y_pred),
        "macro_f1": safe(f1_score, y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": safe(
            f1_score, y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    if n_classes == 2:
        positive = int(positive_index)
        y_binary = (y_true == positive).astype(int)
        pred_binary = (y_pred == positive).astype(int)
        score = probs[:, positive]
        tn, fp, fn, tp = confusion_matrix(
            y_binary, pred_binary, labels=[0, 1]
        ).ravel()
        precision, recall, _, _ = precision_recall_fscore_support(
            y_binary,
            pred_binary,
            labels=[1],
            average="binary",
            zero_division=0,
        )
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        metrics.update(
            {
                "auroc": safe(roc_auc_score, y_binary, score),
                "auprc": safe(average_precision_score, y_binary, score),
                "positive_index": positive,
                "n_positive": int(y_binary.sum()),
                "n_negative": int(len(y_binary) - y_binary.sum()),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    else:
        present_classes = np.unique(y_true)
        if len(present_classes) > 1:
            metrics["auroc_ovr_macro"] = safe(
                roc_auc_score,
                y_true,
                probs,
                multi_class="ovr",
                average="macro",
                labels=list(range(n_classes)),
            )
        else:
            metrics["auroc_ovr_macro"] = float("nan")
    return metrics


def summarize(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return {
        "n": int(len(values)),
        "mean": float(values.mean()) if len(values) else np.nan,
        "std": float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else np.nan),
    }


def build_prediction_pairs(pred):
    import pandas as pd
    columns = ["disease", "patient_id", "eye", "equipment_i", "equipment_j", "equipment_pair",
               "image_i", "image_j", "prob_positive_i", "prob_positive_j",
               "probability_shift_signed", "probability_shift_abs", "pred_i", "pred_j",
               "prediction_consistent", "target_positive_i", "target_positive_j",
               "pred_positive_i", "pred_positive_j", "both_correct_binary"]
    rows = []
    for _, group in pred.groupby(["patient_id", "eye", "disease"], sort=False):
        if group["equipment"].nunique() < 2:
            continue
        indices = sorted(group.index, key=lambda i: (str(pred.at[i, "equipment"]), str(pred.at[i, "image_path"])))
        for i, j in combinations(indices, 2):
            if pred.at[i, "equipment"] == pred.at[j, "equipment"]:
                continue
            pi, pj = float(pred.at[i, "prob_positive"]), float(pred.at[j, "prob_positive"])
            yi, yj = int(pred.at[i, "target_positive"]), int(pred.at[j, "target_positive"])
            qi = int(pred.at[i, "pred_label"] == pred.at[i, "positive_label"])
            qj = int(pred.at[j, "pred_label"] == pred.at[j, "positive_label"])
            rows.append({
                "disease": pred.at[i, "disease"], "patient_id": pred.at[i, "patient_id"], "eye": pred.at[i, "eye"],
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
    return pd.DataFrame(rows, columns=columns)


def evaluate_prediction_robustness(pred):
    import pandas as pd
    detail = build_prediction_pairs(pred)
    metrics = ["probability_shift_signed", "probability_shift_abs", "prediction_consistency"]
    source = {"prediction_consistency": "prediction_consistent"}
    summary = []
    for disease in sorted(pred["disease"].unique().tolist()) + ["ALL"]:
        group = detail if disease == "ALL" else detail[detail["disease"] == disease]
        for metric in metrics:
            summary.append({"metric": metric, "disease": disease, **summarize(group[source.get(metric, metric)])})
    by_pair = []
    for (disease, pair), group in detail.groupby(["disease", "equipment_pair"]):
        for metric in metrics:
            by_pair.append({"disease": disease, "equipment_pair": pair, "metric": metric,
                            **summarize(group[source.get(metric, metric)])})
    return pd.DataFrame(summary), detail, pd.DataFrame(by_pair,
        columns=["disease", "equipment_pair", "metric", "n", "mean", "std"])


def main() -> None:
    args = parse_args()
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for classifier evaluation.")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from src.extract.extract_mirage import read_metadata

    build_mirage_transforms, MIRAGEClassifier = import_mirage_components()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, CUDA_VISIBLE_DEVICES={args.cuda}")

    classes, positive_index = infer_class_order(args.diseases, args.positive_label, args.class_order)
    num_classes = args.num_classes or len(classes)
    if num_classes != len(classes):
        raise ValueError(f"num_classes={num_classes}, but class_order has {len(classes)} classes: {classes}")
    metadata = read_metadata(args.csv, args.image_root, args.diseases)
    unknown = set(metadata["disease"]) - set(classes)
    if unknown:
        raise ValueError(f"CSV has diseases absent from --class_order: {sorted(unknown)}")
    label_map = {label: index for index, label in enumerate(classes)}
    index_to_label = dict(enumerate(classes))

    dataset = MirageEvalDataset(
        metadata=metadata,
        label_map=label_map,
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

    logits_chunks = []
    label_chunks = []
    index_chunks = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Eval {args.run_name}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(
                enabled=args.amp and device.type == "cuda"
            ):
                logits = model(images)
            logits_chunks.append(logits.float().cpu().numpy())
            label_chunks.append(batch["label"].cpu().numpy())
            index_chunks.append(batch["index"].cpu().numpy())

    logits_unordered = np.concatenate(logits_chunks, axis=0)
    labels_unordered = np.concatenate(label_chunks, axis=0)
    indices = np.concatenate(index_chunks, axis=0)
    logits = np.empty_like(logits_unordered)
    labels = np.empty_like(labels_unordered)
    logits[indices] = logits_unordered
    labels[indices] = labels_unordered

    logits_for_softmax = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_for_softmax)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    prediction_df = metadata.copy()
    prediction_df["label_index"] = labels
    prediction_df["label_name"] = [index_to_label[int(value)] for value in labels]
    prediction_df["pred_index"] = preds
    prediction_df["pred_label"] = [index_to_label.get(int(value), str(value)) for value in preds]
    for index in range(logits.shape[1]):
        label_name = index_to_label.get(index, str(index))
        prediction_df[f"logit_{label_name}"] = logits[:, index]
        prediction_df[f"prob_{label_name}"] = probs[:, index]
    if logits.shape[1] == 2:
        prediction_df["prob_positive"] = probs[:, positive_index]
        prediction_df["target_positive"] = (labels == positive_index).astype(int)

    prediction_df["positive_label"] = classes[positive_index]
    prediction_df["prob_positive"] = probs[:, positive_index]
    prediction_df["target_positive"] = (labels == positive_index).astype(int)

    classification = pd.DataFrame([compute_classification_metrics(labels, probs, positive_index)])
    classification = classification.rename(columns={"accuracy": "acc", "balanced_accuracy": "bacc"})
    classification.insert(0, "run_name", args.run_name)
    classification["positive_label"] = classes[positive_index]
    classification["negative_label"] = classes[1 - positive_index] if len(classes) == 2 else ""
    summary, detail, by_equipment = evaluate_prediction_robustness(prediction_df)
    summary.insert(0, "run_name", args.run_name)

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_dir / "predictions.csv", index=False)
    classification.to_csv(output_dir / "classification_metrics.csv", index=False)
    summary.to_csv(output_dir / "prediction_robustness_summary.csv", index=False)
    detail.to_csv(output_dir / "prediction_robustness_pair_detail.csv", index=False)
    by_equipment.to_csv(output_dir / "prediction_robustness_by_equipment_pair.csv", index=False)

    config = vars(args).copy()
    config["num_classes"] = num_classes
    config["label_map"] = {str(key): int(value) for key, value in label_map.items()}
    config["index_to_label"] = index_to_label
    config["positive_index"] = int(positive_index)
    config["positive_label"] = index_to_label.get(int(positive_index), "")
    config["checkpoint_load_report"] = load_report
    config["n_samples"] = len(metadata)
    config["class_order_final"] = classes
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print("\nClassification metrics:")
    print(classification.to_string(index=False))
    print("\nPrediction robustness summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {output_dir}")


if __name__ == "__main__":
    main()
