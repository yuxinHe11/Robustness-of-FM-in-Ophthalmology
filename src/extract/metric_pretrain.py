from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_DIFFERENT_DEVICE_DIR = Path("")
DEFAULT_SAME_DEVICE_DIR = Path("")

PATIENT_COLUMNS = ("patient_id", "patientID", "patient")
EYE_COLUMNS = ("eye", "laterality", "side")
DISEASE_COLUMNS = ("disease", "label")
EQUIPMENT_COLUMNS = ("equipment", "device", "scanner")
IMAGE_COLUMNS = ("image_path", "path", "filename", "image")


@dataclass(frozen=True)
class EmbeddingRun:
    model: str
    run_name: str
    path: Path
    features: np.ndarray
    meta: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--same-device-dir",type=Path,default=DEFAULT_SAME_DEVICE_DIR,help="Root containing repeated fundus embedding runs.",)
    parser.add_argument("--different-device-dir",type=Path,default=DEFAULT_DIFFERENT_DEVICE_DIR, help="Root containing different-device embedding runs.",)
    parser.add_argument( "--output-dir",type=Path,default=None)
    parser.add_argument("--target-strategy",choices=("all_pairs", "first_pair"), default="all_pairs")
    parser.add_argument("--model",nargs="*",default=None,help="Optional canonical model names to evaluate, e.g. dinov3_vitb16 eyeclip.",)
    parser.add_argument("--chunk-size",type=int,default=8192,help="Reserved for compatibility; per-query ranking is memory-light by default.",)
    return parser.parse_args()


def to_strings(values: Iterable[object]) -> np.ndarray:
    result = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if pd.isna(value):
            value = ""
        result.append(str(value).strip())
    return np.asarray(result, dtype=str)


def first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def normalize_model_name(name: str) -> str:
    value = name.lower().strip()
    value = re.sub(r"[-\s]+", "_", value)
    for suffix in ("_base", "_amd", "_glaucoma", "_health"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if value == "flair_merged30m":
        return "flair"
    if value == "flair":
        return "flair"
    if value == "retfound_dinov2_sh":
        return "retfound_dinov2_shanghai"
    return value


def is_base_run(run_dir: Path) -> bool:
    name = run_dir.name.lower()
    if name.endswith("_base"):
        return True
    if name == "base":
        return True
    return False


def collect_repeated_runs(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"same-device repeated directory not found: {root}")
    paths = sorted(root.rglob("embeddings.npz"))
    if not paths:
        raise FileNotFoundError(f"no embeddings.npz files found below {root}")
    runs: dict[str, Path] = {}
    for path in paths:
        model = normalize_model_name(path.parent.name)
        runs.setdefault(model, path.parent)
    return runs


def collect_base_runs(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"different-device directory not found: {root}")
    paths = sorted(root.rglob("embeddings.npz"))
    if not paths:
        raise FileNotFoundError(f"no embeddings.npz files found below {root}")

    runs: dict[str, Path] = {}
    for path in paths:
        run_dir = path.parent
        if not is_base_run(run_dir):
            continue
        parent_name = run_dir.parent.name if run_dir.name.lower() == "base" else run_dir.name
        model = normalize_model_name(parent_name)
        runs.setdefault(model, run_dir)
    if not runs:
        raise FileNotFoundError(f"no base embedding runs found below {root}")
    return runs


def load_run(run_dir: Path, model: str, domain: str) -> EmbeddingRun:
    npz_path = run_dir / "embeddings.npz"
    meta_path = run_dir / "meta.csv"

    with np.load(npz_path, allow_pickle=True) as data:
        if "features" not in data:
            raise KeyError(f"{npz_path} missing 'features'; keys={list(data.files)}")
        features = np.asarray(data["features"], dtype=np.float32)
        npz_columns = {
            key: to_strings(data[key])
            for key in data.files
            if key != "features" and getattr(data[key], "ndim", 0) == 1
        }

    if features.ndim != 2:
        raise ValueError(f"{npz_path}: features must be 2-D, got {features.shape}")

    if meta_path.exists():
        meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    else:
        meta = pd.DataFrame(npz_columns)

    if len(meta) != len(features):
        raise ValueError(
            f"{run_dir}: metadata rows ({len(meta)}) != feature rows ({len(features)})"
        )

    for key, values in npz_columns.items():
        if key not in meta.columns and len(values) == len(meta):
            meta[key] = values

    meta = standardize_meta(meta, domain)
    return EmbeddingRun(
        model=model,
        run_name=run_dir.name,
        path=run_dir,
        features=features,
        meta=meta,
    )


def standardize_meta(meta: pd.DataFrame, domain: str) -> pd.DataFrame:
    output = meta.copy().reset_index(drop=True)
    patient_col = first_existing_column(output.columns, PATIENT_COLUMNS)
    eye_col = first_existing_column(output.columns, EYE_COLUMNS)
    disease_col = first_existing_column(output.columns, DISEASE_COLUMNS)
    equipment_col = first_existing_column(output.columns, EQUIPMENT_COLUMNS)
    image_col = first_existing_column(output.columns, IMAGE_COLUMNS)

    missing = []
    if patient_col is None:
        missing.append("patient_id/patientID")
    if eye_col is None:
        missing.append("eye")
    if disease_col is None:
        missing.append("disease/label")
    if missing:
        raise KeyError(f"metadata missing required columns: {missing}")

    output["_patient"] = to_strings(output[patient_col])
    output["_eye"] = np.char.upper(to_strings(output[eye_col]))
    output["_disease"] = to_strings(output[disease_col])
    output["_equipment"] = (
        to_strings(output[equipment_col]) if equipment_col is not None else ""
    )
    if image_col is not None:
        output["_image"] = to_strings(output[image_col])
    else:
        output["_image"] = np.asarray([f"{domain}_{i}" for i in range(len(output))])

    output["_domain"] = domain
    output["_same_eye_key"] = output["_patient"] + "||" + output["_eye"]
    return output


def l2_normalize(features: np.ndarray) -> np.ndarray:
    finite = np.isfinite(features).all(axis=1)
    if not finite.all():
        features = features.copy()
        features[~finite] = 0
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return features / norms


def iter_target_pairs(same_meta: pd.DataFrame, strategy: str):
    valid = same_meta[(same_meta["_patient"] != "") & (same_meta["_eye"] != "")]
    for same_eye_key, group in valid.groupby("_same_eye_key", sort=False):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        for query_idx in indices:
            targets = indices[indices != query_idx]
            if strategy == "first_pair":
                targets = targets[:1]
            for target_idx in targets:
                yield same_eye_key, int(query_idx), int(target_idx)


def evaluate_model(
    same_run: EmbeddingRun,
    different_run: EmbeddingRun,
    target_strategy: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    if same_run.features.shape[1] != different_run.features.shape[1]:
        raise ValueError(
            f"{same_run.model}: embedding dimensions differ: "
            f"{same_run.features.shape[1]} vs {different_run.features.shape[1]}"
        )

    same_meta = same_run.meta.copy()
    different_meta = different_run.meta.copy()
    same_meta["_source_row"] = np.arange(len(same_meta))
    different_meta["_source_row"] = np.arange(len(different_meta))

    pool_meta = pd.concat([same_meta, different_meta], ignore_index=True)
    pool_meta["_pool_index"] = np.arange(len(pool_meta))
    pool_meta["_sample_uid"] = (
        pool_meta["_domain"] + ":" + pool_meta["_source_row"].astype(str)
    )
    same_pool_indices = pool_meta.index[pool_meta["_domain"] == "same_device"].to_numpy()

    pool_features = l2_normalize(np.vstack([same_run.features, different_run.features]))

    detail_rows = []
    pair_count = 0
    top1_hits = 0
    top5_hits = 0
    cross_label_events = 0
    cross_label_evaluable = 0
    ranks = []
    candidate_counts = []

    for same_eye_key, query_same_idx, target_same_idx in iter_target_pairs(
        same_meta, target_strategy
    ):
        pair_count += 1
        query_pool_idx = int(same_pool_indices[query_same_idx])
        target_pool_idx = int(same_pool_indices[target_same_idx])

        query = pool_features[query_pool_idx]
        scores = pool_features @ query

        candidate_mask = np.ones(len(pool_meta), dtype=bool)
        candidate_mask[query_pool_idx] = False
        same_eye_mask = pool_meta["_same_eye_key"].to_numpy() == same_eye_key
        candidate_mask[same_eye_mask] = False
        candidate_mask[target_pool_idx] = True

        candidate_indices = np.flatnonzero(candidate_mask)
        candidate_scores = scores[candidate_indices]
        order = np.argsort(-candidate_scores, kind="mergesort")
        ranked_indices = candidate_indices[order]

        target_positions = np.flatnonzero(ranked_indices == target_pool_idx)
        if len(target_positions) != 1:
            raise RuntimeError("target image missing from candidate ranking")
        rank = int(target_positions[0]) + 1
        top1 = rank <= 1
        top5 = rank <= 5

        query_label = str(pool_meta.at[query_pool_idx, "_disease"]).strip()
        before_target = ranked_indices[: rank - 1]
        if query_label:
            before_labels = pool_meta.iloc[before_target]["_disease"].astype(str).str.strip()
            cross_label = bool(((before_labels != "") & (before_labels != query_label)).any())
            cross_label_evaluable += 1
            cross_label_events += int(cross_label)
        else:
            cross_label = np.nan

        top1_hits += int(top1)
        top5_hits += int(top5)
        ranks.append(rank)
        candidate_counts.append(len(candidate_indices))

        detail_rows.append(
            {
                "model": same_run.model,
                "query_image": pool_meta.at[query_pool_idx, "_image"],
                "target_image": pool_meta.at[target_pool_idx, "_image"],
                "patient": pool_meta.at[query_pool_idx, "_patient"],
                "eye": pool_meta.at[query_pool_idx, "_eye"],
                "disease": query_label,
                "query_equipment": pool_meta.at[query_pool_idx, "_equipment"],
                "target_equipment": pool_meta.at[target_pool_idx, "_equipment"],
                "target_rank": rank,
                "target_score": float(scores[target_pool_idx]),
                "top1_hit": top1,
                "top5_hit": top5,
                "cross_label_outranking": cross_label,
                "n_candidates": len(candidate_indices),
            }
        )

    if pair_count == 0:
        raise ValueError(f"{same_run.model}: no same-eye repeated target pairs found")

    top1_recall = top1_hits / pair_count
    top5_recall = top5_hits / pair_count
    cross_label_ratio = (
        cross_label_events / cross_label_evaluable
        if cross_label_evaluable
        else np.nan
    )
    robustness_score = (
        top1_recall + top5_recall + (1 - cross_label_ratio)
    ) / 3 if np.isfinite(cross_label_ratio) else np.nan

    summary = {
        "model": same_run.model,
        "same_device_run": same_run.run_name,
        "different_device_run": different_run.run_name,
        "n_same_device_images": len(same_run.meta),
        "n_different_device_images": len(different_run.meta),
        "n_target_pairs": pair_count,
        "mean_candidates": float(np.mean(candidate_counts)),
        "top1_retrieval_recall": top1_recall,
        "top5_retrieval_recall": top5_recall,
        "cross_label_outranking_ratio": cross_label_ratio,
        "median_target_rank": float(np.median(ranks)),
        "mean_target_rank": float(np.mean(ranks)),
        "mrr": float(np.mean(1 / np.asarray(ranks, dtype=np.float64))),
        "robustness_score": robustness_score,
        "same_device_path": str(same_run.path),
        "different_device_path": str(different_run.path),
    }
    return summary, pd.DataFrame(detail_rows)


def write_markdown_summary(summary: pd.DataFrame, output_path: Path) -> None:
    display = summary[
        [
            "model",
            "n_target_pairs",
            "top1_retrieval_recall",
            "top5_retrieval_recall",
            "cross_label_outranking_ratio",
            "median_target_rank",
            "mrr",
            "robustness_score",
        ]
    ].copy()
    for column in [
        "top1_retrieval_recall",
        "top5_retrieval_recall",
        "cross_label_outranking_ratio",
        "mrr",
        "robustness_score",
    ]:
        display[column] = display[column].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    display["median_target_rank"] = display["median_target_rank"].map(
        lambda x: f"{x:.1f}" if pd.notna(x) else ""
    )
    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    lines = [
        "| " + " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    same_dirs = collect_repeated_runs(args.same_device_dir)
    different_dirs = collect_base_runs(args.different_device_dir)
    requested_models = {normalize_model_name(name) for name in args.model} if args.model else None

    models = sorted(set(same_dirs) & set(different_dirs))
    if requested_models is not None:
        models = [model for model in models if model in requested_models]
    if not models:
        raise RuntimeError(
            "no matched models between same-device repeated runs and different-device base runs"
        )

    summary_rows = []
    detail_frames = []
    failed_rows = []

    print(f"Matched {len(models)} base models: {', '.join(models)}")
    for model in models:
        try:
            same_run = load_run(same_dirs[model], model=model, domain="same_device")
            different_run = load_run(
                different_dirs[model],
                model=model,
                domain="different_device",
            )
            summary, details = evaluate_model(
                same_run,
                different_run,
                target_strategy=args.target_strategy,
            )
            summary_rows.append(summary)
            detail_frames.append(details)
            print(
                f"{model}: Top1={summary['top1_retrieval_recall']:.4f}, "
                f"Top5={summary['top5_retrieval_recall']:.4f}, "
                f"Cross-label={summary['cross_label_outranking_ratio']:.4f}"
            )
        except Exception as error:
            print(f"FAILED {model}: {error!r}")
            failed_rows.append(
                {
                    "model": model,
                    "same_device_path": str(same_dirs.get(model, "")),
                    "different_device_path": str(different_dirs.get(model, "")),
                    "error": repr(error),
                }
            )

    if not summary_rows:
        raise RuntimeError("all matched models failed")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["robustness_score", "top5_retrieval_recall", "top1_retrieval_recall"],
        ascending=[False, False, False],
    )
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)
    write_markdown_summary(summary_df, args.output_dir / "summary.md")
    pd.concat(detail_frames, ignore_index=True).to_csv(
        args.output_dir / "pair_details.csv",
        index=False,
    )
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(args.output_dir / "failed_models.csv", index=False)

    print("\nCOMBINED RESULTS")
    print(
        summary_df[
            [
                "model",
                "n_target_pairs",
                "top1_retrieval_recall",
                "top5_retrieval_recall",
                "cross_label_outranking_ratio",
                "median_target_rank",
                "mrr",
                "robustness_score",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
