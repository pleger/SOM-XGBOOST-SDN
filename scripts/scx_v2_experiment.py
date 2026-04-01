#!/usr/bin/env python3
"""Run baseline XGBoost and SCX v2 experiments on CICIDS2017 Friday CSV files."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


FRIDAY_FILES = [
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


@dataclass
class SplitData:
    x_seed: np.ndarray
    y_seed: np.ndarray
    x_stream: np.ndarray
    y_stream: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SCX v2 experiment runner")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/cicids2017/friday"),
        help="Directory containing CICIDS2017 Friday CSVs",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "scx_v2", "both"],
        default="both",
        help="Which model(s) to run",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated integer seeds",
    )
    parser.add_argument("--seed-ratio", type=float, default=0.10)
    parser.add_argument("--stream-ratio", type=float, default=0.65)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument(
        "--split-strategy",
        choices=["global_time", "per_file_time", "random_stratified"],
        default="per_file_time",
        help="How to split seed/stream/test",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-tune",
        choices=["none", "accuracy", "f1", "recall_at_fpr"],
        default="none",
        help="Tune decision threshold on calibration slice from stream labels",
    )
    parser.add_argument(
        "--fpr-ceiling",
        type=float,
        default=0.0005,
        help="Maximum allowed FPR for recall_at_fpr tuning (e.g., 0.0005 = 0.05 pct)",
    )
    parser.add_argument(
        "--calibration-ratio",
        type=float,
        default=0.10,
        help="Fraction of stream held out for threshold calibration",
    )
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=199,
        help="Number of threshold candidates in [0.001, 0.999]",
    )
    parser.add_argument(
        "--keep-port-features",
        action="store_true",
        help="Keep port-related features (may inflate benchmark performance)",
    )
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=3.0)
    parser.add_argument("--reg-lambda", type=float, default=2.0)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--som-side", type=int, default=25)
    parser.add_argument(
        "--som-iterations",
        type=int,
        default=20000,
        help="MiniSom random training iterations",
    )
    parser.add_argument("--som-sigma", type=float, default=3.0)
    parser.add_argument("--som-learning-rate", type=float, default=0.5)
    parser.add_argument(
        "--som-max-train-samples",
        type=int,
        default=50000,
        help="Cap seed samples used to train SOM (for speed)",
    )
    parser.add_argument("--min-neuron-support", type=int, default=30)
    parser.add_argument("--attack-conf-threshold", type=float, default=0.92)
    parser.add_argument("--benign-conf-threshold", type=float, default=0.97)
    parser.add_argument("--buffer-cap", type=int, default=250000)
    parser.add_argument("--retrain-trigger", type=int, default=25000)
    parser.add_argument("--stream-batch-size", type=int, default=25000)
    parser.add_argument(
        "--neg-pos-max-ratio",
        type=float,
        default=2.0,
        help="Max negatives kept per positive during retraining",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on loaded rows (0 means full dataset)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/scx_v2_results.json"),
        help="Where to write JSON results",
    )
    return parser.parse_args()


def load_dataset(
    data_dir: Path,
    max_rows: int = 0,
    keep_port_features: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[str], list[int]]:
    dfs: list[pd.DataFrame] = []
    file_lengths: list[int] = []
    for filename in FRIDAY_FILES:
        csv_path = data_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {csv_path}")
        print(f"[load] {csv_path}")
        file_df = pd.read_csv(csv_path, low_memory=False)
        dfs.append(file_df)
        file_lengths.append(len(file_df))

    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    labels_raw = df["Label"].astype(str).str.strip()
    label_counts = labels_raw.value_counts().sort_index().to_dict()
    y = (labels_raw.str.upper() != "BENIGN").astype(np.int8).to_numpy()

    x_df = df.drop(columns=["Label"]).copy()
    if not keep_port_features:
        drop_candidates = {"destination port", "source port", "src port", "dst port"}
        drop_cols = [c for c in x_df.columns if c.strip().lower() in drop_candidates]
        if drop_cols:
            print(f"[prep] dropping endpoint identifiers: {drop_cols}")
            x_df = x_df.drop(columns=drop_cols)
    else:
        print("[prep] keeping port-related features")

    for col in x_df.columns:
        x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
    x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if max_rows > 0 and len(x_df) > max_rows:
        x_df = x_df.iloc[:max_rows].copy()
        y = y[:max_rows]
        print(f"[load] capped rows to {max_rows}")
        running = 0
        capped_lengths: list[int] = []
        for length in file_lengths:
            if running >= max_rows:
                break
            keep = min(length, max_rows - running)
            capped_lengths.append(keep)
            running += keep
        file_lengths = capped_lengths

    x = x_df.to_numpy(dtype=np.float32)
    feature_names = x_df.columns.tolist()
    return x, y, label_counts, feature_names, file_lengths


def split_time_order(
    x: np.ndarray,
    y: np.ndarray,
    seed_ratio: float,
    stream_ratio: float,
    test_ratio: float,
) -> SplitData:
    ratio_sum = seed_ratio + stream_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    n = len(x)
    n_seed = int(n * seed_ratio)
    n_stream = int(n * stream_ratio)
    n_test = n - n_seed - n_stream

    if min(n_seed, n_stream, n_test) <= 0:
        raise ValueError("One split is empty. Adjust ratios.")

    x_seed = x[:n_seed]
    y_seed = y[:n_seed]
    x_stream = x[n_seed : n_seed + n_stream]
    y_stream = y[n_seed : n_seed + n_stream]
    x_test = x[n_seed + n_stream :]
    y_test = y[n_seed + n_stream :]

    assert len(x_test) == n_test
    return SplitData(x_seed, y_seed, x_stream, y_stream, x_test, y_test)


def split_per_file_time(
    x: np.ndarray,
    y: np.ndarray,
    file_lengths: list[int],
    seed_ratio: float,
    stream_ratio: float,
    test_ratio: float,
) -> SplitData:
    ratio_sum = seed_ratio + stream_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    seed_parts_x: list[np.ndarray] = []
    seed_parts_y: list[np.ndarray] = []
    stream_parts_x: list[np.ndarray] = []
    stream_parts_y: list[np.ndarray] = []
    test_parts_x: list[np.ndarray] = []
    test_parts_y: list[np.ndarray] = []

    start = 0
    for length in file_lengths:
        end = start + length
        x_f = x[start:end]
        y_f = y[start:end]
        n_f = len(x_f)

        n_seed = int(n_f * seed_ratio)
        n_stream = int(n_f * stream_ratio)
        n_test = n_f - n_seed - n_stream
        if min(n_seed, n_stream, n_test) <= 0:
            raise ValueError("One split is empty in per_file_time mode. Adjust ratios.")

        seed_parts_x.append(x_f[:n_seed])
        seed_parts_y.append(y_f[:n_seed])
        stream_parts_x.append(x_f[n_seed : n_seed + n_stream])
        stream_parts_y.append(y_f[n_seed : n_seed + n_stream])
        test_parts_x.append(x_f[n_seed + n_stream :])
        test_parts_y.append(y_f[n_seed + n_stream :])
        start = end

    return SplitData(
        x_seed=np.concatenate(seed_parts_x, axis=0),
        y_seed=np.concatenate(seed_parts_y, axis=0),
        x_stream=np.concatenate(stream_parts_x, axis=0),
        y_stream=np.concatenate(stream_parts_y, axis=0),
        x_test=np.concatenate(test_parts_x, axis=0),
        y_test=np.concatenate(test_parts_y, axis=0),
    )


def split_random_stratified(
    x: np.ndarray,
    y: np.ndarray,
    seed_ratio: float,
    stream_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> SplitData:
    ratio_sum = seed_ratio + stream_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    x_train_stream, x_test, y_train_stream, y_test = train_test_split(
        x, y, test_size=test_ratio, random_state=random_seed, stratify=y
    )
    seed_share_within_train_stream = seed_ratio / (seed_ratio + stream_ratio)
    x_seed, x_stream, y_seed, y_stream = train_test_split(
        x_train_stream,
        y_train_stream,
        test_size=(1.0 - seed_share_within_train_stream),
        random_state=random_seed,
        stratify=y_train_stream,
    )
    return SplitData(x_seed, y_seed, x_stream, y_stream, x_test, y_test)


def build_xgb(args: argparse.Namespace, random_seed: int, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        n_jobs=args.n_jobs,
        tree_method="hist",
    )


def fit_xgb(args: argparse.Namespace, random_seed: int, x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    model = build_xgb(args, random_seed=random_seed, scale_pos_weight=scale_pos_weight)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: XGBClassifier, x: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics: dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y, prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y, prob))
    except ValueError:
        metrics["pr_auc"] = float("nan")
    return metrics


def tune_threshold(
    model: XGBClassifier,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    metric: str,
    grid_size: int,
    default_threshold: float,
    fpr_ceiling: float,
) -> float:
    if metric == "none" or len(y_cal) == 0:
        return float(default_threshold)

    prob = model.predict_proba(x_cal)[:, 1]
    thresholds = np.linspace(0.001, 0.999, max(3, grid_size))
    best_t = float(default_threshold)
    best_score = -1.0

    for t in thresholds:
        pred = (prob >= t).astype(np.int8)
        if metric == "accuracy":
            score = accuracy_score(y_cal, pred)
        elif metric == "f1":
            score = f1_score(y_cal, pred, zero_division=0)
        else:
            tn, fp, fn, tp = confusion_matrix(y_cal, pred, labels=[0, 1]).ravel()
            fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            if fpr <= fpr_ceiling:
                score = 10_000.0 + recall
            else:
                score = -fpr

        # Tie-break towards thresholds close to 0.5 for stability.
        if (score > best_score) or (np.isclose(score, best_score) and abs(t - 0.5) < abs(best_t - 0.5)):
            best_score = float(score)
            best_t = float(t)

    return best_t


def train_som(
    x_seed: np.ndarray,
    som_side: int,
    som_iterations: int,
    sigma: float,
    learning_rate: float,
    max_train_samples: int,
    seed: int,
) -> MiniSom:
    rng = np.random.default_rng(seed)
    if len(x_seed) > max_train_samples:
        idx = rng.choice(len(x_seed), size=max_train_samples, replace=False)
        x_train = x_seed[idx]
    else:
        x_train = x_seed

    som = MiniSom(
        x=som_side,
        y=som_side,
        input_len=x_seed.shape[1],
        sigma=sigma,
        learning_rate=learning_rate,
        random_seed=seed,
        topology="rectangular",
    )
    som.random_weights_init(x_train)
    som.train_random(x_train, som_iterations, verbose=False)
    return som


def summarize_neurons(
    weights_flat: np.ndarray,
    x_seed: np.ndarray,
    y_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn.fit(weights_flat)
    distances, bmu_idx = nn.kneighbors(x_seed, return_distance=True)
    qe_seed = distances[:, 0]
    bmu_idx = bmu_idx[:, 0]

    n_nodes = len(weights_flat)
    support = np.zeros(n_nodes, dtype=np.int32)
    attack_count = np.zeros(n_nodes, dtype=np.int32)
    qe_lists: list[list[float]] = [[] for _ in range(n_nodes)]

    for i in range(len(x_seed)):
        idx = int(bmu_idx[i])
        support[idx] += 1
        attack_count[idx] += int(y_seed[i] == 1)
        qe_lists[idx].append(float(qe_seed[i]))

    global_attack_rate = float(np.mean(y_seed))
    global_med_qe = float(np.median(qe_seed))
    global_p90_qe = float(np.percentile(qe_seed, 90))

    p_attack = np.full(n_nodes, global_attack_rate, dtype=np.float32)
    med_qe = np.full(n_nodes, global_med_qe, dtype=np.float32)
    p90_qe = np.full(n_nodes, global_p90_qe, dtype=np.float32)

    nonzero = support > 0
    p_attack[nonzero] = attack_count[nonzero] / support[nonzero]

    for idx in np.where(nonzero)[0]:
        vals = np.array(qe_lists[idx], dtype=np.float32)
        med_qe[idx] = np.median(vals)
        p90_qe[idx] = np.percentile(vals, 90)

    return support, p_attack, med_qe, p90_qe


def class_balanced_indices(
    y: np.ndarray,
    rng: np.random.Generator,
    neg_pos_max_ratio: float,
) -> np.ndarray:
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)

    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return np.arange(len(y))

    neg_keep = min(len(idx_neg), int(len(idx_pos) * neg_pos_max_ratio))
    neg_sel = rng.choice(idx_neg, size=neg_keep, replace=False)
    keep = np.concatenate([idx_pos, neg_sel])
    rng.shuffle(keep)
    return keep


def run_baseline(
    args: argparse.Namespace,
    split: SplitData,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    start = time.time()
    model = fit_xgb(args, random_seed=seed, x_train=split.x_seed, y_train=split.y_seed)
    decision_threshold = tune_threshold(
        model,
        x_cal,
        y_cal,
        metric=args.threshold_tune,
        grid_size=args.threshold_grid_size,
        default_threshold=args.threshold,
        fpr_ceiling=args.fpr_ceiling,
    )
    metrics = evaluate_model(model, split.x_test, split.y_test, decision_threshold)
    if len(y_cal) > 0 and args.threshold_tune != "none":
        cal_metrics = evaluate_model(model, x_cal, y_cal, decision_threshold)
        metrics["calibration_accuracy"] = cal_metrics["accuracy"]
        metrics["calibration_f1"] = cal_metrics["f1"]
    metrics["elapsed_sec"] = float(time.time() - start)
    return metrics


def run_scx_v2(
    args: argparse.Namespace,
    split: SplitData,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    run_start = time.time()
    rng = np.random.default_rng(seed)

    model = fit_xgb(args, random_seed=seed, x_train=split.x_seed, y_train=split.y_seed)

    som = train_som(
        x_seed=split.x_seed,
        som_side=args.som_side,
        som_iterations=args.som_iterations,
        sigma=args.som_sigma,
        learning_rate=args.som_learning_rate,
        max_train_samples=args.som_max_train_samples,
        seed=seed,
    )
    weights_flat = som.get_weights().reshape(-1, split.x_seed.shape[1]).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(weights_flat)
    support, p_attack, med_qe, p90_qe = summarize_neurons(weights_flat, split.x_seed, split.y_seed)

    buffer_x_parts: list[np.ndarray] = []
    buffer_y_parts: list[np.ndarray] = []
    buffer_count = 0
    accepted_since_retrain = 0
    total_accepted = 0
    retrain_rounds = 0

    n_stream = len(split.x_stream)
    for start in range(0, n_stream, args.stream_batch_size):
        end = min(start + args.stream_batch_size, n_stream)
        batch_x = split.x_stream[start:end]

        distances, bmu_idx = nn.kneighbors(batch_x, return_distance=True)
        qe = distances[:, 0]
        idx = bmu_idx[:, 0]

        p_a = p_attack[idx]
        pseudo = (p_a >= 0.5).astype(np.int8)
        conf_base = np.maximum(p_a, 1.0 - p_a)
        conf = conf_base * np.exp(-qe / (med_qe[idx] + 1e-9))
        conf_threshold = np.where(
            pseudo == 1,
            args.attack_conf_threshold,
            args.benign_conf_threshold,
        )
        accept = (
            (support[idx] >= args.min_neuron_support)
            & (qe <= p90_qe[idx])
            & (conf >= conf_threshold)
        )

        xgb_pred = (model.predict_proba(batch_x)[:, 1] >= args.threshold).astype(np.int8)
        accept &= xgb_pred == pseudo

        if np.any(accept):
            accepted_x = batch_x[accept]
            accepted_y = pseudo[accept]
            m = len(accepted_x)

            buffer_x_parts.append(accepted_x)
            buffer_y_parts.append(accepted_y)
            buffer_count += m
            total_accepted += m
            accepted_since_retrain += m

            if buffer_count > args.buffer_cap:
                all_x = np.concatenate(buffer_x_parts, axis=0)[-args.buffer_cap :]
                all_y = np.concatenate(buffer_y_parts, axis=0)[-args.buffer_cap :]
                buffer_x_parts = [all_x]
                buffer_y_parts = [all_y]
                buffer_count = len(all_y)

        if accepted_since_retrain >= args.retrain_trigger and buffer_count > 0:
            retrain_x = np.concatenate([split.x_seed] + buffer_x_parts, axis=0)
            retrain_y = np.concatenate([split.y_seed] + buffer_y_parts, axis=0)
            keep_idx = class_balanced_indices(retrain_y, rng, args.neg_pos_max_ratio)
            retrain_x = retrain_x[keep_idx]
            retrain_y = retrain_y[keep_idx]

            model = fit_xgb(args, random_seed=seed + retrain_rounds + 1, x_train=retrain_x, y_train=retrain_y)
            retrain_rounds += 1
            accepted_since_retrain = 0
            print(
                f"[scx] retrain #{retrain_rounds}: buffer={buffer_count}, "
                f"train_rows={len(retrain_y)}, stream_seen={end}/{n_stream}"
            )

    decision_threshold = tune_threshold(
        model,
        x_cal,
        y_cal,
        metric=args.threshold_tune,
        grid_size=args.threshold_grid_size,
        default_threshold=args.threshold,
        fpr_ceiling=args.fpr_ceiling,
    )
    metrics = evaluate_model(model, split.x_test, split.y_test, decision_threshold)
    if len(y_cal) > 0 and args.threshold_tune != "none":
        cal_metrics = evaluate_model(model, x_cal, y_cal, decision_threshold)
        metrics["calibration_accuracy"] = cal_metrics["accuracy"]
        metrics["calibration_f1"] = cal_metrics["f1"]
    metrics["elapsed_sec"] = float(time.time() - run_start)
    metrics["accepted_pseudo_labels"] = float(total_accepted)
    metrics["retrain_rounds"] = float(retrain_rounds)
    metrics["acceptance_rate_stream"] = float(total_accepted / len(split.x_stream))
    return metrics


def summarize_runs(all_runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for run in all_runs:
        by_mode.setdefault(run["mode"], []).append(run)

    summary: dict[str, Any] = {}
    for mode, runs in by_mode.items():
        metric_keys = [k for k in runs[0]["metrics"].keys() if isinstance(runs[0]["metrics"][k], float)]
        mode_summary: dict[str, float] = {}
        for key in metric_keys:
            values = np.array([float(r["metrics"][key]) for r in runs], dtype=np.float64)
            finite_values = values[np.isfinite(values)]
            if len(finite_values) == 0:
                mode_summary[f"{key}_mean"] = float("nan")
                mode_summary[f"{key}_std"] = float("nan")
            else:
                mode_summary[f"{key}_mean"] = float(np.mean(finite_values))
                mode_summary[f"{key}_std"] = float(np.std(finite_values))
        summary[mode] = mode_summary
    return summary


def serialize_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        config[key] = str(value) if isinstance(value, Path) else value
    return config


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")

    x, y, label_counts, feature_names, file_lengths = load_dataset(
        args.data_dir,
        max_rows=args.max_rows,
        keep_port_features=args.keep_port_features,
    )
    if args.split_strategy == "global_time":
        split = split_time_order(x, y, args.seed_ratio, args.stream_ratio, args.test_ratio)
    elif args.split_strategy == "per_file_time":
        split = split_per_file_time(x, y, file_lengths, args.seed_ratio, args.stream_ratio, args.test_ratio)
    else:
        split = split_random_stratified(
            x,
            y,
            args.seed_ratio,
            args.stream_ratio,
            args.test_ratio,
            random_seed=seeds[0],
        )

    scaler = StandardScaler()
    scaler.fit(split.x_seed)
    split = SplitData(
        x_seed=scaler.transform(split.x_seed).astype(np.float32),
        y_seed=split.y_seed,
        x_stream=scaler.transform(split.x_stream).astype(np.float32),
        y_stream=split.y_stream,
        x_test=scaler.transform(split.x_test).astype(np.float32),
        y_test=split.y_test,
    )

    n_cal = int(len(split.x_stream) * args.calibration_ratio)
    if n_cal > 0:
        x_cal = split.x_stream[:n_cal]
        y_cal = split.y_stream[:n_cal]
        split = SplitData(
            x_seed=split.x_seed,
            y_seed=split.y_seed,
            x_stream=split.x_stream[n_cal:],
            y_stream=split.y_stream[n_cal:],
            x_test=split.x_test,
            y_test=split.y_test,
        )
    else:
        x_cal = np.empty((0, split.x_seed.shape[1]), dtype=np.float32)
        y_cal = np.empty((0,), dtype=np.int8)

    run_records: list[dict[str, Any]] = []
    for seed in seeds:
        if args.mode in {"baseline", "both"}:
            print(f"[run] baseline seed={seed}")
            metrics = run_baseline(args, split, x_cal, y_cal, seed)
            run_records.append({"mode": "baseline", "seed": seed, "metrics": metrics})
            print(f"[result] baseline seed={seed} accuracy={metrics['accuracy']:.6f} f1={metrics['f1']:.6f}")

        if args.mode in {"scx_v2", "both"}:
            print(f"[run] scx_v2 seed={seed}")
            metrics = run_scx_v2(args, split, x_cal, y_cal, seed)
            run_records.append({"mode": "scx_v2", "seed": seed, "metrics": metrics})
            print(f"[result] scx_v2 seed={seed} accuracy={metrics['accuracy']:.6f} f1={metrics['f1']:.6f}")

    output_payload: dict[str, Any] = {
        "config": serialize_config(args),
        "dataset": {
            "rows_total": int(len(x)),
            "rows_seed": int(len(split.x_seed)),
            "rows_stream": int(len(split.x_stream)),
            "rows_calibration": int(len(x_cal)),
            "rows_test": int(len(split.x_test)),
            "features_used": int(split.x_seed.shape[1]),
            "raw_label_counts": label_counts,
            "binary_positive_ratio": float(np.mean(y)),
            "feature_sample": feature_names[:10],
        },
        "runs": run_records,
        "summary": summarize_runs(run_records),
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"[done] wrote results to {args.output}")


if __name__ == "__main__":
    main()
