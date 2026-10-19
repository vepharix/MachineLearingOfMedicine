#!/usr/bin/env python3
"""Paired bootstrap comparison of the two selected 48-hour mortality models.

This script does not refit any model. It reads the frozen confirmation-set
predictions written by run_experiment.py and scores gradient boosting and
logistic regression on the same resampled patients in every iteration, so the
interval describes the difference between the two models instead of two
separate intervals compared by eye. The resampling seed is the one used for the
48-hour intervals in output/test_metrics.csv, which means both analyses draw
the same 1,000 patient samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


EXPERIMENT_DIR = Path(__file__).resolve().parent
HORIZON_HOURS = 48
REFERENCE_MODEL = "logistic"
CANDIDATE_MODEL = "hist_gradient_boosting"
METRICS = {
    "auroc": (roc_auc_score, True),
    "auprc": (average_precision_score, True),
    "brier": (brier_score_loss, False),
}


def aligned_predictions(
    predictions: pd.DataFrame, selected: dict[str, str], horizon: int
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return labels and per-model probabilities ordered by RecordID."""

    frames = {}
    for model_name in (CANDIDATE_MODEL, REFERENCE_MODEL):
        frame = predictions[
            predictions["model"].eq(model_name)
            & predictions["preprocess"].eq(selected[model_name])
            & predictions["horizon_hours"].eq(horizon)
        ].sort_values("RecordID")
        if frame.empty or frame["RecordID"].duplicated().any():
            raise ValueError(f"Expected one {horizon}h prediction per record for {model_name}")
        frames[model_name] = frame.reset_index(drop=True)
    candidate = frames[CANDIDATE_MODEL]
    reference = frames[REFERENCE_MODEL]
    if not np.array_equal(candidate["RecordID"].to_numpy(), reference["RecordID"].to_numpy()):
        raise ValueError("The two models were not scored on the same confirmation records")
    if not np.array_equal(
        candidate["In-hospital_death"].to_numpy(), reference["In-hospital_death"].to_numpy()
    ):
        raise ValueError("Outcome labels differ between the two prediction sets")
    record_ids = candidate["RecordID"].to_numpy(dtype=int)
    y = candidate["In-hospital_death"].to_numpy(dtype=np.int8)
    probabilities = {
        model_name: frame["probability"].to_numpy(dtype=float) for model_name, frame in frames.items()
    }
    return record_ids, y, probabilities


def paired_bootstrap(
    y: np.ndarray,
    probabilities: dict[str, np.ndarray],
    random_state: int,
    iterations: int,
) -> tuple[dict[tuple[str, str], np.ndarray], int]:
    """Score every model on the same resampled records.

    The sampling loop matches bootstrap_intervals() in run_experiment.py:
    one draw of record indices per iteration, skipping single-class samples.
    """

    rng = np.random.default_rng(random_state)
    samples = {(model, metric): [] for model in probabilities for metric in METRICS}
    valid = 0
    for _ in range(iterations):
        sample = rng.integers(0, len(y), len(y))
        ys = y[sample]
        if np.unique(ys).size < 2:
            continue
        valid += 1
        for model, probability in probabilities.items():
            for metric, (function, _) in METRICS.items():
                samples[(model, metric)].append(function(ys, probability[sample]))
    return {key: np.asarray(values) for key, values in samples.items()}, valid


def summarize(
    y: np.ndarray,
    probabilities: dict[str, np.ndarray],
    samples: dict[tuple[str, str], np.ndarray],
    valid: int,
    iterations: int,
    random_state: int,
    selected: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for metric, (function, higher_is_better) in METRICS.items():
        candidate = function(y, probabilities[CANDIDATE_MODEL])
        reference = function(y, probabilities[REFERENCE_MODEL])
        difference = samples[(CANDIDATE_MODEL, metric)] - samples[(REFERENCE_MODEL, metric)]
        low, high = np.quantile(difference, [0.025, 0.975])
        candidate_better = difference > 0 if higher_is_better else difference < 0
        row = {
            "metric": metric,
            "higher_is_better": higher_is_better,
            "horizon_hours": HORIZON_HOURS,
            "n": int(len(y)),
            "deaths": int(y.sum()),
            "hist_gradient_boosting_preprocess": selected[CANDIDATE_MODEL],
            "logistic_preprocess": selected[REFERENCE_MODEL],
            "hist_gradient_boosting": float(candidate),
            "logistic": float(reference),
            "difference_hgb_minus_lr": float(candidate - reference),
            "difference_ci_2_5": float(low),
            "difference_ci_97_5": float(high),
            "interval_excludes_zero": bool(low > 0 or high < 0),
            "share_resamples_hgb_better": float(np.mean(candidate_better)),
        }
        for model in (CANDIDATE_MODEL, REFERENCE_MODEL):
            model_low, model_high = np.quantile(samples[(model, metric)], [0.025, 0.975])
            row[f"{model}_ci_2_5"] = float(model_low)
            row[f"{model}_ci_97_5"] = float(model_high)
        row.update(
            {
                "bootstrap_repeats": iterations,
                "valid_repeats": valid,
                "bootstrap_random_state": random_state,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_with_test_metrics(summary: pd.DataFrame, test_metrics: pd.DataFrame) -> float:
    """Largest absolute gap to the per-model values already in test_metrics.csv."""

    gaps = []
    for model in (CANDIDATE_MODEL, REFERENCE_MODEL):
        saved = test_metrics[
            test_metrics["model"].eq(model) & test_metrics["horizon_hours"].eq(HORIZON_HOURS)
        ].iloc[0]
        for row in summary.itertuples(index=False):
            gaps.append(abs(getattr(row, model) - saved[row.metric]))
            gaps.append(abs(getattr(row, f"{model}_ci_2_5") - saved[f"{row.metric}_ci_low"]))
            gaps.append(abs(getattr(row, f"{model}_ci_97_5") - saved[f"{row.metric}_ci_high"]))
    return float(max(gaps))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = args.output_dir / "data" / "test_predictions.csv"
    metadata_path = args.output_dir / "run_metadata.json"
    if not prediction_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run Experiment 03's run_experiment.py before this comparison.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = metadata["selected_configs"]
    iterations = int(metadata["bootstrap_iterations"])
    random_state = int(metadata["random_state"]) + HORIZON_HOURS
    _, y, probabilities = aligned_predictions(
        pd.read_csv(prediction_path), selected, HORIZON_HOURS
    )
    samples, valid = paired_bootstrap(y, probabilities, random_state, iterations)
    summary = summarize(y, probabilities, samples, valid, iterations, random_state, selected)
    summary.to_csv(args.output_dir / "paired_model_comparison.csv", index=False)

    test_metrics_path = args.output_dir / "test_metrics.csv"
    if test_metrics_path.exists():
        gap = compare_with_test_metrics(summary, pd.read_csv(test_metrics_path))
        print(
            "Largest gap to per-model values and intervals in test_metrics.csv: "
            f"{gap:.2e}",
            flush=True,
        )
    for row in summary.itertuples(index=False):
        print(
            f"{row.metric}: HGB {row.hist_gradient_boosting:.4f} vs LR {row.logistic:.4f}; "
            f"difference {row.difference_hgb_minus_lr:+.4f} "
            f"(95% CI {row.difference_ci_2_5:+.4f} to {row.difference_ci_97_5:+.4f}); "
            f"HGB better in {row.share_resamples_hgb_better:.1%} of resamples",
            flush=True,
        )
    print(f"Paired comparison written to {args.output_dir / 'paired_model_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
