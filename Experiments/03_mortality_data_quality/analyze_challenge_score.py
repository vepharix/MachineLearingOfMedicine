#!/usr/bin/env python3
"""PhysioNet/CinC Challenge 2012 Event 1 score for the two selected 48-hour models.

Event 1 of the 2012 Challenge scored binary in-hospital mortality predictions by
min(sensitivity, positive predictive value). This script refits the two
selected 48-hour pipelines on the training split exactly as run_experiment.py
does, checks that the refitted confirmation-set probabilities equal the frozen
predictions, chooses the threshold that maximises the score on the validation
split, and evaluates that threshold once on the confirmation split.

The local confirmation split is not the Challenge's hidden test set B or C, and
the training split has more labelled records than Challenge set A, so the score
is a same-metric reference rather than a leaderboard position.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
MAIN_PATH = EXPERIMENT_DIR / "run_experiment.py"
HORIZON_HOURS = 48
MODELS = ("hist_gradient_boosting", "logistic")


def load_main_module():
    spec = importlib.util.spec_from_file_location("mortality_quality_main", MAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = load_main_module()


def event1_curve(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    """Sensitivity, PPV and min(Se, PPV) at every distinct probability.

    A record is flagged when probability >= threshold, as in metric_row().
    Every threshold is an observed probability, so at least one record is
    flagged and PPV is always defined.
    """

    y = np.asarray(y, dtype=np.int8)
    probability = np.asarray(probability, dtype=float)
    positive = np.sort(probability[y == 1])
    negative = np.sort(probability[y == 0])
    thresholds = np.unique(probability)
    true_positive = len(positive) - np.searchsorted(positive, thresholds, side="left")
    false_positive = len(negative) - np.searchsorted(negative, thresholds, side="left")
    sensitivity = true_positive / len(positive)
    ppv = true_positive / (true_positive + false_positive)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "flagged": true_positive + false_positive,
            "sensitivity": sensitivity,
            "ppv": ppv,
            "event1_score": np.minimum(sensitivity, ppv),
        }
    )


def event1_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """Threshold with the highest min(Se, PPV); ties go to the higher threshold."""

    curve = event1_curve(y, probability)
    best = np.flatnonzero(curve["event1_score"].to_numpy() == curve["event1_score"].max())
    return float(curve["threshold"].iloc[best[-1]])


def event1_score(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Binary Event 1 counts at a fixed threshold; no alerts counts as PPV 0."""

    y = np.asarray(y, dtype=np.int8)
    flagged = np.asarray(probability, dtype=float) >= threshold
    true_positive = int(((y == 1) & flagged).sum())
    false_positive = int(((y == 0) & flagged).sum())
    false_negative = int(((y == 1) & ~flagged).sum())
    sensitivity = true_positive / (true_positive + false_negative)
    ppv = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    return {
        "n": int(len(y)),
        "deaths": int((y == 1).sum()),
        "alerts": int(flagged.sum()),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "sensitivity": float(sensitivity),
        "ppv": float(ppv),
        "event1_score": float(min(sensitivity, ppv)),
    }


def bootstrap_event1(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    random_state: int,
    iterations: int,
) -> tuple[float, float]:
    """Percentile interval with the threshold fixed, resampling as in run_experiment.py."""

    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(iterations):
        sample = rng.integers(0, len(y), len(y))
        ys = y[sample]
        if np.unique(ys).size < 2:
            continue
        scores.append(event1_score(ys, probability[sample], threshold)["event1_score"])
    low, high = np.quantile(scores, [0.025, 0.975])
    return float(low), float(high)


def frozen_probability(
    predictions: pd.DataFrame, model_name: str, preprocess: str, record_ids: np.ndarray
) -> np.ndarray:
    frame = predictions[
        predictions["model"].eq(model_name)
        & predictions["preprocess"].eq(preprocess)
        & predictions["horizon_hours"].eq(HORIZON_HOURS)
    ].set_index("RecordID")
    return frame.loc[record_ids, "probability"].to_numpy(dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.output_dir / "data" / "feature_profiles.npz"
    prediction_path = args.output_dir / "data" / "test_predictions.csv"
    metadata_path = args.output_dir / "run_metadata.json"
    if not cache_path.exists() or not prediction_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run Experiment 03's run_experiment.py before this analysis.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache = np.load(cache_path)
    columns = cache["columns"].tolist()
    y = cache["y"].astype(np.int8)
    record_ids = cache["RecordID"].astype(int)
    random_state = int(metadata["random_state"])
    iterations = int(metadata["bootstrap_iterations"])
    splits = experiment.fixed_splits(y, random_state)
    predictions = pd.read_csv(prediction_path)
    configs = {config.name: config for config in experiment.CONFIGS}

    rows = []
    for model_name in MODELS:
        config = configs[metadata["selected_configs"][model_name]]
        x = cache[f"{config.feature_profile}_{HORIZON_HOURS}h"]
        model = experiment.build_model(model_name, config, columns, random_state)
        model.fit(x[splits["train"]], y[splits["train"]])
        validation_probability = model.predict_proba(x[splits["validation"]])[:, 1]
        test_probability = model.predict_proba(x[splits["test"]])[:, 1]
        frozen = frozen_probability(predictions, model_name, config.name, record_ids[splits["test"]])
        if not np.allclose(test_probability, frozen, rtol=0.0, atol=1e-9):
            raise ValueError(f"Refitted {model_name} does not reproduce output/data/test_predictions.csv")

        threshold = event1_threshold(y[splits["validation"]], validation_probability)
        validation = event1_score(y[splits["validation"]], validation_probability, threshold)
        test = event1_score(y[splits["test"]], test_probability, threshold)
        low, high = bootstrap_event1(
            y[splits["test"]], test_probability, threshold, random_state + HORIZON_HOURS, iterations
        )
        row = {
            "model": model_name,
            "preprocess": config.name,
            "horizon_hours": HORIZON_HOURS,
            "threshold_rule": "max validation min(Se, PPV); ties to higher threshold",
            "threshold": threshold,
        }
        row.update({f"validation_{key}": value for key, value in validation.items()})
        row.update({f"test_{key}": value for key, value in test.items()})
        row.update(
            {
                "test_alerts_per_100": float(100 * test["alerts"] / test["n"]),
                "test_event1_ci_2_5": low,
                "test_event1_ci_97_5": high,
                "bootstrap_repeats": iterations,
                "bootstrap_random_state": random_state + HORIZON_HOURS,
            }
        )
        rows.append(row)
        print(
            f"{model_name}: threshold {threshold:.4f}; validation score {validation['event1_score']:.4f}; "
            f"test Se {test['sensitivity']:.4f}, PPV {test['ppv']:.4f}, "
            f"score {test['event1_score']:.4f} (95% CI {low:.4f}-{high:.4f})",
            flush=True,
        )

    output_path = args.output_dir / "challenge2012_event1_score.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Challenge 2012 Event 1 scores written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
