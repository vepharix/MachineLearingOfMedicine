#!/usr/bin/env python3
"""Plain-average bed-day baseline for Experiment 06's cohort readout.

The script rebuilds Experiment 06's remaining-stay cohort and quantile-stratified
split, then predicts every test patient with a single constant learned from the
training and validation records, the same development set used to refit the
final models. It reports the bed-day total of that constant against the observed
total, overall and within each ICU type, so the cohort model can be compared with
the simplest capacity estimate. The bootstrap reuses the seed of the
cohort_smeared readout, so both intervals are drawn from the same test resamples.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import run_experiment as base  # noqa: E402
from Experiments.shared.icu_feature_cache import (  # noqa: E402
    default_data_dir,
    load_or_extract_features,
    read_outcomes,
)


ICU_UNITS = {1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"}
# Experiment 06 seeds its readout intervals with random_state + 29 + offset;
# offset 1 is cohort_smeared.
COHORT_BOOTSTRAP_OFFSET = 29 + 1


def bias_percent(predicted_sum: float, true_sum: float) -> float:
    return float(100.0 * (predicted_sum - true_sum) / true_sum)


def constant_baselines(
    remaining: np.ndarray,
    development: np.ndarray,
    test: np.ndarray,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Development mean and median as constant predictions, with bootstrap intervals."""

    frames = []
    for readout, constant in (
        ("development_mean", float(np.mean(remaining[development]))),
        ("development_median", float(np.median(remaining[development]))),
    ):
        prediction = np.full(len(test), constant)
        frame = base.bootstrap_intervals(remaining[test], prediction, repeats, random_state)
        point = base.regression_metrics(remaining[test], prediction)
        frame.insert(0, "constant_days", constant)
        frame.insert(0, "readout", readout)
        frame["true_sum_bed_days"] = point["true_sum_bed_days"]
        frame["predicted_sum_bed_days"] = point["predicted_sum_bed_days"]
        frame["bootstrap_random_state"] = random_state
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def averages_by_icu(
    remaining: np.ndarray,
    icu_type: np.ndarray,
    development: np.ndarray,
    test: np.ndarray,
) -> pd.DataFrame:
    """Bed-day totals when each ICU type uses its own or the pooled development mean."""

    pooled = float(np.mean(remaining[development]))
    rows = []
    for value, unit in [(None, "All")] + list(ICU_UNITS.items()):
        dev = development if value is None else development[icu_type[development] == value]
        tst = test if value is None else test[icu_type[test] == value]
        own = float(np.mean(remaining[dev]))
        true_sum = float(np.sum(remaining[tst]))
        rows.append(
            {
                "subgroup": "all" if value is None else f"ICUType_{value}",
                "icu_unit": unit,
                "n_development": int(len(dev)),
                "n_test": int(len(tst)),
                "true_sum_bed_days": true_sum,
                "own_average_days": own,
                "own_average_sum_bed_days": own * len(tst),
                "own_average_bias_percent": bias_percent(own * len(tst), true_sum),
                "pooled_average_days": pooled,
                "pooled_average_sum_bed_days": pooled * len(tst),
                "pooled_average_bias_percent": bias_percent(pooled * len(tst), true_sum),
            }
        )
    return pd.DataFrame(rows)


def attach_cohort_model(
    by_icu: pd.DataFrame, primary: pd.DataFrame, subgroups: pd.DataFrame
) -> pd.DataFrame:
    """Add the cohort_smeared bias already reported by run_experiment.py."""

    model = subgroups[subgroups["readout"].eq("cohort_smeared")].set_index("subgroup")
    overall = primary[primary["readout"].eq("cohort_smeared")].iloc[0]
    result = by_icu.copy()
    result["cohort_model_bias_percent"] = [
        float(overall["sum_bias_percent"]) if subgroup == "all" else float(model.loc[subgroup, "sum_bias_percent"])
        for subgroup in result["subgroup"]
    ]
    return result


def check_split_matches(by_icu: pd.DataFrame, primary: pd.DataFrame, subgroups: pd.DataFrame) -> None:
    """Stop if the rebuilt test set differs from Experiment 06's saved outputs."""

    model = subgroups[subgroups["readout"].eq("cohort_smeared")].set_index("subgroup")
    overall = primary[primary["readout"].eq("cohort_smeared")].iloc[0]
    for row in by_icu.itertuples(index=False):
        saved = overall if row.subgroup == "all" else model.loc[row.subgroup]
        if int(saved["n"]) != row.n_test or not np.isclose(saved["true_sum_bed_days"], row.true_sum_bed_days):
            raise ValueError(
                f"{row.subgroup}: rebuilt test set has n={row.n_test}, bed-days={row.true_sum_bed_days:.0f}; "
                f"Experiment 06 reports n={int(saved['n'])}, bed-days={saved['true_sum_bed_days']:.0f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "Experiments" / "shared" / "output" / "data")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_path = args.output_dir / "primary_test_metrics.csv"
    subgroup_path = args.output_dir / "subgroup_metrics.csv"
    if not primary_path.exists() or not subgroup_path.exists():
        raise FileNotFoundError("Run Experiment 06's run_experiment.py before this baseline.")
    primary = pd.read_csv(primary_path)
    subgroups = pd.read_csv(subgroup_path)

    all_outcomes = read_outcomes(args.data_dir)
    feature_names, matrices, _, _, cache_hit = load_or_extract_features(
        data_dir=args.data_dir,
        record_ids=all_outcomes["RecordID"].to_numpy(np.int64),
        horizons=[48],
        cache_dir=args.cache_dir,
    )
    los = all_outcomes["Length_of_stay"].to_numpy(float)
    valid = np.isfinite(los) & (los >= 2)
    remaining = los[valid] - 2.0
    icu_type = base.icu_types_from_matrix(matrices[48][valid], feature_names)
    splits = base.make_quantile_splits(remaining, args.random_state)
    development = np.sort(np.concatenate([splits["train"], splits["validation"]]))
    test = splits["test"]

    by_icu = averages_by_icu(remaining, icu_type, development, test)
    check_split_matches(by_icu, primary, subgroups)
    by_icu = attach_cohort_model(by_icu, primary, subgroups)
    seed = args.random_state + COHORT_BOOTSTRAP_OFFSET
    overall = constant_baselines(remaining, development, test, args.bootstrap_repeats, seed)

    overall.to_csv(args.output_dir / "plain_average_baseline.csv", index=False)
    by_icu.to_csv(args.output_dir / "plain_average_baseline_by_icu.csv", index=False)
    metadata = {
        "target": "Length_of_stay - 2 days",
        "split_sizes": {name: int(len(index)) for name, index in splits.items()},
        "constant_fitted_on": "training + validation (development) records",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_random_state": seed,
        "bootstrap_note": "same seed and resamples as the cohort_smeared rows of bootstrap_confidence_intervals.csv",
        "cache_hit": bool(cache_hit),
        "random_state": args.random_state,
    }
    (args.output_dir / "plain_average_baseline_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(overall.to_string(index=False))
    print(by_icu.to_string(index=False))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
