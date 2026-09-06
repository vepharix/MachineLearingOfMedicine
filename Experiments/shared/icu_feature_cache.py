#!/usr/bin/env python3
"""Shared, leakage-safe ICU feature extraction and local cache helpers."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = ROOT / "Experiments" / "02_mortality_prediction_baseline" / "run_experiment.py"
CACHE_VERSION = 1


def default_data_dir() -> Path:
    """Prefer the private sibling data directory, then the in-repo layout."""

    candidates = (ROOT.parent / "release", ROOT / "release")
    for candidate in candidates:
        if (candidate / "outcomes.csv").exists() and (candidate / "icu_records").is_dir():
            return candidate
    return candidates[0]


def _load_baseline_module():
    spec = importlib.util.spec_from_file_location("mortality_feature_extractor", BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load feature extractor from {BASELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_id_digest(record_ids: np.ndarray) -> str:
    normalized = np.asarray(record_ids, dtype="<i8")
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def read_outcomes(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "outcomes.csv"
    record_dir = data_dir / "icu_records"
    if not path.exists() or not record_dir.is_dir():
        raise FileNotFoundError(
            f"Expected outcomes.csv and icu_records/ under {data_dir}. "
            "Pass --data-dir when the private dataset is elsewhere."
        )
    outcomes = pd.read_csv(path)
    required = {
        "RecordID",
        "SAPS-I",
        "SOFA",
        "Length_of_stay",
        "Survival",
        "In-hospital_death",
    }
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"outcomes.csv is missing columns: {missing}")
    for column in required:
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce")
    outcomes = outcomes.dropna(subset=["RecordID"]).copy()
    outcomes["RecordID"] = outcomes["RecordID"].astype(int)
    if outcomes["RecordID"].duplicated().any():
        raise ValueError("outcomes.csv contains duplicate RecordID values")
    return outcomes.sort_values("RecordID").reset_index(drop=True)


def load_or_extract_features(
    *,
    data_dir: Path,
    record_ids: np.ndarray,
    horizons: list[int],
    cache_dir: Path,
    use_cache: bool = True,
) -> tuple[list[str], dict[int, np.ndarray], dict[str, int], list[dict[str, object]], bool]:
    """Load an exact-record cache or extract features with Experiment 02 rules."""

    record_ids = np.asarray(record_ids, dtype=np.int64)
    horizons = sorted(set(int(value) for value in horizons))
    if not horizons or any(value <= 0 or value > 48 for value in horizons):
        raise ValueError("horizons must be unique integers between 1 and 48")

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    digest = _record_id_digest(record_ids)
    if use_cache and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "cache_version": CACHE_VERSION,
            "record_count": int(len(record_ids)),
            "record_id_sha256": digest,
        }
        cache_matches = all(manifest.get(key) == value for key, value in expected.items())
        cache_matches = cache_matches and set(horizons).issubset(set(manifest.get("horizons_hours", [])))
        matrices: dict[int, np.ndarray] = {}
        if cache_matches:
            for horizon in horizons:
                path = cache_dir / f"features_{horizon}h.npz"
                if not path.exists():
                    cache_matches = False
                    break
                with np.load(path, allow_pickle=False) as saved:
                    cached_ids = saved["record_ids"].astype(np.int64, copy=False)
                    if not np.array_equal(cached_ids, record_ids):
                        cache_matches = False
                        break
                    matrices[horizon] = saved["matrix"].astype(np.float32, copy=False)
        if cache_matches:
            return (
                list(manifest["feature_columns"]),
                matrices,
                {str(key): int(value) for key, value in manifest.get("quality_counters", {}).items()},
                list(manifest.get("coverage_rows", [])),
                True,
            )

    baseline = _load_baseline_module()
    record_dir = data_dir / "icu_records"
    record_paths = [record_dir / f"{record_id}.csv" for record_id in record_ids]
    missing_paths = [path for path in record_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing {len(missing_paths)} ICU files; first: {missing_paths[0]}")
    columns, matrices, quality, coverage_rows = baseline.extract_feature_matrices(record_paths, horizons)
    for horizon, matrix in matrices.items():
        np.savez_compressed(
            cache_dir / f"features_{horizon}h.npz",
            record_ids=record_ids,
            matrix=matrix.astype(np.float32, copy=False),
        )
    manifest = {
        "cache_version": CACHE_VERSION,
        "record_count": int(len(record_ids)),
        "record_id_sha256": digest,
        "horizons_hours": horizons,
        "feature_columns": columns,
        "quality_counters": dict(sorted(quality.items())),
        "coverage_rows": coverage_rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return columns, matrices, dict(quality), coverage_rows, False
