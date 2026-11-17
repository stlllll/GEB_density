"""Shared, reproducible split storage and result-table initialization."""

import numpy as np
import pandas as pd

from experiment_utils.config import DENSITY_RESULTS_ROOT, result_file


def save_split(folder, repetition, train, test, features, seed, extras=None):
    """Save one train/test pair and its metadata as NPZ files."""
    folder.mkdir(parents=True, exist_ok=True)
    shared = {
        "feature_names": np.asarray(features, dtype=str),
        "split_seed": np.asarray(seed),
    }
    extras = extras or {}
    normalized = {}
    for split, values in extras.items():
        normalized[split] = {
            key: (np.asarray(value, dtype=str)
                  if np.asarray(value).dtype == object else np.asarray(value))
            for key, value in values.items()
        }
    np.savez_compressed(
        folder / f"repetition_{repetition:02d}_train.npz",
        x=np.asarray(train, dtype=float), **shared, **normalized.get("train", {}),
    )
    np.savez_compressed(
        folder / f"repetition_{repetition:02d}_test.npz",
        x=np.asarray(test, dtype=float), **shared, **normalized.get("test", {}),
    )


def initialize_results(name, rows, reset=False):
    """Create a result skeleton without erasing compatible completed results."""
    DENSITY_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    path = result_file(name)
    if path.exists() and not reset:
        preflight_result_design(name, rows)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def preflight_result_design(name, rows, reset=False):
    """Reject a design change before overwriting any saved split."""
    path = result_file(name)
    if not path.exists() or reset:
        return
    existing = pd.read_csv(path)
    numeric = pd.to_numeric(existing["repetition"], errors="coerce").notna()
    observed = existing.loc[numeric, ["repetition", "n", "D"]].copy()
    observed["repetition"] = observed["repetition"].astype(int)
    observed["n"] = pd.to_numeric(observed["n"])
    observed["D"] = pd.to_numeric(observed["D"])
    if observed.to_dict("records") != rows:
        raise ValueError(
            f"Existing result design disagrees with the requested data: {path}. "
            f"Rerun with --reset-results after archiving any results you need."
        )
