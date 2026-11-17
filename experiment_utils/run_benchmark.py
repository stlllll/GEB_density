"""Restartable Python runner for GEB, PT, and raw-data VB-DPMM."""

import argparse
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from experiment_utils.transforms import transform_train_test
from experiment_utils.tables import write_density_tables
from experiment_utils.config import (
    DATASETS, GEB_M, GEB_ALPHA, N_REPETITIONS, PT_ALPHA_SCALE, PT_MAX_DEPTH,
    TREE_TRANSFORMS, result_file, split_file,
)
from methods.baseline.dpmm import fit_and_score_dpmm
from methods.GEB import GEBTree
from methods.baseline.polya_tree import ClassicalPolyaTree


KEY = "repetition"
SUMMARY_LABELS = ("Mean", "Standard error")


def _numeric_repetition_rows(table):
    repetitions = pd.to_numeric(table[KEY], errors="coerce")
    numeric = repetitions.notna()
    result = table.loc[numeric].copy()
    result[KEY] = repetitions.loc[numeric].astype(int)
    return result


def add_summary_rows(table):
    """Replace any old summary rows with current mean and standard error."""
    data = _numeric_repetition_rows(table).sort_values(KEY)
    mean_row = {column: np.nan for column in data.columns}
    se_row = {column: np.nan for column in data.columns}
    mean_row[KEY], se_row[KEY] = SUMMARY_LABELS

    for column in data.columns:
        if column in {KEY, "n", "D"}:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        # The first GEB timing contains JAX compilation and is not reported.
        if column.startswith("GEB_") and column.endswith("_running_time"):
            values = values.loc[data[KEY] != 1]
        values = values.dropna()
        if len(values):
            mean_row[column] = float(values.mean())
        if len(values) > 1:
            se_row[column] = float(values.std(ddof=1) / np.sqrt(len(values)))

    summary = pd.DataFrame([mean_row, se_row], columns=data.columns)
    return pd.concat([data, summary], ignore_index=True)


def completed_repetitions(path, score_column):
    path = Path(path)
    if not path.exists():
        return set()
    table = pd.read_csv(path)
    if score_column not in table:
        return set()
    repetitions = pd.to_numeric(table[KEY], errors="coerce")
    scores = pd.to_numeric(table[score_column], errors="coerce")
    return set(repetitions.loc[repetitions.notna() & scores.notna()].astype(int))


def update_result(path, repetition, values):
    path = Path(path)
    table = _numeric_repetition_rows(pd.read_csv(path))
    row = table.index[table[KEY] == repetition]
    if len(row) != 1:
        raise ValueError(f"Expected one row for repetition {repetition} in {path}")
    for column, value in values.items():
        if column not in table:
            table[column] = np.nan
        table.loc[row[0], column] = value
    temporary = path.with_suffix(path.suffix + ".tmp")
    add_summary_rows(table).to_csv(temporary, index=False)
    os.replace(temporary, path)
    write_density_tables(path)


def load_split(dataset, repetition, split):
    path = split_file(dataset, repetition, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python -m data.preprocess.generate."
        )
    with np.load(path, allow_pickle=False) as saved:
        x = saved["x"].copy()
    expected_d = DATASETS.get(dataset, {}).get("analysis_d")
    if expected_d is not None and x.shape[1] != expected_d:
        raise ValueError(
            f"Stale {dataset} split has D={x.shape[1]}, expected D={expected_d}. "
            "Regenerate it with --reset-results."
        )
    return x


def fit_and_score_geb(u_train, u_test, m=GEB_M, alpha=GEB_ALPHA, grid=True):
    d = u_train.shape[1]
    start = time.perf_counter()
    model = GEBTree(
        d, np.zeros(d), np.ones(d), alpha,
        u_train, grid=grid, m=m,
    )
    model.tree_joint_density_set(model.root)
    density = model.tree_joint_density_batch(u_test)
    return (
        np.log(np.maximum(density, 1e-300)),
        time.perf_counter() - start,
        model,
    )


def score_geb(u_train, u_test, m=GEB_M, alpha=GEB_ALPHA, grid=True):
    score, elapsed, _ = fit_and_score_geb(u_train, u_test, m, alpha, grid)
    return score, elapsed


def score_pt(u_train, u_test, max_depth=PT_MAX_DEPTH, alpha_scale=PT_ALPHA_SCALE):
    start = time.perf_counter()
    model = ClassicalPolyaTree(max_depth, alpha_scale).fit(u_train)
    return model.score_samples(u_test), time.perf_counter() - start


def run_tree(dataset, model_name, transform, repetitions=None, parameters=None):
    scorer = {"GEB": score_geb, "PT": score_pt}[model_name]
    stem = f"{model_name}_{transform}"
    score_column = f"{stem}_log_score"
    time_column = f"{stem}_running_time"
    path = result_file(dataset)
    done = completed_repetitions(path, score_column)
    repetitions = repetitions or range(1, N_REPETITIONS + 1)
    for repetition in repetitions:
        if repetition in done:
            print(f"{stem} {dataset} repetition {repetition:02d}: completed", flush=True)
            continue
        x_train = load_split(dataset, repetition, "train")
        x_test = load_split(dataset, repetition, "test")
        u_train, (u_test, log_jacobian) = transform_train_test(
            x_train, x_test, transform
        )
        print(f"{stem} {dataset} repetition {repetition:02d}: running", flush=True)
        score, elapsed = scorer(u_train, u_test, **(parameters or {}))
        raw_score = score + log_jacobian
        update_result(path, repetition, {
            score_column: float(np.mean(raw_score)),
            time_column: float(elapsed),
        })
        print(
            f"{stem} {dataset} repetition {repetition:02d}: "
            f"log_score={raw_score.mean():.6f}, time={elapsed:.3f}s",
            flush=True,
        )


def run_dpmm(dataset, repetitions=None, parameters=None):
    scale = "pca" if dataset == "isolet" else "raw"
    score_column = f"DPMM_{scale}_log_score"
    time_column = f"DPMM_{scale}_running_time"
    path = result_file(dataset)
    done = completed_repetitions(path, score_column)
    repetitions = repetitions or range(1, N_REPETITIONS + 1)
    for repetition in repetitions:
        if repetition in done:
            print(
                f"DPMM_{scale} {dataset} repetition {repetition:02d}: completed",
                flush=True,
            )
            continue
        x_train = load_split(dataset, repetition, "train")
        x_test = load_split(dataset, repetition, "test")
        print(f"DPMM {dataset} repetition {repetition:02d}: running", flush=True)
        score, elapsed = fit_and_score_dpmm(
            x_train, x_test, seed=2026 + 10_000 * repetition + x_train.shape[1],
            **(parameters or {}),
        )
        update_result(path, repetition, {
            score_column: float(np.mean(score)),
            time_column: float(elapsed),
        })
        print(
            f"DPMM_{scale} {dataset} repetition {repetition:02d}: "
            f"log_score={score.mean():.6f}, time={elapsed:.3f}s",
            flush=True,
        )


def run_dataset(dataset, models=("GEB", "PT", "DPMM"), repetitions=None, parameters=None):
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset: {dataset}")
    if not result_file(dataset).exists():
        raise FileNotFoundError(
            f"Missing result skeleton for {dataset}; generate the data first."
        )
    for model in models:
        model_parameters = (parameters or {}).get(model, {})
        if model in ("GEB", "PT"):
            for transform in TREE_TRANSFORMS[dataset]:
                run_tree(dataset, model, transform, repetitions, model_parameters)
        elif model == "DPMM":
            run_dpmm(dataset, repetitions, model_parameters)
        else:
            raise ValueError(f"Unknown Python model: {model}")
    write_density_tables(result_file(dataset))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--model", choices=["all", "GEB", "PT", "DPMM"], default="all")
    parser.add_argument("--repetition", type=int, choices=range(1, N_REPETITIONS + 1))
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    models = ("GEB", "PT", "DPMM") if args.model == "all" else (args.model,)
    repetitions = None if args.repetition is None else (args.repetition,)
    for dataset in datasets:
        run_dataset(dataset, models=models, repetitions=repetitions)


if __name__ == "__main__":
    main()
