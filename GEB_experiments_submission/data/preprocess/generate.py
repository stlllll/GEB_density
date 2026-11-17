"""Construct the twenty saved train/test splits for each paper dataset."""

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from experiment_utils.config import DATASETS, DATA_ROOT, MASTER_SEED, N_REPETITIONS, split_file
from data.preprocess.datasets import (
    load_data_dry_bean,
    load_data_isolet,
    prepare_isolet_split,
    load_data_power,
    sample_gaussian_mixture,
    sample_hyperrectangle_mixture,
)
from data.preprocess.storage import (
    initialize_results,
    preflight_result_design,
    save_split,
)


SIMULATION_SAMPLERS = {
    "simulation_gaussian": sample_gaussian_mixture,
    "simulation_hyperrectangle": sample_hyperrectangle_mixture,
}

REAL_DATA_LOADERS = {
    "dry_bean": load_data_dry_bean,
    "isolet": load_data_isolet,
}


def generate_simulation(name, spec, reset_results=False):
    """Generate the repeated train/test samples for one simulation."""
    folder = DATA_ROOT / name
    rows = [
        {"repetition": repetition, "n": spec["n_train"], "D": spec["d"]}
        for repetition in range(1, N_REPETITIONS + 1)
    ]
    preflight_result_design(name, rows, reset_results)
    sampler = SIMULATION_SAMPLERS[spec["kind"]]
    for repetition in range(1, N_REPETITIONS + 1):
        seed = MASTER_SEED + repetition - 1
        rng = np.random.default_rng(seed)
        train, train_component = sampler(spec["n_train"], spec["d"], rng)
        test, test_component = sampler(spec["n_test"], spec["d"], rng)
        save_split(
            folder, repetition, train, test,
            [f"X{j}" for j in range(1, spec["d"] + 1)], seed,
            {"train": {"component": train_component},
             "test": {"component": test_component}},
        )
    initialize_results(name, rows, reset_results)
    return {
        "dataset": name, "source": spec["kind"], "raw_rows": None,
        "complete_rows": None, "D": spec["d"], "dropped": "",
    }


def generate_power(name, spec, reset_results=False):
    """Generate the repeated 50,000/50,000 splits for POWER."""
    n_train, n_test = spec["n_train"], spec["n_test"]
    rows = [
        {"repetition": repetition, "n": n_train, "D": 6}
        for repetition in range(1, N_REPETITIONS + 1)
    ]
    preflight_result_design(name, rows, reset_results)
    data, source_rows, mean, sd, features = load_data_power()
    total = n_train + n_test
    if total > len(data):
        raise ValueError(
            f"POWER requests {total} distinct rows, but only {len(data)} exist."
        )

    folder = DATA_ROOT / name
    positions = np.arange(len(data))
    for repetition in range(1, N_REPETITIONS + 1):
        seed = MASTER_SEED + repetition - 1
        rng = np.random.default_rng(seed)
        chosen = rng.choice(positions, total, replace=False)
        train_pos = chosen[:n_train]
        test_pos = chosen[n_train:]
        metadata = {
            "train": {
                "row_index": source_rows[train_pos],
                "maf_standardization_mean": mean,
                "maf_standardization_sd": sd,
            },
            "test": {
                "row_index": source_rows[test_pos],
                "maf_standardization_mean": mean,
                "maf_standardization_sd": sd,
            },
        }
        save_split(
            folder, repetition, data[train_pos], data[test_pos],
            features, seed, metadata,
        )
    initialize_results(name, rows, reset_results)
    return {
        "dataset": name, "source": "MAF Zenodo 1161203 (UCI 235)",
        "raw_rows": len(data), "complete_rows": len(data), "D": 6,
        "source_D": 8, "analysis_scale": "MAF standardized",
        "dropped": "Global_reactive_power | Global_intensity",
    }


def generate_real_dataset(name, spec, reset_results=False):
    """Generate all repetitions for Dry Bean or ISOLET."""
    data, labels, dropped, raw_rows = REAL_DATA_LOADERS[name]()
    positions = np.arange(len(data))
    features = data.columns.to_numpy(str)
    folder = DATA_ROOT / name
    rows = []
    analysis_d = spec.get("standardize_pca_components", data.shape[1])
    train_n = len(data) // 2
    expected_rows = [
        {"repetition": repetition, "n": train_n, "D": analysis_d}
        for repetition in range(1, N_REPETITIONS + 1)
    ]
    preflight_result_design(name, expected_rows, reset_results)

    for repetition in range(1, N_REPETITIONS + 1):
        seed = MASTER_SEED + repetition - 1
        train_pos, test_pos = train_test_split(
            positions, test_size=0.5, random_state=seed, shuffle=True
        )
        x_train = data.iloc[train_pos].to_numpy(float)
        x_test = data.iloc[test_pos].to_numpy(float)
        analysis_features = features

        pca_components = spec.get("standardize_pca_components")
        if pca_components is not None:
            x_train, x_test, analysis_features = prepare_isolet_split(
                x_train, x_test, pca_components
            )

        extras = {
            "train": {"row_index": data.index.to_numpy()[train_pos]},
            "test": {"row_index": data.index.to_numpy()[test_pos]},
        }
        if labels is not None:
            extras["train"]["label"] = labels[train_pos]
            extras["test"]["label"] = labels[test_pos]
        save_split(
            folder, repetition, x_train, x_test, analysis_features, seed, extras,
        )
        rows.append({
            "repetition": repetition, "n": len(train_pos), "D": x_train.shape[1]
        })

    initialize_results(name, rows, reset_results)
    return {
        "dataset": name,
        "source": f"UCI {spec['uci_id']}",
        "raw_rows": raw_rows,
        "complete_rows": len(data),
        "D": x_train.shape[1],
        "source_D": spec.get("source_d", data.shape[1]),
        "analysis_scale": (
            "standardized training PCA" if pca_components is not None
            else "retained raw"
        ),
        "dropped": " | ".join(map(str, dropped)),
    }


def update_manifest(row):
    """Update the human-readable dataset manifest after successful generation."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    path = DATA_ROOT / "dataset_manifest.csv"
    if path.exists():
        manifest = pd.read_csv(path)
        manifest = manifest.loc[manifest["dataset"] != row["dataset"]]
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    else:
        manifest = pd.DataFrame([row])
    order = {dataset_name: i for i, dataset_name in enumerate(DATASETS)}
    manifest["_order"] = manifest["dataset"].map(order)
    manifest = manifest.sort_values("_order").drop(columns="_order")
    manifest.to_csv(path, index=False)


def generate_dataset(name, reset_results=False):
    """Generate and save all twenty splits for one configured dataset."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; choose from {list(DATASETS)}")
    spec = DATASETS[name]
    print(f"Generating {name} ...", flush=True)
    if name == "household_power":
        row = generate_power(name, spec, reset_results)
    elif spec["kind"].startswith("simulation_"):
        row = generate_simulation(name, spec, reset_results)
    else:
        row = generate_real_dataset(name, spec, reset_results)
    update_manifest(row)
    print(row, flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--reset-results", action="store_true",
                        help="Regenerate the selected splits and clear their density results.")
    args = parser.parse_args()
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        complete = all(
            split_file(dataset, repetition, part).exists()
            for repetition in range(1, N_REPETITIONS + 1)
            for part in ("train", "test")
        )
        if args.reset_results or not complete:
            generate_dataset(dataset, reset_results=args.reset_results)
            continue

        rows = []
        for repetition in range(1, N_REPETITIONS + 1):
            with np.load(split_file(dataset, repetition, "train"), allow_pickle=False) as saved:
                n, dimension = saved["x"].shape
            rows.append({"repetition": repetition, "n": n, "D": dimension})
        initialize_results(dataset, rows)
        print(f"[{dataset}] All {N_REPETITIONS} split pairs already exist.", flush=True)


if __name__ == "__main__":
    main()
