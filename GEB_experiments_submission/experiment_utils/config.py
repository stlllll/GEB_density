"""Dataset definitions and default parameters for the density benchmark."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "train_test"
RESULTS_ROOT = PROJECT_ROOT / "results"
DENSITY_RESULTS_ROOT = RESULTS_ROOT / "density_estimation"

MASTER_SEED = 2026
N_REPETITIONS = 20
CDF_EPS = 1e-12
UNIFORM_WEIGHT = 0.999
NORMAL_WEIGHT = 0.001

GEB_M = 32
GEB_ALPHA = 1.0
PT_MAX_DEPTH = 15
PT_ALPHA_SCALE = 0.1

DPMM_MAX_COMPONENTS = 50
DPMM_MAX_ITER = 1_000
DPMM_N_INIT = 5
DPMM_WEIGHT_CONCENTRATION = 1.0
DPMM_TOL = 1e-3
DPMM_REG_COVAR = 1e-6

DATASETS = {
    "gaussian_mixture_low_d": {
        "kind": "simulation_gaussian", "d": 2,
        "n_train": 5_000, "n_test": 5_000,
    },
    "gaussian_mixture_high_d_sparse": {
        "kind": "simulation_gaussian", "d": 50,
        "n_train": 5_000, "n_test": 5_000,
    },
    "hyperrectangle_structure": {
        "kind": "simulation_hyperrectangle", "d": 5,
        "n_train": 5_000, "n_test": 5_000,
    },
    "dry_bean": {"kind": "uci", "uci_id": 602},
    "isolet": {
        "kind": "uci", "uci_id": 54,
        "standardize_pca_components": 100,
        "analysis_d": 100,
    },
    "household_power": {
        "kind": "uci", "uci_id": 235,
        "n_train": 50_000, "n_test": 50_000,
        "source_d": 8,
        "analysis_d": 6,
    },
}

TREE_TRANSFORMS = {
    "gaussian_mixture_low_d": ("mixture", "normal"),
    "gaussian_mixture_high_d_sparse": ("mixture", "normal"),
    "hyperrectangle_structure": ("mixture", "normal"),
    "dry_bean": ("normal",),
    "isolet": ("normal",),
    "household_power": ("mixture",),
}


def split_file(dataset, repetition, split):
    return DATA_ROOT / dataset / f"repetition_{repetition:02d}_{split}.npz"


def result_file(dataset):
    return DENSITY_RESULTS_ROOT / f"{dataset}_results.csv"
