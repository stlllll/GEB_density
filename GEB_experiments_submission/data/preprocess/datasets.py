"""Simulation definitions and real-data preprocessing for the paper datasets."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from experiment_utils.config import PROJECT_ROOT


def sample_gaussian_mixture(n, dimension, rng):
    """Sample the paper's mixture, adding Gaussian noise after coordinate two."""
    component = rng.binomial(1, 0.5, size=n)
    means = (np.array([-2.0, -1.0]), np.array([2.0, 1.0]))
    covariances = (
        np.array([[0.70, 0.35], [0.35, 0.80]]),
        np.array([[0.60, -0.25], [-0.25, 0.90]]),
    )
    x = np.empty((n, dimension))
    for group in (0, 1):
        mask = component == group
        x[mask, :2] = rng.multivariate_normal(
            means[group], covariances[group], mask.sum()
        )
    if dimension > 2:
        x[:, 2:] = rng.normal(size=(n, dimension - 2))
    return x, component


def sample_hyperrectangle_mixture(n, dimension, rng):
    """Sample 90% Uniform([.2,.8]^D) plus 10% Uniform([0,1]^D)."""
    component = rng.binomial(1, 0.9, size=n)
    x = rng.uniform(0.0, 1.0, size=(n, dimension))
    central = component == 1
    x[central] = rng.uniform(0.2, 0.8, size=(central.sum(), dimension))
    return x, component


def _load_clean_uci(uci_id, keep_two_level=False):
    """Load numeric complete cases and preserve labels for metadata only."""
    dataset = fetch_ucirepo(id=uci_id)
    raw = dataset.data.features.copy()
    targets = dataset.data.targets

    seen = {}
    columns = []
    for position, column in enumerate(map(str, raw.columns), start=1):
        column = column or f"Attribute{position}"
        seen[column] = seen.get(column, 0) + 1
        suffix = "" if seen[column] == 1 else f"__{seen[column]}"
        columns.append(column + suffix)
    raw.columns = columns

    numeric = raw.apply(pd.to_numeric, errors="coerce")
    unique = numeric.nunique(dropna=True)
    threshold = 1 if keep_two_level else 2
    dropped = numeric.columns[
        numeric.isna().all() | (unique <= threshold)
    ].tolist()
    complete = numeric.drop(columns=dropped).dropna(axis=0, how="any")
    if complete.shape[1] == 0 or len(complete) < 2:
        raise ValueError("No usable complete-case continuous data remain")

    aligned = targets.reindex(complete.index)
    labels = None
    if isinstance(aligned, pd.Series):
        labels = aligned.astype(str).to_numpy()
    elif aligned.shape[1] >= 1:
        labels = aligned.iloc[:, 0].astype(str).to_numpy()
    return complete, labels, dropped, len(raw)


def load_data_dry_bean():
    """Return cleaned observations, labels, dropped columns, and raw row count."""
    return _load_clean_uci(uci_id=602)


def load_data_isolet():
    """Return all cleaned ISOLET predictors before split-specific PCA."""
    return _load_clean_uci(uci_id=54, keep_two_level=True)


def prepare_isolet_split(x_train, x_test, n_components=100):
    """Fit standardization and PCA on training data, then transform both splits."""
    scaler = StandardScaler().fit(x_train)
    standardized_train = scaler.transform(x_train)
    standardized_test = scaler.transform(x_test)
    pca = PCA(n_components=n_components, svd_solver="full").fit(
        standardized_train
    )
    train = pca.transform(standardized_train)
    test = pca.transform(standardized_test)
    features = np.asarray(
        [f"PC{j}" for j in range(1, n_components + 1)], dtype=str
    )
    return train, test, features


def load_data_power():
    """Load POWER and reproduce the six-dimensional MAF preprocessing."""
    path = PROJECT_ROOT / "data" / "raw" / "power.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the MAF data archive from "
            "https://zenodo.org/records/1161203 and save power/data.npy as "
            "data/raw/power.npy."
        )
    source = np.load(path, mmap_mode="r")
    if source.shape != (2_049_280, 8) or source.dtype != np.float64:
        raise ValueError(
            f"Unexpected MAF POWER matrix {source.shape}, {source.dtype}; "
            "expected (2049280, 8), float64"
        )
    if not np.all(np.isfinite(source)):
        raise ValueError("MAF POWER matrix contains non-finite values")
    n = source.shape[0]
    rng = np.random.RandomState(42)

    source_rows = np.arange(n)
    rng.shuffle(source_rows)

    retained_columns = (0, 2, 4, 5, 6, 7)
    data = np.empty((n, len(retained_columns)), dtype=float)
    for target, original in enumerate(retained_columns):
        data[:, target] = source[source_rows, original]

    voltage_noise = 0.01 * rng.rand(n)
    active_power_noise = 0.001 * rng.rand(n)
    sub_metering_noise = rng.rand(n, 3)
    data[:, 0] += active_power_noise
    data[:, 1] += voltage_noise
    data[:, 2:5] += sub_metering_noise

    n_test = int(0.1 * n)
    train_validation_end = n - n_test
    mean = data[:train_validation_end].mean(axis=0)
    sd = data[:train_validation_end].std(axis=0, ddof=0)
    if np.any(~np.isfinite(sd)) or np.any(sd <= np.finfo(float).eps):
        raise ValueError("Degenerate MAF POWER standardization scale")
    data -= mean
    data /= sd
    features = np.asarray([
        "Global_active_power", "Voltage", "Sub_metering_1",
        "Sub_metering_2", "Sub_metering_3", "Time_of_day_minutes",
    ], dtype=str)
    return data, source_rows, mean, sd, features
