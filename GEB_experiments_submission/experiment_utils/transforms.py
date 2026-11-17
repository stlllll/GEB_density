"""Training-only base-measure transforms and exact raw-scale Jacobians."""

import numpy as np
from scipy.special import ndtr

from experiment_utils.config import CDF_EPS, NORMAL_WEIGHT, UNIFORM_WEIGHT


def _check_scale(scale, label):
    bad = (~np.isfinite(scale)) | (scale <= np.finfo(float).eps)
    if np.any(bad):
        raise ValueError(f"Degenerate {label} coordinates: {np.flatnonzero(bad).tolist()}")


def fit_normal_cdf(x_train):
    mean = np.mean(x_train, axis=0)
    sd = np.std(x_train, axis=0, ddof=0)
    _check_scale(sd, "Normal-CDF")
    return mean, sd


def apply_normal_cdf(x, parameters, jacobian=False):
    mean, sd = parameters
    q = (x - mean) / sd
    u = np.clip(ndtr(q), CDF_EPS, 1.0 - CDF_EPS)
    if not jacobian:
        return u
    log_jacobian = (
        -np.log(sd) - 0.5 * q**2 - 0.5 * np.log(2.0 * np.pi)
    ).sum(axis=1)
    return u, log_jacobian


def fit_mixture_cdf(x_train):
    lower = np.min(x_train, axis=0)
    width = np.max(x_train, axis=0) - lower
    mean = np.mean(x_train, axis=0)
    sd = np.std(x_train, axis=0, ddof=0)
    _check_scale(width, "training-range")
    _check_scale(sd, "mixture-CDF")
    return lower, width, mean, sd


def apply_mixture_cdf(x, parameters, jacobian=False):
    lower, width, mean, sd = parameters
    uniform_cdf = np.clip((x - lower) / width, 0.0, 1.0)
    q = (x - mean) / sd
    u = UNIFORM_WEIGHT * uniform_cdf + NORMAL_WEIGHT * ndtr(q)
    u = np.clip(u, CDF_EPS, 1.0 - CDF_EPS)
    if not jacobian:
        return u
    inside = (x >= lower) & (x <= lower + width)
    log_uniform = np.where(
        inside, np.log(UNIFORM_WEIGHT) - np.log(width), -np.inf
    )
    log_normal = (
        np.log(NORMAL_WEIGHT) - np.log(sd)
        - 0.5 * q**2 - 0.5 * np.log(2.0 * np.pi)
    )
    return u, np.logaddexp(log_uniform, log_normal).sum(axis=1)


def transform_train_test(x_train, x_test, method):
    if method == "normal":
        parameters = fit_normal_cdf(x_train)
        apply = apply_normal_cdf
    elif method == "mixture":
        parameters = fit_mixture_cdf(x_train)
        apply = apply_mixture_cdf
    else:
        raise ValueError(f"Unknown transform: {method}")
    return apply(x_train, parameters), apply(x_test, parameters, jacobian=True)
