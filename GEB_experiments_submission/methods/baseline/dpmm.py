"""Diagonal-covariance variational Dirichlet process mixture."""

import time

from sklearn.mixture import BayesianGaussianMixture

from experiment_utils.config import (
    DPMM_MAX_COMPONENTS,
    DPMM_MAX_ITER,
    DPMM_N_INIT,
    DPMM_REG_COVAR,
    DPMM_TOL,
    DPMM_WEIGHT_CONCENTRATION,
)


def fit_and_score_dpmm(
    x_train, x_test, seed,
    max_components=DPMM_MAX_COMPONENTS,
    max_iter=DPMM_MAX_ITER,
    n_init=DPMM_N_INIT,
    weight_concentration=DPMM_WEIGHT_CONCENTRATION,
    tol=DPMM_TOL,
    reg_covar=DPMM_REG_COVAR,
):
    """Fit VB-DPMM and return sklearn's test log scores and elapsed time."""
    start = time.perf_counter()
    model = BayesianGaussianMixture(
        n_components=min(max_components, len(x_train)),
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=weight_concentration,
        max_iter=max_iter,
        tol=tol,
        reg_covar=reg_covar,
        init_params="kmeans",
        n_init=n_init,
        random_state=seed,
        verbose=0,
    ).fit(x_train)
    return model.score_samples(x_test), time.perf_counter() - start


__all__ = ["fit_and_score_dpmm"]
