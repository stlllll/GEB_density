# -*- coding: utf-8 -*-
"""GEB tree nodes and JAX kernels for scoring candidate splits."""

import jax

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln


def _log_split_Bayes_factor(
    n_obs, splits, split_dims, n_left, lower, upper, prior_mass
):
    """Return log split Bayes factors for all candidate cuts on JAX arrays."""
    prior_mean_left = (splits - lower[split_dims]) / (
        upper[split_dims] - lower[split_dims]
    )
    prior_mean_right = 1.0 - prior_mean_left
    n_right = n_obs - n_left

    prior_left = prior_mean_left * prior_mass
    prior_right = prior_mean_right * prior_mass
    posterior_left = prior_left + n_left
    posterior_right = prior_right + n_right

    a = gammaln(prior_left + prior_right) - (
        gammaln(prior_left) + gammaln(prior_right)
    )
    b = gammaln(posterior_left + posterior_right) - (
        gammaln(posterior_left) + gammaln(posterior_right)
    )
    c = -n_left * jnp.log(prior_mean_left) - n_right * jnp.log(
        prior_mean_right
    )
    return a - b + c


@jax.jit
def _grid_left_counts(padded_dataset, n_obs, grid_points, lower, upper):
    """Count points at or below each grid cut using bin counts and cumulative sums."""
    n_grid = grid_points.shape[1]

    scaled = (
        (padded_dataset - lower[None, :])
        / (upper - lower)[None, :]
        * jnp.asarray(n_grid + 1, dtype=padded_dataset.dtype)
    )
    first_grid_index = jnp.clip(
        jnp.ceil(scaled).astype(jnp.int32) - 1, 0, n_grid
    )

    # Check neighboring cuts to correct rounding near bin boundaries.
    dimensions = jnp.arange(padded_dataset.shape[1])[None, :]
    previous = grid_points[
        dimensions, jnp.clip(first_grid_index - 1, 0, n_grid - 1)
    ]
    first_grid_index = jnp.where(
        (first_grid_index > 0) & (padded_dataset <= previous),
        first_grid_index - 1,
        first_grid_index,
    )
    current = grid_points[
        dimensions, jnp.clip(first_grid_index, 0, n_grid - 1)
    ]
    first_grid_index = jnp.where(
        (first_grid_index < n_grid) & (padded_dataset > current),
        first_grid_index + 1,
        first_grid_index,
    )

    # Send padded rows to the bin excluded from the cumulative counts.
    real_row = jnp.arange(padded_dataset.shape[0]) < n_obs
    first_grid_index = jnp.where(real_row[:, None], first_grid_index, n_grid).T

    histograms = jax.vmap(
        lambda indices: jnp.bincount(indices, length=n_grid + 1)
    )(first_grid_index)
    n_left = jnp.cumsum(histograms, axis=1)[:, :n_grid]

    return n_left


@jax.jit
def _best_grid_split(
    padded_dataset, n_obs, grid_points, lower, upper, prior_mass
):
    """Count and score every regular-grid candidate entirely in float32 JAX."""
    n_left = _grid_left_counts(
        padded_dataset, n_obs, grid_points, lower, upper
    )
    split_dims = jnp.broadcast_to(
        jnp.arange(padded_dataset.shape[1], dtype=jnp.int32)[:, None],
        grid_points.shape,
    )
    scores = _log_split_Bayes_factor(
        n_obs,
        grid_points,
        split_dims,
        n_left,
        lower,
        upper,
        prior_mass,
    )
    # Both children must contain at least one observation.
    valid = (n_left >= 1) & (n_left <= n_obs - 1)
    scores = jnp.where(valid, scores, -jnp.inf)
    flat_index = jnp.argmax(scores)
    n_grid = grid_points.shape[1]
    split_dim = flat_index // n_grid
    grid_index = flat_index % n_grid
    return (
        split_dim,
        grid_points[split_dim, grid_index],
        scores.reshape(-1)[flat_index],
    )


@jax.jit
def _best_observed_split(dataset, lower, upper, prior_mass):
    """Score observed cuts and their left limits using actual routed counts."""
    n_obs, dimension_x = dataset.shape
    sorted_data = jnp.sort(dataset, axis=0)
    splits = jnp.concatenate(
        (sorted_data, jnp.nextafter(sorted_data, -jnp.inf)), axis=0
    )
    n_left = jax.vmap(
        lambda values, cuts: jnp.searchsorted(values, cuts, side="right"),
        in_axes=(1, 1), out_axes=1,
    )(sorted_data, splits)
    split_dims = jnp.broadcast_to(
        jnp.arange(dimension_x, dtype=jnp.int32)[None, :],
        splits.shape,
    )
    scores = _log_split_Bayes_factor(
        n_obs,
        splits,
        split_dims,
        n_left,
        lower,
        upper,
        prior_mass,
    )
    valid = (
        (n_left >= 1) & (n_left <= n_obs - 1)
        & (splits > lower[None, :]) & (splits < upper[None, :])
    )
    scores = jnp.where(valid, scores, -jnp.inf)
    flat_index = jnp.argmax(scores)
    split_dim = flat_index % dimension_x
    row = flat_index // dimension_x
    return split_dim, splits[row, split_dim], scores.reshape(-1)[flat_index]


class GEBTreeNode:
    """A cell A with prior mass alpha * H(A) and its candidate split."""

    def __init__(
        self,
        depth=None,
        dimension_x=None,
        prior_mean=None,
        prior_mass=None,
        left_child=None,
        right_child=None,
        lower=None,
        upper=None,
        N0=0,
        N1=0,
        N=0,
        grid=True,
        m=32,
    ):
        self.depth = depth
        self.dimension_x = dimension_x
        self.prior_mean = prior_mean
        self.prior_mass = np.float32(prior_mass)
        self.left_child = left_child
        self.right_child = right_child
        self.lower = np.asarray(lower, dtype=np.float32) if lower is not None else None
        self.upper = np.asarray(upper, dtype=np.float32) if upper is not None else None
        self.split = None
        self.split_dim = None
        self.N = N
        self.N0 = N0
        self.N1 = N1
        self.grid = grid
        self.m = m
        self.root_log_marginal_likelihood = 0
        self.log_split_Bayes_factor = 0

    def partition(self, dataset):
        """Create both children and return the mask assigning observations left."""
        if self.left_child is not None:
            return

        left_mask = dataset[:, self.split_dim] <= self.split
        self.N0 = int(np.sum(left_mask))
        self.N1 = len(dataset) - self.N0
        upper_left = self.upper.copy()
        lower_right = self.lower.copy()
        upper_left[self.split_dim] = self.split
        lower_right[self.split_dim] = self.split

        prior_mean_left = (self.split - self.lower[self.split_dim]) / (
            self.upper[self.split_dim] - self.lower[self.split_dim]
        )
        self.prior_mean = np.array(
            [prior_mean_left, 1.0 - prior_mean_left], dtype=np.float32
        )

        common = dict(
            depth=self.depth + 1,
            dimension_x=self.dimension_x,
            prior_mean=None,
            left_child=None,
            right_child=None,
            N0=0,
            N1=0,
            grid=self.grid,
            m=self.m,
        )
        self.left_child = GEBTreeNode(
            **common,
            prior_mass=self.prior_mass * self.prior_mean[0],
            lower=self.lower,
            upper=upper_left,
            N=self.N0,
        )
        self.right_child = GEBTreeNode(
            **common,
            prior_mass=self.prior_mass * self.prior_mean[1],
            lower=lower_right,
            upper=self.upper,
            N=self.N1,
        )
        return left_mask

    def split_candidate(self, dataset):
        """Return the best cut dimension, cut location, log Bayes factor, and count."""
        dataset = np.asarray(dataset, dtype=np.float32)
        n_obs = len(dataset)
        if n_obs <= 2:
            return 0, 0, -np.inf, n_obs

        lower = jnp.asarray(self.lower)
        upper = jnp.asarray(self.upper)
        if self.grid:
            if self.m < 2:
                return 0, 0, -np.inf, n_obs

            # Pad to a power of two so nodes can reuse JAX compilations.
            bucket_size = 1 << (n_obs - 1).bit_length()
            if bucket_size == n_obs:
                padded = dataset
            else:
                padded = np.empty(
                    (bucket_size, self.dimension_x), dtype=np.float32
                )
                padded[:n_obs] = dataset
                padded[n_obs:] = self.upper
            grid_points = np.linspace(
                self.lower, self.upper, self.m + 1, axis=-1, dtype=np.float32
            )[:, 1:-1]
            dim_max, div_max, log_bf_max = _best_grid_split(
                jnp.asarray(padded),
                jnp.asarray(n_obs, dtype=jnp.int32),
                jnp.asarray(grid_points),
                lower,
                upper,
                self.prior_mass,
            )
        else:
            dim_max, div_max, log_bf_max = _best_observed_split(
                jnp.asarray(dataset), lower, upper, self.prior_mass
            )

        # Converting the three scalar results together is a single synchronization.
        dim_max, div_max, log_bf_max = jax.device_get((dim_max, div_max, log_bf_max))
        return int(dim_max), np.float32(div_max), np.float32(log_bf_max), n_obs

    def is_leaf(self):
        return self.left_child is None

    def __repr__(self):
        return (
            f"GEBTreeNode(depth={self.depth}, children={0 if self.is_leaf() else 2}, "
            f"partition dimension={self.split_dim}, partition boundary={self.split}, "
            f"prior_mass={self.prior_mass}, lower={self.lower}, "
            f"upper={self.upper}, N0 = {self.N0}, N1={self.N1})"
        )
