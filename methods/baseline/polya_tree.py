"""Sparse classical Polya tree on the unit hypercube."""

from collections import Counter
import numpy as np


class ClassicalPolyaTree:
    """Midpoint PT; level k uses Dirichlet alpha_scale * k**2."""

    def __init__(self, max_depth=15, alpha_scale=0.1):
        if max_depth < 1 or max_depth > 16:
            raise ValueError("max_depth must be between 1 and 16")
        if not np.isfinite(alpha_scale) or alpha_scale <= 0:
            raise ValueError("alpha_scale must be positive")
        self.max_depth = int(max_depth)
        self.alpha_scale = float(alpha_scale)
        self.dimension_ = None
        self.n_train_ = None
        self.counts_by_level_ = None

    @staticmethod
    def _validate_unit_data(x, expected_dimension=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("x must be a non-empty two-dimensional array")
        if expected_dimension is not None and x.shape[1] != expected_dimension:
            raise ValueError("x has a different dimension from the training data")
        if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
            raise ValueError("all coordinates must lie in [0, 1]")
        return np.minimum(x, np.nextafter(1.0, 0.0))

    def _cell_keys_by_level(self, x):
        # Each coarser midpoint cell is a prefix of the finest cell's bits.
        finest = np.asarray(x * (1 << self.max_depth), dtype=np.uint16, order="C")
        row_key = np.dtype((np.void, self.dimension_ * finest.itemsize))
        for shift in reversed(range(self.max_depth)):
            cells = finest >> shift
            # Read each contiguous row as one byte key, including trailing zeros.
            yield np.ndarray(len(cells), dtype=row_key, buffer=cells).tolist()

    def fit(self, x):
        x = self._validate_unit_data(x)
        self.dimension_ = x.shape[1]
        self.n_train_ = len(x)
        self.counts_by_level_ = [
            Counter(keys) for keys in self._cell_keys_by_level(x)
        ]
        return self

    def score_samples(self, x):
        if self.counts_by_level_ is None:
            raise RuntimeError("fit must be called before score_samples")
        x = self._validate_unit_data(x, self.dimension_)
        log_density = np.zeros(len(x), dtype=np.float64)
        parent_counts = np.full(len(x), self.n_train_, dtype=np.float64)
        d_log_2 = self.dimension_ * np.log(2.0)

        for level, keys in enumerate(self._cell_keys_by_level(x), start=1):
            counts = self.counts_by_level_[level - 1]
            child_counts = np.fromiter(
                (counts.get(key, 0) for key in keys),
                dtype=np.float64,
                count=len(keys),
            )
            alpha = self.alpha_scale * level**2
            log_parent = np.full_like(parent_counts, -np.inf)
            positive = parent_counts > 0
            log_parent[positive] = np.log(parent_counts[positive])
            log_denominator = np.logaddexp(
                log_parent, np.log(alpha) + d_log_2
            )
            log_density += np.log(child_counts + alpha) - log_denominator
            parent_counts = child_counts

        # A depth-K leaf has volume 2**(-d*K).
        log_density += self.max_depth * d_log_2
        return log_density
