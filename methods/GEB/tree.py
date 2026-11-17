# -*- coding: utf-8 -*-
"""Greedy empirical Bayes density tree with JAX split kernels."""

import numpy as np
from scipy.special import ndtr, ndtri

from .node import GEBTreeNode


# Fixed seed for reproducible predictive sampling.
rng = np.random.RandomState(0)


class GEBTree:
    """Greedy empirical Bayes tree with PPB concentration alpha and grid size m."""
    def __init__(
        self,
        dimension_x=None,
        lower_global=None,
        upper_global=None,
        alpha=None,
        dataset=None,
        grid=True,
        m=32,
    ):
        if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m < 2:
            raise ValueError("m must be an integer >= 2 (number of grid subintervals)")
        self.m = int(m)
        self.grid = bool(grid)
        self.dimension_x = dimension_x
        self.lower_global = np.asarray(lower_global, dtype=np.float32)
        self.upper_global = np.asarray(upper_global, dtype=np.float32)
        self.alpha = np.float32(alpha)
        self._flat_tree = None

        self.root = GEBTreeNode(
            depth=0,
            dimension_x=dimension_x,
            prior_mean=None,
            prior_mass=alpha,
            left_child=None,
            right_child=None,
            lower=self.lower_global,
            upper=self.upper_global,
            N0=0,
            N1=0,
            grid=grid,
            m=m,
        )

        dataset = np.asarray(dataset, dtype=np.float32)
        in_bounds = np.all(
            (dataset >= self.lower_global) & (dataset <= self.upper_global), axis=1
        )
        dataset = dataset[in_bounds]
        (
            self.root.split_dim,
            self.root.split,
            self.root.log_split_Bayes_factor,
            self.root.N,
        ) = self.root.split_candidate(dataset)

        if self.root.log_split_Bayes_factor > 0:
            self.grow_tree(self.root, dataset)

    def grow_tree(self, node, dataset):
        """Grow descendants with positive log split Bayes factors."""
        work = [(node, np.asarray(dataset, dtype=np.float32))]
        while work:
            current, current_data = work.pop()
            left_mask = current.partition(current_data)
            left = current.left_child
            right = current.right_child

            # Reuse the mask that counted the children; equality goes left.
            left_data = current_data[left_mask]
            right_data = current_data[~left_mask]

            (
                left.split_dim,
                left.split,
                left.log_split_Bayes_factor,
                left.N,
            ) = left.split_candidate(left_data)
            (
                right.split_dim,
                right.split,
                right.log_split_Bayes_factor,
                right.N,
            ) = right.split_candidate(right_data)

            # Push right first so the traversal order remains left-before-right.
            if right.log_split_Bayes_factor > 0:
                work.append((right, right_data))
            if left.log_split_Bayes_factor > 0:
                work.append((left, left_data))

        self._flat_tree = None

    def _nodes_preorder(self, start=None):
        nodes = []
        stack = [self.root if start is None else start]
        while stack:
            node = stack.pop()
            nodes.append(node)
            if not node.is_leaf():
                stack.append(node.right_child)
                stack.append(node.left_child)
        return nodes

    def _build_flat_tree(self):
        nodes = self._nodes_preorder()
        index = {id(node): i for i, node in enumerate(nodes)}
        count = len(nodes)
        split_dims = np.full(count, -1, dtype=np.int64)
        splits = np.zeros(count, dtype=np.float32)
        left_indices = np.full(count, -1, dtype=np.int64)
        right_indices = np.full(count, -1, dtype=np.int64)
        densities = np.zeros(count, dtype=np.float32)
        is_leaf = np.zeros(count, dtype=bool)

        for i, node in enumerate(nodes):
            is_leaf[i] = node.is_leaf()
            if is_leaf[i]:
                densities[i] = node.density
            else:
                split_dims[i] = node.split_dim
                splits[i] = node.split
                left_indices[i] = index[id(node.left_child)]
                right_indices[i] = index[id(node.right_child)]

        self._flat_tree = (
            split_dims,
            splits,
            left_indices,
            right_indices,
            densities,
            is_leaf,
        )

    def tree_joint_density_set(self, node):
        for current in self._nodes_preorder(node):
            if current.is_leaf():
                area = np.prod(current.upper - current.lower)
                prior = area * self.alpha
                current.density = np.float32(
                    ((prior + current.N) / (self.root.N + self.alpha)) / area
                )
        if node is self.root:
            self._build_flat_tree()

    def tree_joint_density_batch(self, points):
        """Evaluate the posterior mean at an (n, d) array of points."""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != self.dimension_x:
            raise ValueError("Expected one row per point and one column per dimension")
        if self._flat_tree is None:
            self.tree_joint_density_set(self.root)
        dims, splits, left, right, densities, is_leaf = self._flat_tree
        inside = np.all((points >= self.lower_global) & (points <= self.upper_global), axis=1)
        # Preserve float32 predictions in bounds; use float64 when zero
        # densities outside the support may need a 1e-300 floor.
        result = np.zeros(len(points), dtype=densities.dtype if inside.all() else np.float64)
        stack = [(0, np.flatnonzero(inside))]
        while stack:
            node, rows = stack.pop()
            if is_leaf[node]:
                result[rows] = densities[node]
            elif rows.size:
                goes_left = points[rows, dims[node]] <= splits[node]
                stack.append((left[node], rows[goes_left]))
                stack.append((right[node], rows[~goes_left]))
        return result

    def tree_sampling(self, n, random_state=None):
        """Draw observations from the PPB posterior predictive density."""
        if n < 0:
            raise ValueError("n must be non-negative")
        upper, lower, weights = self.collect_leaves()
        if n == 0:
            return np.empty((0, self.dimension_x), dtype=np.float32)
        probabilities = weights / weights.sum()
        generator = rng if random_state is None else np.random.default_rng(random_state)
        leaf_indices = generator.choice(len(weights), size=n, p=probabilities)
        return generator.uniform(lower[leaf_indices], upper[leaf_indices]).astype(np.float32)

    def draw_tree(self, filename="GEB_tree"):
        try:
            from graphviz import Digraph
        except ImportError as exc:
            raise ImportError(
                "draw_tree requires the optional 'graphviz' Python package"
            ) from exc
        dot = Digraph(format="png")
        self._add_node_to_graph(dot, self.root)
        dot.render(filename, view=True)

    def _add_node_to_graph(self, dot, node, parent_id=None):
        if node is None:
            return
        node_id = f"Node_{id(node)}"
        split_text = "leaf" if node.is_leaf() else f"{node.split:.6f}"
        label = (
            f"depth: {node.depth}\nsplit Dim: {node.split_dim}\n"
            f"split: {split_text}\nPrior mass: {node.prior_mass:.2f}"
        )
        dot.node(node_id, label)
        if parent_id is not None:
            dot.edge(parent_id, node_id)
        if node.left_child is not None:
            self._add_node_to_graph(dot, node.left_child, node_id)
        if node.right_child is not None:
            self._add_node_to_graph(dot, node.right_child, node_id)

    def init_log_marginal_likelihood(self, node):
        stack = [(node, False)]
        while stack:
            current, visited = stack.pop()
            if current.is_leaf():
                current.root_log_marginal_likelihood = 0
            elif visited:
                current.root_log_marginal_likelihood = (
                    current.log_split_Bayes_factor
                    + current.left_child.root_log_marginal_likelihood
                    + current.right_child.root_log_marginal_likelihood
                )
            else:
                stack.append((current, True))
                stack.append((current.right_child, False))
                stack.append((current.left_child, False))

    def collect_leaves(self):
        """Return leaf upper bounds, lower bounds, and posterior parameters."""
        upper, lower, weights = [], [], []
        stack = [self.root]
        while stack:
            current = stack.pop()
            if current.is_leaf():
                upper.append(current.upper)
                lower.append(current.lower)
                weights.append(current.prior_mass + current.N)
            else:
                stack.append(current.right_child)
                stack.append(current.left_child)
        return np.asarray(upper), np.asarray(lower), np.asarray(weights)

    def leaf_posterior(self):
        """Return leaf bounds, volumes, and conditional Dirichlet parameters.

        The tree partition is held fixed at its empirical-Bayes estimate.  The
        prior mass of a leaf is ``alpha * volume`` on the unit cube, so its
        posterior parameter is that mass plus the number of observations in
        the leaf.
        """
        upper, lower, alpha = self.collect_leaves()
        upper = np.asarray(upper, dtype=np.float64)
        lower = np.asarray(lower, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        volume = np.prod(upper - lower, axis=1, dtype=np.float64)
        if np.any(volume <= 0) or np.any(alpha <= 0):
            raise ValueError("Leaves must have positive volume and posterior mass")
        return lower, upper, volume, alpha

    def sample_integrated_squared_density(
        self, n_draws, normal_sd=None, random_state=None,
    ):
        """Draw ISD values using analytic leaf integrals.

        For a Gaussian CDF transform, pass the training standard deviations
        as ``normal_sd`` to integrate on the original scale. Otherwise,
        integrate the piecewise-constant density on the tree's support.
        """
        if n_draws < 1:
            raise ValueError("n_draws must be positive")
        lower, upper, volume, alpha = self.leaf_posterior()
        if normal_sd is None:
            coefficients = 1.0 / volume
        else:
            sd = np.broadcast_to(np.asarray(normal_sd, dtype=float), (self.dimension_x,))
            if np.any(~np.isfinite(sd)) or np.any(sd <= 0):
                raise ValueError("normal_sd must be positive and finite")
            if np.any(lower < 0) or np.any(upper > 1):
                raise ValueError("Gaussian-transformed leaves must lie in [0, 1]")
            z_lower, z_upper = np.sqrt(2.0) * ndtri(lower), np.sqrt(2.0) * ndtri(upper)
            # Use survival probabilities in the right tail to avoid cancellation.
            probability = np.where(
                z_lower >= 0,
                ndtr(-z_lower) - ndtr(-z_upper),
                ndtr(z_upper) - ndtr(z_lower),
            )
            integral = np.prod(probability / (2.0 * np.sqrt(np.pi) * sd), axis=1)
            coefficients = integral / volume**2
        weights = np.random.default_rng(random_state).dirichlet(alpha, size=n_draws)
        return np.einsum("ij,j,ij->i", weights, coefficients, weights)
