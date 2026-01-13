from __future__ import annotations

from typing import Optional, Union, Literal

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


TransitionNorm = Literal["random_walk", "symmetric"]
Propagation = Literal["rwr", "heat_kernel", "laplacian_solve"]
SolveMode = Literal["iter", "closed_form"]


def diffuse_network_mutation_profiles(
        G: nx.Graph,
        data: Union[str, pd.DataFrame],
        # --- transition operator ---
        transition_norm: TransitionNorm = "random_walk",
        weight: Optional[str] = None,
        # --- propagation choice ---
        propagation: Propagation = "rwr",
        alpha: float = 0.7,  # RWR
        t: float = 1.0,  # heat kernel
        lam: float = 1.0,  # for Laplacian solve
        # --- solver controls ---
        solve_mode: SolveMode = "iter",
        max_iter: int = 200,
        tol: float = 1e-8,
        chunksize: int = 64,
        dtype: Union[str, np.dtype] = np.float64,
) -> pd.DataFrame:
    """
    Compute diffused (network-propagated) profiles for each sample.

    Given a feature network ``G`` (e.g., an undirected PPI) and an input feature matrix
    X (features x samples, often binary 0/1, e.g., mutations) given by ``data``, compute the
    diffused / network-smoothed profiles matrix ``X_out`` (same shape/order) for each sample
    by propagating signal along network edges.

    Parameters
    ----------
    G : nx.Graph
        Feature network (e.g., nodes are genes).
        Node IDs must match the IDs in the input ``data`` matrix row index.
    data : {`str`, `pd.DataFrame`}
        Feature-by-sample data matrix (rows=features, columns=samples)
    transition_norm : {"random_walk", "symmetric"}, default="random_walk"
        How to normalize adjacency into the transition matrix / diffusion operator W.

        Let A be adjacency and D the diagonal degree matrix (D_ii = sum_j A_ij).
        We construct a sparse operator W used via left-multiplication: W @ X.

        transition_norm options:

        - "random_walk": Computes the random-walk transition P = D^{-1}A.
                         The random walk with restart (RWR) update rule is implemented
                         using the operator W := P^T:
                         X_{t+1} = (1-α) X0 + α P^T X_t.
                         Recommended default choice for network propagation of sparse binary mutations.
        - "symmetric": Computes the symmetric normalized adjacency: W = D^{-1/2} A D^{-1/2}.
                       Recommended if a symmetric operator is wanted (often nicer mathematically / numerically).
                       Sometimes preferable because it can reduce some degree-driven bias compared with pure
                       random-walk operators.
    weight : {`str`, `None`}, default=None
        Edge attribute name in G to use as weights. If None, edges are treated as weight=1.
    propagation : {"rwr", "heat_kernel", "laplacian_solve"}, default="rwr"
        Which diffusion model to use.

        propagation options

        - "rwr" : Random Walk with Restart
            Update rule:
                X_{k+1} = (1-α) X0 + α W X_k
            Closed form (stationary solution):
                X* = (1-α) (I - αW)^{-1} X0

            Interpretation:
            - Each step mixes an “anchor to original mutation profile” term (X0) with a
              “spread across network” term (W X_t).
            - α controls locality: smaller α -> more local; larger α -> more global smoothing.

            Examples in literature:
            - Network-based stratification / network propagation pipelines (e.g., NBS-style smoothing)
            - Classic “network propagation” for biological signal diffusion

            Recommendation:
            - Default for smoothing sparse binary mutation profiles prior to similarity/clustering.
            - Start α ≈ 0.7; sweep 0.5–0.85 for sensitivity.
            - Set transition_norm = "random_walk"

            Uses parameters: alpha, solve_mode, (max_iter,tol if solve_mode="iter"),
                             (chunksize if solve_mode="closed_form")
            Ignores parameters: beta, t, lam
        - "heat_kernel" : Heat-kernel diffusion
            Continuous-time diffusion:
                X*(t) = exp(-t L) X0, with L = I - W

            Recommendation:
            - Use if you want a “diffusion time” knob t with a multi-scale interpretation.
            - Often explored by scanning t across scales.
            - Set transition_norm = "symmetric"

            Uses parameters: t
            Ignores parameters: alpha, beta, lam, solve_mode, max_iter, tol, chunksize
        - "laplacian_solve" Laplacian-regularized smoothing
            Linear solve:
                X* = (I + λL)^{-1} X0, with L = I - W

            Equivalent optimization view:
                argmin_X  ||X - X0||_F^2 + λ * Tr(X^T L X)

            Recommendation:
            - Use when you prefer a regularization knob λ and a single sparse solve.
            - Often stable and efficient for large graphs and many samples.
            - Set transition_norm = "symmetric"

            Uses parameters: lam, chunksize
            Ignores parameters: alpha, beta, t, solve_mode, max_iter, tol
    alpha : `float`, default=0.7
        RWR diffusion strength (restart complement). Must satisfy 0 <= alpha < 1.
        Larger alpha => more diffusion; smaller alpha => more anchored to X0 (less diffusion).
        Used only if propagation="rwr". Ignored otherwise.

        Practical guidance based on literature (binary mutations on undirected PPI):
        - alpha ~ 0.7 is a strong default
        - sweep 0.5–0.85 for sensitivity
    t : `float`, default=1.0
        Heat-kernel diffusion time (t >= 0).
        Used only if propagation="heat_kernel". Ignored otherwise.
    lam : `float`, default=1.0
        Laplacian smoothing strength (lambda >= 0).
        Used only if propagation="laplacian_solve". Ignored otherwise.
    solve_mode : {"iter", "closed_form"}, default="iter"
        How to compute stationary solutions for restart diffusion.
        Used only if propagation="rwr". Ignored otherwise.
        Note: propagation="heat_kernel" uses expm_multiply and
        propagation="laplacian_solve" uses a sparse LU solver

        options:

        - "iter": fixed-point iteration to convergence
        - "closed_form": sparse LU solve of the corresponding linear system
                         (I - αW) X* = (1-α) X0
    max_iter : `int`, default=200
        Max iterations for solve_mode="iter".
        Used only when propagation="rwr" AND solve_mode="iter".
        Ignored otherwise.
    tol : `float`, default=1e-8
        Convergence tolerance for solve_mode="iter".
        Used only when propagation="rwr" AND solve_mode="iter".
        Ignored otherwise.
    chunksize : `int`, default=64
        RHS block size (number of samples per block) for LU-based multi-RHS solves.
        Used when:
          - propagation="rwr" AND solve_mode="closed_form"
          - propagation="laplacian_solve"
        Ignored otherwise.
    dtype : {numpy dtype, `str`}, default=np.float64
        Numeric dtype used for computations.

    Returns
    -------
    X_out : `pd.DataFrame`
        Diffused feature-by-sample matrix with features (e.g., genes) and samples ordered exactly as in input.

    Notes
    -----
    Recommended defaults for undirected PPI + binary mutation matrix:
        propagation="rwr", transition_norm="random_walk", alpha=0.7, solve_mode="iter"

    Examples
    --------
    RWR default:
        df_out = diffuse_network_mutation_profiles(G, df, propagation="rwr", alpha=0.7)

    Symmetric operator + RWR:
        df_out = diffuse_network_mutation_profiles(G, df, transition_norm="symmetric", propagation="rwr", alpha=0.7)

    Heat kernel:
        df_out = diffuse_network_mutation_profiles(G, df, propagation="heat_kernel", t=1.0)
    """
    feats = data.index.to_list()
    samples = data.columns.to_list()

    if set(feats) != set(G):
        raise ValueError("Feature IDs in the data must match node names in the graph.")

    X_full = data.to_numpy(dtype=dtype, copy=True)

    feat_to_idx = {g: i for i, g in enumerate(feats)}
    idx = np.fromiter((feat_to_idx[g] for g in feats), dtype=int, count=len(feats))
    X0 = X_full[idx, :]  # (n_feats x n_samples)

    W = _build_operator_W(
        G=G,
        feats_order=feats,
        transition_norm=transition_norm,
        weight=weight,
        dtype=dtype,
    )

    # Parameter validation
    if propagation == "rwr" and not (0.0 <= alpha < 1.0):
        raise ValueError("For propagation='rwr', require 0 <= alpha < 1.")
    if propagation == "heat_kernel" and t < 0:
        raise ValueError("For propagation='heat_kernel', require t >= 0.")
    if propagation == "laplacian_solve" and lam < 0:
        raise ValueError("For propagation='laplacian_solve', require lam >= 0.")

    # Compute diffusion
    if propagation == "rwr":
        if solve_mode == "iter":
            X_star = _fixed_point_rwr(W=W, X0=X0, alpha=alpha, max_iter=max_iter, tol=tol)
        elif solve_mode == "closed_form":
            n = W.shape[0]
            M = sp.eye(n, format="csr", dtype=dtype) - (alpha * W)     # (I - αW)
            B = (1.0 - alpha) * X0                                    # (1-α)X0
            X_star = _closed_form_solve(M=M, B=B, chunksize=chunksize)
        else:
            raise ValueError(f"Unknown solve_mode: {solve_mode}")

    elif propagation == "heat_kernel":
        n = W.shape[0]
        L = sp.eye(n, format="csr", dtype=dtype) - W   # L = I - W
        X_star = spla.expm_multiply((-t) * L, X0)      # exp(-tL)X0

    elif propagation == "laplacian_solve":
        n = W.shape[0]
        L = sp.eye(n, format="csr", dtype=dtype) - W   # L = I - W
        M = sp.eye(n, format="csr", dtype=dtype) + (lam * L)  # I + λL
        X_star = _closed_form_solve(M=M, B=X0, chunksize=chunksize)

    else:
        raise ValueError(f"Unknown propagation: {propagation}")

    X_out = X_full
    X_out[idx, :] = X_star
    X_out = pd.DataFrame(X_out, index=feats, columns=samples)
    return X_out


# -----------------------------
# Helpers
# -----------------------------
def _build_operator_W(
    G: nx.Graph,
    feats_order: list[str],
    transition_norm: TransitionNorm = "random_walk",
    weight: Optional[str] = None,
    dtype: Union[str, np.dtype] = np.float64,
) -> sp.csr_matrix:
    """
    Build the sparse graph diffusion operator W over a specified feature ordering.

    This constructs adjacency A in the *exact* order `featss_order`, then normalizes it into W.

    Parameters
    ----------
    G: `nx.Graph`
        Graph containing the features in `feats_order`.
    feats_order:
        List of features (e.g., genes) IDs defining the matrix row/column order.
    transition_norm : {"random_walk", "symmetric"}, default="random_walk"
        How to normalize adjacency into the transition matrix / diffusion operator W.

        Let A be adjacency and D the diagonal degree matrix (D_ii = sum_j A_ij).
        We construct a sparse operator W used via left-multiplication: W @ X.

        options:

        - "random_walk": Computes the random-walk transition P := D^{-1}A.
                         The random walk with restart (RWR) update rule is implemented
                         using the operator W := P^T:
                         X_{t+1} = (1-α) X0 + α W X_t.
                         Recommended default choice for network propagation of sparse binary mutations.
        - "symmetric": Computes the symmetric normalized adjacency: W = D^{-1/2} A D^{-1/2}.
                       Recommended if a symmetric operator is wanted (often nicer mathematically / numerically).
                       Sometimes preferable because it can reduce some degree-driven bias compared with pure
                       random-walk operators.
    weight : {`str`, `None`}, default=None
        Edge attribute name in G to use as weights. If None, edges are treated as weight=1.
    dtype : {numpy dtype, `str`}, default=np.float64
        Numeric dtype used for computations.

    Returns
    -------
    W : `scipy.sparse.csr_matrix`
        W with shape (n_feats, n_feats), CSR format.

    Notes
    -----
    - Self-loops are removed from the adjacency by default (diagonal set to 0).
    - Isolated nodes (degree 0) will not propagate; restart-based diffusions keep them anchored.
    """
    A = nx.to_scipy_sparse_array(G, nodelist=feats_order, weight=weight, format="csr", dtype=dtype)

    # Remove self loops unless intentionally included
    A = A.copy()
    A.setdiag(0)
    A.eliminate_zeros()

    d = np.asarray(A.sum(axis=1)).ravel()

    if transition_norm == "random_walk":
        inv_d = np.zeros_like(d, dtype=np.float64)
        nz = d > 0
        inv_d[nz] = 1.0 / d[nz]
        Dinv = sp.diags(inv_d.astype(dtype))
        P = Dinv @ A          # D^{-1}A
        W = P.T.tocsr()       # P^T
    elif transition_norm == "symmetric":
        inv_sqrt = np.zeros_like(d, dtype=np.float64)
        nz = d > 0
        inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
        Dmhalf = sp.diags(inv_sqrt.astype(dtype))
        W = (Dmhalf @ A @ Dmhalf).tocsr()
    else:
        raise ValueError(f"Unknown transition_norm: {transition_norm}")

    return W


def _fixed_point_rwr(
    W: sp.csr_matrix,
    X0: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """
    Fixed-point iteration solver for Random Walk with Restart (RWR) diffusion.

    Model
    -----
    Update rule:
        X_{t+1} = (1-α) X0 + α W X_t

    Stationary solution (if it converges):
        X* = (1-α) (I - αW)^{-1} X0

    Parameters
    ----------
    W: `csr_matrix`
        Sparse diffusion operator (n x n).
    X0: `np.ndarray`
        Seed matrix (n x m) with m samples.
    alpha: `float`
        0 <= α < 1. Larger α => more diffusion.
    max_iter: `int`
        Maximum iterations.
    tol: `float`
        Relative convergence criterion on Frobenius norm:
            ||X_new - X|| / (||X|| + eps) < tol

    Returns
    -------
    X : `np.ndarray`
        Diffused profiles X* (n x m).
    """
    X = X0.copy()
    one_minus = 1.0 - alpha

    for _ in range(max_iter):
        X_new = one_minus * X0 + alpha * (W @ X)
        denom = np.linalg.norm(X) + 1e-12
        if np.linalg.norm(X_new - X) / denom < tol:
            return X_new
        X = X_new

    return X


def _closed_form_solve(M: sp.csr_matrix, B: np.ndarray, chunksize: int) -> np.ndarray:
    """
    Solve the sparse linear system M X = B for multiple RHS columns via one LU factorization.

    Used for closed-form stationary solutions:
      - RWR:       (I - αW) X* = (1-α)X0
      - Laplacian: (I + λL) X* = X0

    Implementation
    --------------
    - Factorize M once using sparse LU (splu).
    - Solve in blocks of RHS columns to limit peak memory.

    Parameters
    ----------
    M: `csr_matrix`
        Sparse square matrix (n x n).
    B: `np.ndarray`
        Dense RHS matrix (n x m).
    chunksize: `int`
        Number of RHS columns to solve at a time.

    Returns
    -------
    X : `np.ndarray`
        Dense solution X with shape (n x m).

    Notes
    -----
    - For extremely large n, might need iterative linear solvers (cg/gmres) + preconditioning.
      This LU approach is simple and usually robust for medium-to-large graphs.
    """
    if M.shape[0] == 0:
        return B

    lu = spla.splu(M.tocsc())
    n, m = B.shape
    X = np.empty((n, m), dtype=B.dtype)

    for j0 in range(0, m, chunksize):
        j1 = min(m, j0 + chunksize)
        X[:, j0:j1] = lu.solve(B[:, j0:j1])

    return X
