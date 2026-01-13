from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd
# from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, spearmanr, pearsonr
import scipy.stats as scstats

from functools import partial
from statsmodels.stats.multitest import multipletests

# -----------------------------------------------------------------------------
# Public type aliases (helps IDEs + keeps the API self-documenting)
# -----------------------------------------------------------------------------
TestType = Literal["MWU", "t-test"]
ReturnType = Literal["dict", "wide"]
ParallelBackend = Literal["processes", "threads", "none"]

# -----------------------------------------------------------------------------
# Multiple testing correction methods supported by statsmodels.multipletests
# (kept explicit so users get a clear error message if they pass something else)
# -----------------------------------------------------------------------------
_ALLOWED_MULTITEST = {
    # one-step correction
    "bonferroni",
    "sidak",
    # step down method
    "holm-sidak",      # step down method using Sidak adjustments
    "holm",            # step-down method using Bonferroni adjustments
    # step-up / closed methods
    "simes-hochberg",  # step-up method (independent)
    "hommel",          # closed method based on Simes tests (non-negative)
    # FDR methods
    "fdr_bh",          # Benjamini/Hochberg (non-negative)
    "fdr_by",          # Benjamini/Yekutieli (negative)
    "fdr_tsbh",        # two stage fdr correction (non-negative)
    "fdr_tsbky",       # two stage fdr correction (non-negative)
}


def _coerce_samples_x_features(
    groups: pd.Series,
    feats: pd.DataFrame,
    *,
    samples_axis: Literal["auto", "index", "columns"] = "auto",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Coerce and align inputs so feature data ``feats`` is shaped as (n_samples, n_features).

    This helper ensures all downstream computations can assume:

    - ``X_df`` is a DataFrame with **samples as rows** and **features as columns**
    - ``X_df.index`` matches ``groups.index`` exactly (same sample IDs, same order)

    It supports two common input conventions:

    1) ``feats`` provided as **samples x features** (samples on index)
    2) ``feats`` provided as **features x samples** (samples on columns), in which case
       it will be transposed.

    Parameters
    ----------
    groups : pandas.Series
        Group labels indexed by sample ID. The sample IDs are used to align
        ``groups`` and ``feats``.
    feats : pandas.DataFrame
        Feature matrix in one of two orientations:

        - samples x features (``feats.index`` are sample IDs, ``feats.columns`` feature names)
        - features x samples (``feats.columns`` are sample IDs, ``feats.index`` feature names)

    samples_axis : {"auto", "index", "columns"}, default="auto"
        Controls how alignment/orientation is determined.

        - "auto": infer orientation from whether ``groups.index`` matches
          ``feats.index`` or ``feats.columns``.
        - "index": enforce that samples are stored on ``feats.index``.
        - "columns": enforce that samples are stored on ``feats.columns``.

    Returns
    -------
    groups_aligned : pandas.Series
        ``groups`` reindexed to match the sample order of the returned feature matrix.
    X_df : pandas.DataFrame
        Feature matrix oriented as samples x features, with:

        - ``X_df.index`` = sample IDs (same order as ``groups_aligned.index``)
        - ``X_df.columns`` = feature names

    Raises
    ------
    TypeError
        If ``groups`` is not a Series or ``feats`` is not a DataFrame.
    ValueError
        If the function cannot determine a consistent alignment between
        ``groups`` and ``feats`` under the given ``samples_axis``.

    Notes
    -----
    This function is intentionally strict: it does **not** silently drop samples.
    If indices do not align, it raises with a clear message rather than
    producing a subtly misaligned analysis.
    """
    if not isinstance(groups, pd.Series):
        raise TypeError("groups must be a pandas Series.")
    if not isinstance(feats, pd.DataFrame):
        raise TypeError("feats must be a pandas DataFrame.")

    if samples_axis == "auto":
        if feats.index.equals(groups.index):
            X_df = feats
        elif feats.columns.equals(groups.index):
            X_df = feats.T
        else:
            common_idx = groups.index.intersection(feats.index)
            common_cols = groups.index.intersection(feats.columns)

            if len(common_idx) == len(groups) and len(common_idx) == feats.shape[0]:
                X_df = feats.loc[groups.index]
            elif len(common_cols) == len(groups) and len(common_cols) == feats.shape[1]:
                X_df = feats.loc[:, groups.index].T
            else:
                raise ValueError(
                    "Could not align groups with feats. Expected either:\n"
                    "  - feats.index == groups.index (samples as rows)\n"
                    "  - feats.columns == groups.index (samples as columns)\n"
                    "No safe strict alignment was found."
                )

    elif samples_axis == "index":
        if not feats.index.equals(groups.index):
            raise ValueError("samples_axis='index' requires feats.index == groups.index.")
        X_df = feats

    elif samples_axis == "columns":
        if not feats.columns.equals(groups.index):
            raise ValueError("samples_axis='columns' requires feats.columns == groups.index.")
        X_df = feats.T

    else:
        raise ValueError("samples_axis must be one of {'auto','index','columns'}.")

    groups_aligned = groups.loc[X_df.index]
    return groups_aligned, X_df


def _feature_chunks(n_features: int, chunk_size: int) -> List[Tuple[int, int]]:
    """
    Partition a number of features into contiguous chunks (half-open intervals).

    Parameters
    ----------
    n_features : int
        Number of features.
    chunk_size : int
        Chunk size (number of features per chunk). Must be positive.

    Returns
    -------
    chunks : list of tuple(int, int)
        List of (start, end) chunk intervals covering [0, n_features).

    Raises
    ------
    ValueError
        If ``chunk_size <= 0``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return [(i, min(i + chunk_size, n_features)) for i in range(0, n_features, chunk_size)]


def _apply_multipletests_safe(
    pvals: np.ndarray,
    *,
    alpha: float,
    method: str,
) -> np.ndarray:
    """
    Apply multiple testing correction with stable behavior in presence of NaNs.

    Parameters
    ----------
    pvals : numpy.ndarray, shape (n_features,)
        Raw p-values. May include NaNs (e.g., insufficient samples, constant vectors).
    alpha : float
        Family-wise error rate / FDR control level passed to statsmodels.
    method : str
        Multipletests method name. See `_validate_multitest_method`.

    Returns
    -------
    qvals : numpy.ndarray, shape (n_features,)
        Corrected p-values. NaN p-values remain NaN in qvals.

    Notes
    -----
    `statsmodels.stats.multitest.multipletests` does not always behave well with NaNs.
    This helper replaces NaNs with 1.0 for correction, then restores NaNs.
    """
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan, dtype=np.float64)

    nan_mask = ~np.isfinite(p)
    p_work = p.copy()
    p_work[nan_mask] = 1.0

    q_work = multipletests(p_work, alpha=alpha, method=method)[1]
    q[:] = q_work
    q[nan_mask] = np.nan
    return q


def mann_whitney_u_test(values1, values2, alternative='two-sided', **kwargs):
    """ Perform the Mann-Whitney U rank test on two independent samples.

    The Mann-Whitney U test is a nonparametric test of the null hypothesis
    that the distribution underlying sample `x` is the same as the
    distribution underlying sample `y`. It is often used as a test of
    difference in location between distributions.

    Computed via ``scipy.stats.mannwhitneyu``.

    Parameters
    ----------
    values1, values2 : array-like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default), which can be
        specified in ``kwargs``.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.
    kwargs : `dict`
        Key-word arguments passed to ``scipy.stats.mannwhitneyu``.

    Returns
    -------
    p_value : `float`
        The p-value.
    """
    u_stat, p_value = scstats.mannwhitneyu(values1, values2, alternative=alternative, **kwargs)
    return float(p_value)


def t_test(values1, values2, alternative='two-sided', equal_var=False, **kwargs):
    """ Calculate the T-test for the means of *two independent* samples of scores.

    This is a test for the null hypothesis that 2 independent samples
    have identical average (expected) values. This test assumes that the
    populations have identical variances by default.

    Computed via ``scipy.stats.ttest_ind``.

    Parameters
    ----------
    values1, values2 : array-like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default), which can be
        specified in ``kwargs``.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.
    equal_var : bool, default=False
        Passed to ``scipy.stats.ttest_ind``; default False corresponds to Welch's t-test.
        If True, performs the standard independent 2 sample test that assumes equal population variances.
    kwargs : `dict`
        Key-word arguments passed to ``scipy.stats.ttest_ind``.

    Returns
    -------
    p_value : `float`
        The p-value.
    """
    t_stat, p_value = scstats.ttest_ind(values1, values2, equal_var=equal_var,
                                        alternative=alternative, **kwargs)
    return float(p_value)


def wilcoxon_signed_rank_test(values1, values2=None, alternative='two-sided', **kwargs):
    """ The Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Computed via ``scipy.stats.wilcoxon``.

    Parameters
    ----------
    values1 : array-like
        Either the first set of measurements (in which case ``y`` is the second
        set of measurements), or the differences between two sets of
        measurements (in which case ``y`` is not to be specified.)  Must be
        one-dimensional.
    values2 : array-like
        Optional. Either the second set of measurements (if ``x`` is the first set of
        measurements), or not specified (if ``x`` is the differences between
        two sets of measurements.)  Must be one-dimensional.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the means of the distributions underlying the samples
          are unequal.
        * 'less': the mean of the distribution underlying the first sample
          is less than the mean of the distribution underlying the second
          sample.
        * 'greater': the mean of the distribution underlying the first
          sample is greater than the mean of the distribution underlying
          the second sample.
    kwargs : `dict`
        Key-word arguments passed to ``scipy.stats.wilcoxon``.

    Returns
    -------
    p_value : `float`
        The p-value.
    """
    w_stat, p_value = scstats.wilcoxon(values1, values2, alternative=alternative, **kwargs)
    return float(p_value)


# Function to choose and perform the statistical test
def perform_stat_test(values1, values2, test_type: str, **kwargs) -> float:
    """
    Choose and perform a statistical test, matching your original dispatch behavior.

    Parameters
    ----------
    values1, values2 : array-like
        Vectors of measurements. For 'wilcoxon', these are paired vectors.
    test_type : {"t-test","MWU","wilcoxon"}
        Which test to apply:
          - "t-test": two-sample independent t-test
          - "MWU": Mann–Whitney U test
          - "wilcoxon": Wilcoxon signed-rank test (paired)
    **kwargs : dict
        Forwarded to the underlying SciPy test wrapper. In particular:
          - You may pass `alternative` in kwargs for all supported tests.

    Returns
    -------
    p_value : float
        P-value from the chosen test.

    Raises
    ------
    ValueError
        If an invalid test_type is provided.
    """
    if test_type == 't-test':
        return t_test(values1, values2, **kwargs)
    elif test_type == 'MWU':
        return mann_whitney_u_test(values1, values2, **kwargs)
    elif test_type == 'wilcoxon':
        return wilcoxon_signed_rank_test(values1, values2, **kwargs)
    else:
        raise ValueError("Invalid test type. Choose 't-test', 'MWU', or 'wilcoxon'.")


# =============================================================================
# Matrix-level p-values for two explicit groups: used by stat_test()
# (X1 and X2 are samples x features arrays)
# =============================================================================

_PAIR_GLOBAL: Dict[str, Any] = {}


def _pair_init_worker(
    X1: np.ndarray,
    X2: np.ndarray,
    *,
    test: TestType,
    alternative: str,
    nan_policy: str,
    test_kwargs: Dict[str, Any],
) -> None:
    """
    Initializer for parallel pairwise feature-wise workers (MWU/Wilcoxon).

    Parameters
    ----------
    X1, X2 : numpy.ndarray
        Matrices shaped (n_samples1, n_features) and (n_samples2, n_features).
    test : {"MWU","wilcoxon"}
        Feature-wise tests supported by this worker.
    alternative : {"two-sided","less","greater"}
        Alternative hypothesis passed to SciPy.
    nan_policy : {"omit","raise"}
        NaN handling.
    test_kwargs : dict
        Extra keyword args forwarded to SciPy test functions.

    Returns
    -------
    None
    """
    global _PAIR_GLOBAL
    _PAIR_GLOBAL = dict(
        X1=X1, X2=X2,
        test=test,
        alternative=alternative,
        nan_policy=nan_policy,
        test_kwargs=test_kwargs,
    )


def _pair_chunk(start_end: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    """
    Compute p-values for a chunk of feature columns for MWU or Wilcoxon.

    Parameters
    ----------
    start_end : tuple(int, int)
        Half-open feature interval (start, end).

    Returns
    -------
    start : int
        Start feature index.
    pvals_chunk : numpy.ndarray
        P-values for features in [start, end).

    Notes
    -----
    This is used only for tests that SciPy does not vectorize across features:
    MWU and Wilcoxon.
    """
    X1 = _PAIR_GLOBAL["X1"]
    X2 = _PAIR_GLOBAL["X2"]
    test = _PAIR_GLOBAL["test"]
    alternative = _PAIR_GLOBAL["alternative"]
    nan_policy = _PAIR_GLOBAL["nan_policy"]
    test_kwargs = _PAIR_GLOBAL["test_kwargs"]

    start, end = start_end
    out = np.empty(end - start, dtype=np.float64)
    isnan = np.isnan

    if test == "MWU":
        fn = scstats.mannwhitneyu
        for j, k in enumerate(range(start, end)):
            a = X1[:, k]
            b = X2[:, k]
            if nan_policy == "omit":
                if isnan(a).any():
                    a = a[~isnan(a)]
                if isnan(b).any():
                    b = b[~isnan(b)]
            if a.size < 1 or b.size < 1:
                out[j] = np.nan
                continue
            res = fn(a, b, alternative=alternative, **test_kwargs)
            out[j] = getattr(res, "pvalue", res[1])
        return start, out

    if test == "wilcoxon":
        fn = scstats.wilcoxon
        for j, k in enumerate(range(start, end)):
            a = X1[:, k]
            b = X2[:, k]
            if nan_policy == "omit":
                ok = ~(isnan(a) | isnan(b))
                a = a[ok]
                b = b[ok]
            if a.size < 1:
                out[j] = np.nan
                continue
            res = fn(a, b, alternative=alternative, **test_kwargs)
            out[j] = getattr(res, "pvalue", res[1])
        return start, out

    raise RuntimeError("Internal error: unsupported test in _pair_chunk.")


def perform_stat_test_matrix(
    X1: Union[pd.DataFrame, np.ndarray],
    X2: Union[pd.DataFrame, np.ndarray],
    *,
    test: TestType = "MWU",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    nan_policy: Literal["omit", "raise"] = "omit",
    equal_var: bool = False,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "processes",
    chunk_size_features: int = 256,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Compute per-feature p-values comparing two groups of samples.

    This function assumes inputs are **samples x features** and returns a vector
    of p-values of length n_features.

    Parameters
    ----------
    X1, X2 : pandas.DataFrame or numpy.ndarray
        Two matrices of shape (n_samples1, n_features) and (n_samples2, n_features).
        If DataFrames are given, they are converted once to NumPy arrays.
    test : {"MWU","t-test","wilcoxon"}, default="MWU"
        Statistical test to perform:

        - "t-test": independent two-sample t-test via ``scipy.stats.ttest_ind`` with ``axis=0``
        - "MWU": Mann–Whitney U via ``scipy.stats.mannwhitneyu`` (feature-wise loop/parallel)
        - "wilcoxon": Wilcoxon signed-rank via ``scipy.stats.wilcoxon`` (paired; requires same n_samples)

    alternative : {"two-sided","less","greater"}, default="two-sided"
        Alternative hypothesis passed to SciPy.
    nan_policy : {"omit","raise"}, default="omit"
        NaN handling:
        - "omit": omit NaNs feature-wise (t-test uses SciPy nan_policy; MWU/Wilcoxon omit manually)
        - "raise": raise if NaNs are present (t-test uses SciPy; MWU/Wilcoxon will yield NaNs or raise upstream)
    equal_var : bool, default=False
        Only used for "t-test". False means Welch’s t-test.
    n_jobs : int, default=1
        Number of workers for MWU/Wilcoxon feature loops. If 1, runs serially.
    parallel_backend : {"processes","threads","none"}, default="processes"
        Backend used for MWU/Wilcoxon parallel execution.
    chunk_size_features : int, default=256
        Number of features per parallel chunk for MWU/Wilcoxon.
    test_kwargs : dict, optional
        Extra keyword args forwarded to the underlying SciPy test.
        - MWU: forwarded to ``mannwhitneyu``
        - Wilcoxon: forwarded to ``wilcoxon``
        - t-test: forwarded to ``ttest_ind`` (in addition to alternative/equal_var/nan_policy)

    Returns
    -------
    pvals : numpy.ndarray, shape (n_features,)
        Raw p-values per feature.

    Raises
    ------
    ValueError
        If feature dimensions mismatch, or if wilcoxon is requested with mismatched sample counts.

    Notes
    -----
    Why SciPy t-test is enough here:
    - ``ttest_ind`` supports vectorization across features with ``axis=0``,
      so there is no Python loop per feature.
    - Using SciPy directly keeps this implementation simple and robust.

    Why MWU/Wilcoxon still loop:
    - SciPy does not vectorize these tests across features, so looping (and optional parallelism)
      is necessary if you want one p-value per feature.
    """
    if test_kwargs is None:
        test_kwargs = {}

    A = X1.to_numpy(copy=False) if isinstance(X1, pd.DataFrame) else np.asarray(X1)
    B = X2.to_numpy(copy=False) if isinstance(X2, pd.DataFrame) else np.asarray(X2)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("X1 and X2 must be 2D arrays (n_samples, n_features).")
    if A.shape[1] != B.shape[1]:
        raise ValueError("X1 and X2 must have the same number of features (columns).")
    if test == "wilcoxon" and A.shape[0] != B.shape[0]:
        raise ValueError("Wilcoxon signed-rank requires X1 and X2 to have the same number of samples (rows).")

    A = np.asarray(A, dtype=np.float64, order="C")
    B = np.asarray(B, dtype=np.float64, order="C")

    if test == "t-test":
        res = scstats.ttest_ind(
            A, B,
            axis=0,
            equal_var=equal_var,
            nan_policy=nan_policy,
            alternative=alternative,
            **test_kwargs,
        )
        return np.asarray(res.pvalue, dtype=np.float64)

    n_features = A.shape[1]
    chunks = _feature_chunks(n_features, chunk_size_features)

    # Serial
    if parallel_backend == "none" or n_jobs <= 1:
        _pair_init_worker(A, B, test=test, alternative=alternative,
                          nan_policy=nan_policy, test_kwargs=dict(test_kwargs))
        p = np.empty(n_features, dtype=np.float64)
        for start, end in chunks:
            s, out = _pair_chunk((start, end))
            p[s:s + out.size] = out
        return p

    # Threads
    if parallel_backend == "threads":
        import concurrent.futures as cf

        _pair_init_worker(A, B, test=test, alternative=alternative, nan_policy=nan_policy,
                          test_kwargs=dict(test_kwargs))
        p = np.empty(n_features, dtype=np.float64)
        with cf.ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            for start, out in ex.map(_pair_chunk, chunks):
                p[start:start + out.size] = out
        return p

    # Processes
    if parallel_backend == "processes":
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork") if "fork" in methods else mp.get_context()

        p = np.empty(n_features, dtype=np.float64)

        init = partial(
            _pair_init_worker,
            test=test,
            alternative=alternative,
            nan_policy=nan_policy,
            test_kwargs=dict(test_kwargs),
        )

        with ctx.Pool(
                processes=int(n_jobs),
                initializer=init,
                initargs=(A, B),
        ) as pool:
            for start, out in pool.imap_unordered(_pair_chunk, chunks, chunksize=1):
                p[start: start + out.size] = out
        return p

    raise ValueError("parallel_backend must be one of {'processes','threads','none'}.")


# =============================================================================
# Mask-based MWU (group vs rest) for sig_feats_by_group
# (MWU is feature-wise; avoid building X_group/X_rest matrices)
# =============================================================================


_MWU_MASK_GLOBAL: Dict[str, Any] = {}


def _mwu_mask_init_worker(
    X: np.ndarray,
    *,
    alternative: str,
    nan_policy: str,
    mwu_kwargs: Dict[str, Any],
) -> None:
    """
    Initializer for MWU group-vs-rest workers.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Full matrix.
    alternative : {"two-sided","less","greater"}
        Alternative hypothesis passed to SciPy.
    nan_policy : {"omit","raise"}
        NaN handling.
    mwu_kwargs : dict
        Extra keyword args forwarded to ``scipy.stats.mannwhitneyu``.

    Returns
    -------
    None
    """
    global _MWU_MASK_GLOBAL
    _MWU_MASK_GLOBAL = dict(X=X, alternative=alternative, nan_policy=nan_policy, mwu_kwargs=mwu_kwargs)


def _mwu_mask_chunk(task: Tuple[int, int, np.ndarray, np.ndarray]) -> Tuple[int, np.ndarray]:
    """
    Compute MWU p-values for a chunk of feature columns given group/rest row indices.

    Parameters
    ----------
    task : tuple
        (start, end, g_idx, r_idx) where:
        - start/end define feature interval [start, end)
        - g_idx are sample indices in group
        - r_idx are sample indices in rest

    Returns
    -------
    start : int
        Start feature index.
    pvals_chunk : numpy.ndarray
        MWU p-values for features in [start, end).

    Notes
    -----
    This avoids materializing (n_group x n_features) and (n_rest x n_features) matrices.
    It only gathers the 1D vectors needed per feature.
    """
    X = _MWU_MASK_GLOBAL["X"]
    alternative = _MWU_MASK_GLOBAL["alternative"]
    nan_policy = _MWU_MASK_GLOBAL["nan_policy"]
    mwu_kwargs = _MWU_MASK_GLOBAL["mwu_kwargs"]

    start, end, g_idx, r_idx = task
    out = np.empty(end - start, dtype=np.float64)
    isnan = np.isnan
    fn = scstats.mannwhitneyu

    for j, k in enumerate(range(start, end)):
        a = X[g_idx, k]
        b = X[r_idx, k]
        if nan_policy == "omit":
            if isnan(a).any():
                a = a[~isnan(a)]
            if isnan(b).any():
                b = b[~isnan(b)]
        if a.size < 1 or b.size < 1:
            out[j] = np.nan
            continue
        res = fn(a, b, alternative=alternative, **mwu_kwargs)
        out[j] = getattr(res, "pvalue", res[1])

    return start, out


def _mwu_pvals_group_vs_rest(
    X: np.ndarray,
    g_mask: np.ndarray,
    *,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    nan_policy: Literal["omit", "raise"] = "omit",
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "processes",
    chunk_size_features: int = 256,
    mwu_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Compute MWU p-values for group vs rest without materializing submatrices.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Full matrix (samples x features).
    g_mask : numpy.ndarray, dtype=bool, shape (n_samples,)
        Boolean mask selecting group samples.
    alternative : {"two-sided","less","greater"}, default="two-sided"
        Alternative hypothesis for MWU.
    nan_policy : {"omit","raise"}, default="omit"
        NaN handling.
    n_jobs : int, default=1
        Parallel workers for MWU feature loop.
    parallel_backend : {"processes","threads","none"}, default="processes"
        Backend for parallel execution.
    chunk_size_features : int, default=256
        Feature chunk size.
    mwu_kwargs : dict, optional
        Extra keyword args forwarded to ``scipy.stats.mannwhitneyu``.

    Returns
    -------
    pvals : numpy.ndarray, shape (n_features,)
        MWU p-values per feature.

    Notes
    -----
    MWU is not vectorized across features in SciPy, so we compute it feature-wise.
    This implementation is memory efficient because it does not allocate X_group/X_rest.
    """
    if mwu_kwargs is None:
        mwu_kwargs = {}

    X = np.asarray(X, dtype=np.float64, order="C")
    g_mask = np.asarray(g_mask, dtype=bool)
    if g_mask.ndim != 1 or g_mask.shape[0] != X.shape[0]:
        raise ValueError("g_mask must be 1D boolean of length n_samples (matching X rows).")

    n_features = X.shape[1]
    chunks = _feature_chunks(n_features, chunk_size_features)

    g_idx = np.flatnonzero(g_mask).astype(np.int64, copy=False)
    r_idx = np.flatnonzero(~g_mask).astype(np.int64, copy=False)

    tasks = [(start, end, g_idx, r_idx) for start, end in chunks]

    if parallel_backend == "none" or n_jobs <= 1:
        _mwu_mask_init_worker(X, alternative=alternative, nan_policy=nan_policy, mwu_kwargs=dict(mwu_kwargs))
        p = np.empty(n_features, dtype=np.float64)
        for t in tasks:
            start, out = _mwu_mask_chunk(t)
            p[start:start + out.size] = out
        return p

    if parallel_backend == "threads":
        import concurrent.futures as cf

        _mwu_mask_init_worker(X, alternative=alternative, nan_policy=nan_policy, mwu_kwargs=dict(mwu_kwargs))
        p = np.empty(n_features, dtype=np.float64)
        with cf.ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            for start, out in ex.map(_mwu_mask_chunk, tasks):
                p[start:start + out.size] = out
        return p

    if parallel_backend == "processes":
        methods = mp.get_all_start_methods()
        ctx = mp.get_context("fork") if "fork" in methods else mp.get_context()

        p = np.empty(n_features, dtype=np.float64)

        init = partial(
            _mwu_mask_init_worker,
            alternative=alternative,
            nan_policy=nan_policy,
            mwu_kwargs=dict(mwu_kwargs),
        )

        with ctx.Pool(
                processes=int(n_jobs),
                initializer=init,
                initargs=(X,),
        ) as pool:
            for start, out in pool.imap_unordered(_mwu_mask_chunk, tasks, chunksize=1):
                p[start: start + out.size] = out
        return p

    raise ValueError("parallel_backend must be one of {'processes','threads','none'}.")


def _coerce_two_dfs_samples_x_features(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    samples_axis: Literal["auto", "index", "columns"] = "auto",
    test: Literal["MWU", "t-test", "wilcoxon"] = "MWU",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coerce and strictly align two DataFrames so both are shaped as (n_samples, n_features).

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        Two matrices to compare. Each may be either:
        - samples x features (samples on index), or
        - features x samples (samples on columns)
    samples_axis : {"auto","index","columns"}, default="auto"
        - "index": enforce samples on index (samples x features)
        - "columns": enforce samples on columns (features x samples), then transpose
        - "auto": infer strictly from label agreement between df1 and df2:
            * if df1.columns == df2.columns and df1.index != df2.index  -> samples_axis="index"
            * if df1.index == df2.index and df1.columns != df2.columns  -> samples_axis="columns"
            * if both index and columns match -> ambiguous; choose by heuristic
              (smaller dimension treated as samples); if tie, raise.
            * if neither matches -> raise.
    test : {"MWU","t-test","wilcoxon"}, default="MWU"
        If "wilcoxon", requires paired samples after coercion (same sample index).

    Returns
    -------
    X1, X2 : pandas.DataFrame
        Coerced matrices as samples x features, strictly aligned on features.
        If test="wilcoxon", also strictly aligned on samples.

    Raises
    ------
    ValueError
        If strict alignment is not possible under the requested/inferred orientation.
    """
    if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
        raise TypeError("df1 and df2 must be pandas DataFrames.")

    if samples_axis == "auto":
        idx_match = df1.index.equals(df2.index)
        col_match = df1.columns.equals(df2.columns)

        if col_match and not idx_match:
            samples_axis_use = "index"     # samples on index, features on columns
        elif idx_match and not col_match:
            samples_axis_use = "columns"   # samples on columns, features on index
        elif idx_match and col_match:
            # Ambiguous: both axes match. Use a deterministic heuristic:
            # treat the smaller axis as samples (common in genomics: n_samples < n_features).
            if df1.shape[0] < df1.shape[1]:
                samples_axis_use = "index"
            elif df1.shape[1] < df1.shape[0]:
                samples_axis_use = "columns"
            else:
                raise ValueError(
                    "samples_axis='auto' is ambiguous because both index and columns match and shapes are square. "
                    "Please specify samples_axis='index' or 'columns'."
                )
        else:
            raise ValueError(
                "samples_axis='auto' could not infer orientation. Expected either:\n"
                "  - df1.columns == df2.columns (same features; samples on index)\n"
                "  - df1.index == df2.index (same features; samples on columns)\n"
                "No safe strict alignment was found."
            )
    else:
        samples_axis_use = samples_axis

    if samples_axis_use == "index":
        # samples x features
        if not df1.columns.equals(df2.columns):
            raise ValueError("samples_axis='index' requires df1.columns == df2.columns (same features).")
        X1, X2 = df1, df2
    elif samples_axis_use == "columns":
        # features x samples -> transpose to samples x features
        if not df1.index.equals(df2.index):
            raise ValueError("samples_axis='columns' requires df1.index == df2.index (same features).")
        X1, X2 = df1.T, df2.T
    else:
        raise ValueError("samples_axis must be one of {'auto','index','columns'}.")

    if test == "wilcoxon" and not X1.index.equals(X2.index):
        raise ValueError("wilcoxon requires paired samples: df1 and df2 must have identical sample IDs/order.")

    return X1, X2


# =============================================================================
# Public API: stat_test and sig_feats_by_group
# =============================================================================

def stat_test(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    test: TestType = "MWU",
    alpha: float = 0.05,
    method: str = "fdr_bh",
    samples_axis: Literal["auto", "index", "columns"] = "auto",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    nan_policy: Literal["omit", "raise"] = "omit",
    equal_var: bool = False,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "processes",
    chunk_size_features: int = 256,
    test_kwargs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """ Perform statistical test between groups/datasets and apply multiple test correction.

    This compares `df1` vs `df2` feature-by-feature, returning raw and corrected p-values.

    The statistical tests are Computed via ``scipy.stats``.

    Parameters
    ----------
    df1, df2 : `pandas.DataFrame`
        The measurements, where rows are features and columns are observations.
        The dataframes  must have the same number of features (rows).
        If ``test='wilcoxon'``, they must also have the same number of
        observationas (columns).

        Note: Can now handle datasets oriented as:

        - samples x features  (samples on index, features on columns)
        - features x samples  (features on index, samples on columns)

        The `samples_axis` argument controls whether orientation is inferred or enforced.

        If `test="wilcoxon"` (paired), the sample dimension must match and be aligned
        after coercion.
    test : {"MWU","t-test","wilcoxon"}
        The statistical test that should be performed. Options are:

        - 'MWU' : Mann Whitney-U Test (default).
        - 't-test' : T-test
        - 'wilcoxon' : Wilcoxon Signed Rank Test
    alpha : `float`
        The family-wise error rate (FWER) passed to statsmodels `multipletests`, should be between 0 and 1.
    method : `str`
        Method for multiple test correction, default='fdr_bh'.

        Options:

        - `bonferroni` : one-step correction
        - `sidak` : one-step correction
        - `holm-sidak` : step down method using Sidak adjustments
        - `holm` : step-down method using Bonferroni adjustments
        - `simes-hochberg` : step-up method  (independent)
        - `hommel` : closed method based on Simes tests (non-negative)
        - `fdr_bh` : Benjamini/Hochberg  (non-negative)
        - `fdr_by` : Benjamini/Yekutieli (negative)
        - `fdr_tsbh` : two stage fdr correction (non-negative)
        - `fdr_tsbky` : two stage fdr correction (non-negative)
    samples_axis : {"auto","index","columns"}, default="auto"
        Orientation control:

        - "index": enforce samples on df.index (df is samples x features)
        - "columns": enforce samples on df.columns (df is features x samples), then transpose
        - "auto": infer orientation strictly from label agreement between df1 and df2
          (see `_coerce_two_dfs_samples_x_features` for details)
    alternative : {"two-sided","less","greater"}, default="two-sided"
        Alternative hypothesis passed to the underlying SciPy test.
    nan_policy : {"omit","raise"}, default="omit"
        NaN handling. For t-test, this is passed to SciPy. For MWU/Wilcoxon, NaNs are
        handled feature-wise by the underlying implementation.
    equal_var : bool, default=False
        Used for t-test only. False means Welch’s t-test.
    n_jobs : int, default=1
        Workers for MWU/Wilcoxon (feature-wise). If 1, runs serially.
    parallel_backend : {"processes","threads","none"}, default="processes"
        Backend for MWU/Wilcoxon.
    chunk_size_features : int, default=256
        Feature chunk size for MWU/Wilcoxon.
    test_kwargs : `dict`
        Key-word arguments passed to ``scipy.stats`` for performing the
        statistical test.

    Returns
    -------
    record : `pandas.DataFrame`
        DataFrame indexed by feature name with columns:
        - "p-value"
        - "corrected p-value"

    DataFrame indexed by feature name with columns:
        - "p-value"
        - "corrected p-value"

    Notes
    -----
    - For unpaired tests ("MWU" and "t-test"), this wrapper routes to the optimized
      group-vs-rest engine `_sig_feats_by_group_core` by constructing a two-group
      grouping vector over the concatenated observations.
    - MWU can be parallelized across feature chunks using additional kwargs:
        * n_jobs (int): number of workers (default 1)
        * parallel_backend ({"processes","threads","none"}): backend (default "processes")
        * chunk_size_features (int): features per chunk (default 256)
    - For "wilcoxon" (paired), a paired signed-rank test is computed between df1 and df2 columns, feature-wise.
    - These keys (if present) are **consumed by the wrapper** and not forwarded to SciPy:
        - alternative : {"two-sided","less","greater"} (default "two-sided")
        - nan_policy : {"omit","raise"} (default "omit")
          * For t-test, this is passed to SciPy.
          * For MWU/wilcoxon, NaNs are omitted feature-wise when nan_policy="omit".
        - equal_var : bool (default False) for t-test (Welch vs pooled)
        - n_jobs : int (default 1) for MWU parallelism
        - parallel_backend : {"processes","threads","none"} (default "processes") for MWU
        - chunk_size_features : int (default 256) for MWU
    """
    assert (df1.index == df2.index).all(), "DataFrames must have the same index."
    if test == 'wilcoxon':
        assert (df1.columns == df2.columns).all(), (
            "DataFrames must have the same observations for performing the Wilcoxon Signed Rank Test."
        )

    # ~~~~ legacy implementation ~~~~~
    # p_values = [perform_stat_test(df1.loc[k].values.astype(float),
    #                               df2.loc[k].values.astype(float), test, **test_kwargs) for k in df1.index]

    # corrected_p_values = multipletests(p_values, alpha=alpha, method=method)[1]

    # record = pd.DataFrame(data=[p_values, corrected_p_values],
    #                       columns=df1.index.copy(), index=['p-value', 'corrected p-value']).T
    # ~~~~~ end legacy implementation ~~~~~~

    if test_kwargs is None:
        test_kwargs = {}

    X1_df, X2_df = _coerce_two_dfs_samples_x_features(
        df1,
        df2,
        samples_axis=samples_axis,
        test=test,
    )

    feat_names = X1_df.columns
    X1 = X1_df.to_numpy(copy=False)
    X2 = X2_df.to_numpy(copy=False)

    pvals = perform_stat_test_matrix(
        X1,
        X2,
        test=test,
        alternative=alternative,
        nan_policy=nan_policy,
        equal_var=equal_var,
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        chunk_size_features=chunk_size_features,
        test_kwargs=dict(test_kwargs),
    )

    qvals = _apply_multipletests_safe(pvals, alpha=alpha, method=method)

    record = pd.DataFrame(
        {"p-value": np.asarray(pvals, dtype=np.float64), "corrected p-value": np.asarray(qvals, dtype=np.float64)},
        index=pd.Index(feat_names, name=getattr(feat_names, "name", None)),
    )
    return record


def sig_feats_by_group(
    groups: pd.Series,
    feats: pd.DataFrame,
    *,
    test: TestType = "MWU",
    alpha: float = 0.05,
    method: str = "fdr_bh",
    min_group_size: int = 10,
    samples_axis: Literal["auto", "index", "columns"] = "auto",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    nan_policy: Literal["omit", "raise"] = "omit",
    equal_var: bool = False,
    n_jobs: int = 1,
    parallel_backend: ParallelBackend = "processes",
    chunk_size_features: int = 256,
    test_kwargs: Optional[Dict[str, Any]] = None,
    top_n: Optional[int] = None,
    return_type: ReturnType = "wide",
    add_effect_sizes: bool = True,
    log2fc_pseudocount: float = 1e-9,
) -> Union[Dict[Any, pd.DataFrame], pd.DataFrame]:
    """
    Compute per-group feature significance vs the rest of the cohort.

    For each group label `g`, compare samples in `g` vs all other samples,
    producing per-feature p-values and multiple-test-corrected p-values.

    Parameters
    ----------
    groups : pandas.Series
        Group labels indexed by sample ID.
    feats : pandas.DataFrame
        Feature matrix in either orientation:
        - samples x features (samples on index) or
        - features x samples (samples on columns)

        `samples_axis` controls inference/forcing of orientation.
    test : {"MWU","t-test","wilcoxon"}, default="MWU"
        Statistical test to perform.
        - "MWU": Mann–Whitney U (unpaired, rank-based)
        - "t-test": independent two-sample t-test (Welch by default)
        - "wilcoxon": Wilcoxon signed-rank (paired)

        IMPORTANT: "wilcoxon" is not a generic group-vs-rest test (paired design).
        This function will raise if test="wilcoxon".
    alpha : float, default=0.05
        Error rate for multiple testing correction.
    method : str, default="fdr_bh"
        Multiple test correction method.

        Options
        -------
        - ``bonferroni`` : one-step correction
        - ``sidak`` : one-step correction
        - ``holm-sidak`` : step down method using Sidak adjustments
        - ``holm`` : step-down method using Bonferroni adjustments
        - ``simes-hochberg`` : step-up method  (independent)
        - ``hommel`` : closed method based on Simes tests (non-negative)
        - ``fdr_bh`` : Benjamini/Hochberg  (non-negative)
        - ``fdr_by`` : Benjamini/Yekutieli (negative)
        - ``fdr_tsbh`` : two stage fdr correction (non-negative)
        - ``fdr_tsbky`` : two stage fdr correction (non-negative)

    min_group_size : int, default=10
        Minimum number of samples required for a group to be tested.
        Must be >= 3.
    samples_axis : {"auto","index","columns"}, default="auto"
        Orientation control passed to `_coerce_samples_x_features`.
    alternative : {"two-sided","less","greater"}, default="two-sided"
        Alternative hypothesis.
    nan_policy : {"omit","raise"}, default="omit"
        NaN handling.
    equal_var : bool, default=False
        Used for t-test only. False means Welch’s t-test.
    n_jobs : int, default=1
        Workers for MWU feature-wise computation. If 1, runs serially.
    parallel_backend : {"processes","threads","none"}, default="processes"
        Backend for MWU parallelism.
    chunk_size_features : int, default=256
        Feature chunk size for MWU parallelism.
    test_kwargs : dict, optional
        Extra kwargs forwarded to SciPy test functions.
    top_n : int, optional
        If provided, keep only the top_n features per group after sorting
        by corrected p-value then raw p-value.
    return_type : {"wide", "dict"}, default=False
        If "wide" (default), return a wide DataFrame with MultiIndex columns (group, metric).
        If "dict", return a dict mapping group -> per-group record DataFrame.
    add_effect_sizes : bool, default=True
        If `True`, include effect size summaries per feature:

        - n_in: number of samples within group (constant across features)
        - n_out: number of samples outside group (constant across features)
        - mean_in: mean feature value within group
        - mean_out: mean feature value outside group
        - mean_diff: mean_in - mean_out
        - log2fc: log2((mean_in + pseudocount) / (mean_out + pseudocount))
    log2fc_pseudocount : float, default=1e-9
        Pseudocount added to means for log2 fold-change to avoid division by zero
        and log(0). Only used when `add_effect_sizes=True`.

    Returns
    -------
    records_full : dict or pandas.DataFrame
        If return_type="wide":
            DataFrame with index=features and columns MultiIndex (group, metric),
            where metric in {"p-value","corrected p-value"}.
        If return_type="dict":
            dict[group_label -> DataFrame(index=features, columns=["p-value","corrected p-value"])]
            Each per-group DataFrame is sorted by corrected p-value then p-value.

    Raises
    ------
    ValueError
        If min_group_size < 3, method invalid, alignment fails, or test="wilcoxon".

    Notes
    -----
    - t-test uses SciPy's vectorized implementation across features via axis=0.
    - MWU is computed feature-wise (SciPy is not vectorized). This implementation
      avoids large submatrix allocations by gathering per-feature vectors via indices.
    """
    if test == "wilcoxon":
        raise ValueError(
            "wilcoxon signed-rank is a paired test and is not valid for generic group-vs-rest splits. "
            "Use test='MWU' for an unpaired rank-based alternative, or use stat_test(..., test='wilcoxon') "
            "for explicitly paired matrices."
        )

    if method not in _ALLOWED_MULTITEST:
        raise ValueError(
            f"Unknown multiple-testing method '{method}'. Must be one of: {sorted(_ALLOWED_MULTITEST)}."
        )

    if test_kwargs is None:
        test_kwargs = {}

    if not isinstance(min_group_size, int) or min_group_size < 3:
        raise ValueError("min_group_size must be an int >= 3.")

    if add_effect_sizes:
        if not np.isfinite(log2fc_pseudocount) or log2fc_pseudocount <= 0:
            raise ValueError("log2fc_pseudocount must be a finite positive float.")

    # groups = groups.astype("category")
    groups_aligned, X_df = _coerce_samples_x_features(groups, feats, samples_axis=samples_axis)

    groups_cat = groups_aligned.astype("category")
    codes = groups_cat.cat.codes.to_numpy(np.int32)
    categories = list(groups_cat.cat.categories)

    # X is samples x features
    X = np.asarray(X_df.to_numpy(copy=False), dtype=np.float64, order="C")
    feat_names = X_df.columns.to_numpy()


    n_samples, n_features = X.shape
    if n_samples != len(groups_cat):
        raise RuntimeError("Internal alignment error: feats and groups mismatch after coercion.")

    records: Dict[Any, pd.DataFrame] = {}

    # If there are exactly two groups and doing a two-sided test,
    # running both directions is redundant (same p/q values). Run only once.
    n_cats = len(categories)
    counts = np.bincount(codes[codes >= 0], minlength=n_cats).astype(int)
    # Eligible groups are those large enough to test; "out" can include tiny groups
    eligible_gis = [gi for gi in range(n_cats) if counts[gi] >= min_group_size and counts[gi] < len(codes)]

    # Special case: exactly two categories total AND two-sided test.
    # If both categories are eligible, computing both directions is redundant.
    if n_cats == 2 and alternative == "two-sided":
        if len(eligible_gis) == 2:
            # Choose a deterministic "in" group to compute once.
            # I recommend the smaller group; tie -> first.
            gi_keep = eligible_gis[int(np.argmin([counts[gi] for gi in eligible_gis]))]
            eligible_gis = [gi_keep]

    # for gi, gname in enumerate(categories):
    for gi in eligible_gis:
        gname = categories[gi]
        g_mask = (codes == gi)
        n_in = int(g_mask.sum())
        n_out = int((~g_mask).sum())
        if n_in < min_group_size or n_out < min_group_size:
            continue

        if test == "t-test":
            # SciPy vectorized across features (axis=0 because features are columns)
            res = scstats.ttest_ind(
                X[g_mask, :],
                X[~g_mask, :],
                axis=0,
                equal_var=equal_var,
                nan_policy=nan_policy,
                alternative=alternative,
                **test_kwargs,
            )
            pvals = np.asarray(res.pvalue, dtype=np.float64)

        elif test == "MWU":
            pvals = _mwu_pvals_group_vs_rest(
                X,
                g_mask,
                alternative=alternative,
                nan_policy=nan_policy,
                n_jobs=n_jobs,
                parallel_backend=parallel_backend,
                chunk_size_features=chunk_size_features,
                mwu_kwargs=dict(test_kwargs),
            )
        else:
            raise ValueError("test must be one of {'MWU','t-test','wilcoxon'}.")

        qvals = _apply_multipletests_safe(pvals, alpha=alpha, method=method)

        cols: Dict[str, Any] = {
            "pval": pvals,
            "qval": qvals,
        }

        if add_effect_sizes:
            cols["n_in"] = int(n_in) # np.full(X.shape[1], n_in, dtype=np.int64), # np.full(n_features, n_in, dtype=np.int32),
            cols["n_out"] = int(n_out) # np.full(X.shape[1], n_out, dtype=np.int64),  # np.full(n_features, n_out, dtype=np.int32),

            mean_in = np.nanmean(X[g_mask, :], axis=0) if nan_policy == "omit" else X[g_mask, :].mean(axis=0)
            mean_out = np.nanmean(X[~g_mask, :], axis=0) if nan_policy == "omit" else X[~g_mask, :].mean(axis=0)
            mean_diff = mean_in - mean_out
            log2fc = np.log2((mean_in + log2fc_pseudocount) / (mean_out + log2fc_pseudocount))
            cols["mean_in"] = np.asarray(mean_in, dtype=np.float64)
            cols["mean_out"] = np.asarray(mean_out, dtype=np.float64)
            cols["mean_diff"] = np.asarray(mean_diff, dtype=np.float64)
            cols["log2fc"] = np.asarray(log2fc, dtype=np.float64)

        rec = pd.DataFrame(cols, index=pd.Index(feat_names, name=getattr(feat_names, "name", None)))
        # rec.insert(0, "group", gname)

        rec = rec.sort_values(by=["qval", "pval"], ascending=[True, True])

        if top_n is not None:
            rec = rec.head(int(top_n))

        records[gname] = rec

    if return_type == "dict":
        records_full = {k: v.copy() for k, v in records.items()}
    elif return_type == "wide":
        H = ["pval", "qval"]
        if add_effect_sizes:
            H += ["n_in", "n_out", "mean_in", "mean_out", "mean_diff", "log2fc"]

        # cols = pd.MultiIndex.from_product([list(records.keys()), ["p-value", "corrected p-value"]])
        # cols = pd.MultiIndex.from_product([list(records.keys()), ["pval", "qval"]])
        cols = pd.MultiIndex.from_product([list(records.keys()), H])
        wide = pd.DataFrame(index=feat_names.copy(), columns=cols, dtype=float)
        for gname, rec in records.items():
            aligned = rec.reindex(feat_names)
            for m in H:
                wide[(gname, m)] = aligned[m].to_numpy()
            # wide[(gname, "p-value")] = aligned["p-value"].to_numpy()
            # wide[(gname, "corrected p-value")] = aligned["corrected p-value"].to_numpy()
            # wide[(gname, "pval")] = aligned["pval"].to_numpy()
            # wide[(gname, "qval")] = aligned["qval"].to_numpy()
        wide.attrs["records"] = records

        records_full = wide
    else:
        raise ValueError("Unexpected return_type, must be one of {'wide', 'dict'}.")

    return records_full




def compute_spearman(row_x, row_y):
    """Compute Spearman correlation with each row in Y for a given row in X.
    
    Parameters
    ----------
    row_x, row_y : array_like
        1-D arrays representing multiple observations of a single variable. 
        The correlation is computed between ``row_x`` and ``row_y``.
        
    Returns
    -------
    correlation : `float`
        The correlation.
    p_value : `float`
        The p-value.
    """    
    correlation, pvalue = scstats.spearmanr(row_x, row_y)
    return correlation, pvalue


def compute_spearman_parallel(X, Y, num_processors=None, chunksize=None):
    """Compute Spearman correlation between each row of X and all rows of Y in parallel.
    
    Parameters
    ----------
    X, Y : `pandas.DataFrame` 
        Dataframes containing multiple variables and observations. 
        Each row represents a variable and each column is an observation of each variable.
        `X` and `Y` must have the same number of columns (i.e., the same observations) 
        but they need not have the same number of variables. 
    num_processors : `int`
        Number of processors to use. Defaults to None (uses all available).
    chunksize

    Returns
    -------
    correlations : `dict`
        The resulting correlations in the form ``{index_row_X: {index_row_Y: corr}}``
    p_values : `dict`
        The resulting p_values in the form ``{index_row_X: {index_row_Y: p_value}}``
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimension miss-match, expected X and Y to have the same number of columns")
    
    # ensure the columns of X and Y are ordered the same
    Y = Y[X.columns]
    
    if num_processors is None:
        num_processors = mp.cpu_count()  # Use all available processors

    with mp.Pool(processes=num_processors) as pool:
        result = pool.starmap(compute_spearman, itertools.product(X.values, Y.values), chunksize=chunksize)

    # correlations = {k: v[0] for k, v in zip(itertools.product(X.index, Y.index), result)}
    # p_values = {k: v[1] for k, v in zip(itertools.product(X.index, Y.index), result)}

    correlations = {k: {} for k in X.index}
    p_values = {k: {} for k in X.index}
    for (x_idx, y_idx), (rho, p_val) in zip(itertools.product(X.index, Y.index), result):
        correlations[x_idx][y_idx] = rho
        p_values[x_idx][y_idx] = p_val

    return correlations, p_values


def compute_pearson(row_x, row_y):
    """Compute Pearson correlation with each row in Y for a given row in X.
    
    Parameters
    ----------
    row_x, row_y : array_like
        1-D arrays representing multiple observations of a single variable. 
        The correlation is computed between ``row_x`` and ``row_y``.
        
    Returns
    -------
    correlation : `float`
        The correlation.
    p_value : `float`
        The p-value.
    """    
    correlation, pvalue = scstats.pearsonr(row_x, row_y)
    return correlation, pvalue


def compute_pearson_parallel(X, Y, num_processors=None, chunksize=None):
    """Compute Pearson correlation between each row of X and all rows of Y in parallel.
    
    Parameters
    ----------
    X, Y : `pandas.DataFrame` 
        Dataframes containing multiple variables and observations. 
        Each row represents a variable and each column is an observation of each variable.
        `X` and `Y` must have the same number of columns (i.e., the same observations) 
        but they need not have the same number of variables. 
    num_processors : `int`
        Number of processors to use. Defaults to None (uses all available).
    chunksize

    Returns
    -------
    correlations : `dict`
        The resulting correlations in the form ``{index_row_X: {index_row_Y: corr}}``
    p_values : `dict`
        The resulting p_values in the form ``{index_row_X: {index_row_Y: p_value}}``
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimension miss-match, expected X and Y to have the same number of columns")
    
    # ensure the columns of X and Y are ordered the same
    Y = Y[X.columns]
    
    if num_processors is None:
        num_processors = mp.cpu_count()  # Use all available processors
    
    with mp.Pool(processes=num_processors) as pool:
        result = pool.starmap(compute_pearson, itertools.product(X.values, Y.values), chunksize=chunksize)

    # correlations = {k: v[0] for k, v in zip(itertools.product(X.index, Y.index), result)}
    # p_values = {k: v[1] for k, v in zip(itertools.product(X.index, Y.index), result)}

    correlations = {k: {} for k in X.index}
    p_values = {k: {} for k in X.index}
    for (x_idx, y_idx), (rho, p_val) in zip(itertools.product(X.index, Y.index), result):
        correlations[x_idx][y_idx] = rho
        p_values[x_idx][y_idx] = p_val

    return correlations, p_values
