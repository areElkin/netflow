"""
This script includes methods for describing resulting branches / clusters / co-localization on the POSE.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Union
import numpy as np
import pandas as pd


def unique_significant_features_by_group(
    records_full: Union[Dict[Any, pd.DataFrame], pd.DataFrame],
    *,
    metric: Literal["pval", "qval"] = "qval",
    threshold: float = 0.05,
    dropna: bool = True,
    sort_within_group: bool = True,
) -> Dict[Any, List[str]]:
    """
    Identify features that are significant in exactly one group and not significant in any other group.

    This function consumes the output of `netflow.methods.stats.sig_feats_by_group()`
    in either of its supported formats:

    1) Dictionary format:
       ``records_full[group] -> DataFrame(index=features, columns include metric)``
       where `metric` is either "pval" (p-value) or "qval" (corrected p-value).

    2) Wide DataFrame format:
       A DataFrame indexed by features with MultiIndex columns of the form
       ``(group, metric_name)``, where `metric_name` includes "pval" and/or
       "qval".

    The output is a dictionary mapping each group to a list of features that satisfy:

    - feature is significant for that group according to the chosen `metric` and `threshold`, and
    - feature is NOT significant for every other group in `records_full`.

    Parameters
    ----------
    records_full : dict or pandas.DataFrame
        Results returned by `sig_feats_by_group()`.

        - If dict: keys are group labels; values are per-group DataFrames.
        - If DataFrame: must have MultiIndex columns with level 0 = group and level 1 = metric.

    metric : {"pval", "qval"}, default="qval"
        Which significance column to use when deciding whether a feature is significant.

        - "pval": raw p-values
        - "qval": multiple-testing corrected p-values (q-values)

    threshold : float, default=0.05
        Significance threshold applied to `metric`. A feature is considered significant for a
        given group if `metric_value <= threshold`.

    dropna : bool, default=True
        If True, treat NaN metric values as "not significant".
        If False, NaNs propagate only through comparisons (NaN <= threshold is False anyway),
        so behavior is effectively the same for the significance mask; this mainly affects
        internal bookkeeping.

    sort_within_group : bool, default=True
        If True, features returned per group are sorted by the chosen `metric` ascending
        (most significant first). If False, features follow the underlying feature index order.

    Returns
    -------
    unique_feats : dict
        Dictionary keyed by group label, where values are lists of feature names that are
        significant only for that group and for no other group.

    Raises
    ------
    TypeError
        If `records_full` is neither a dict nor a DataFrame.
    ValueError
        If `metric` is not present in the provided results, or if a wide DataFrame does not
        have the expected MultiIndex column structure.

    Notes
    -----
    - If you ran `netflow.methods.stats.sig_feats_by_group(..., top_n=...)`, then `records_full` may
      only include a subset of features per group. In that case, "unique" is evaluated relative to what
      is present in `records_full` (missing features are treated as not significant for that group).
      For strict uniqueness across all tested features, generate results without `top_n`.
      It is therefore recommended and expected that results were generated without `top_n`.
    - If `netflow.methods.stats.sig_feats_by_group()` used the two-category optimization
      (compute only one direction), then `records_full` may not contain both groups; uniqueness can
      only be assessed across the groups that are present in `records_full`.
    """
    if not (isinstance(threshold, (int, float)) and np.isfinite(threshold)):
        raise ValueError("threshold must be a finite numeric value.")

    # ---- Normalize input into a (features x groups) metric table ----
    if isinstance(records_full, dict):
        groups = list(records_full.keys())
        if len(groups) == 0:
            unique_feats = {}
            return unique_feats

        # Union of all features across groups
        all_features = None
        for g in groups:
            df = records_full[g]
            if metric not in df.columns:
                raise ValueError(f"metric='{metric}' not found in records_full[{g!r}] columns.")
            all_features = df.index if all_features is None else all_features.union(df.index)

        metric_table = pd.DataFrame(index=all_features, columns=groups, dtype=float)
        for g, df in records_full.items():
            metric_table[g] = df[metric].reindex(all_features).astype(float)

    elif isinstance(records_full, pd.DataFrame):
        wide = records_full
        if not isinstance(wide.columns, pd.MultiIndex) or wide.columns.nlevels < 2:
            raise ValueError(
                "Wide records_full must have MultiIndex columns with (group, metric). "
                "Got non-MultiIndex or insufficient levels."
            )

        groups = list(pd.Index(wide.columns.get_level_values(0)).unique())
        metrics_available = set(wide.columns.get_level_values(1).tolist())
        if metric not in metrics_available:
            raise ValueError(
                f"metric='{metric}' not found in wide DataFrame columns level 1: {sorted(metrics_available)}")

        metric_table = pd.DataFrame(index=wide.index)
        for g in groups:
            if (g, metric) in wide.columns:
                metric_table[g] = wide[(g, metric)].astype(float)
            else:
                # group exists but metric missing -> treat as all-NaN (not significant)
                metric_table[g] = np.nan
    else:
        raise TypeError("records_full must be either a dict[group -> DataFrame] or a wide pandas DataFrame.")

    # ---- Build significance mask and compute uniqueness ----
    # NaN comparisons yield False, so NaNs are naturally "not significant".
    sig = metric_table.le(threshold)
    if dropna:
        sig = sig & metric_table.notna()

    # A feature is "unique" for a group if it's significant for exactly one group
    sig_counts = sig.sum(axis=1)
    unique_mask = (sig_counts == 1)

    unique_feats: Dict[Any, List[str]] = {}
    for g in groups:
        g_mask = unique_mask & sig[g]
        feats = metric_table.index[g_mask]

        if sort_within_group and len(feats) > 0:
            vals = metric_table.loc[feats, g].to_numpy()
            order = np.argsort(vals, kind="mergesort")
            feats = feats[order]

        unique_feats[g] = feats.tolist()

    return unique_feats



