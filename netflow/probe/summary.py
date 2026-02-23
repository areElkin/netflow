"""
This script includes methods for describing resulting branches / clusters / co-localization on the POSE.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Union
import gseapy as gp
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


def extract_features(df: pd.DataFrame, metric: str, threshold: Union[str, float],
                     apply_lt: bool = True) -> List[str]:
    """
    Extract subset of samples (i.e., rows) based on threshold.

    Parameters
    ----------
    df : `pd.DataFrame`
        The data, expected to have rows as samples with columns as features on which to threshold.
    metric : `str`
        The column header to use as metric for determining samples to keep.
    threshold : {`float`, `str`}
        Threshold applied to `metric`, depends on type:

        - `float` : If ``apply_lt = True``, a sample is kept if `metric_value <= threshold`.
                    If ``apply_lt = False``, a sample is kept if `metric_value >= threshold`.
        - `str` : A sample is kept if `metric_value = threshold`.
    apply_lt : `bool`, default = True
        Indicate how to compare numeric metric values to the threshold. If `True`, retain metric values
        that are less than (lt) ``threshold``. Otherwise, retain values greater than ``threshold``.
        Ignored if ``threshold`` is a `str`.

    Returns
    -------
    selected : List
        The list of sample index labels that match the threshold
    """
    if isinstance(threshold, str):
        selected = df.index[df[metric] == threshold].tolist()
    else:  # numeric
        if apply_lt:
            selected = df.index[df[metric] <= threshold].tolist()
        else:
            selected = df.index[df[metric] >= threshold].tolist()

    return selected


def get_gsea_library_names(organism='Human'):
    """ Get list of gene libraries for GSEA analysis using gseapy.

    This returns active enrichr library names that can be found at: https://maayanlab.cloud/modEnrichr/.

    Parameters
    ----------
    organism : {'Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm'}, default='Human'
        The database to pull libraries from.

    Returns
    -------
    libraries : `list`
        The list of library names.
    """
    libraries = gp.get_library_name(organism=organism)
    return libraries


def enrichr(**kwargs):
    """ Performs gene set enrichment via gseapy.enrichr with detailed feature sizes.

    Parameters
    ----------
    kwargs : `dict`
        Keyword arguments passed to `gseapy.enrichr.

    Returns
    -------
    results : `pd.DataFrame`
        Dataframe with results returned from gseapy.enrichr analysis, with
        additional "Overlap" information columns:

        - "n" : The number of genes in the provided gene list that overlap with the library.
        - "N" : The library size (number of genes in the library).
        - "n/N" : The ratio of overlap of provided genes that are in the library to the library size.

        The following columns returned by gseapy.enrichr are dropped:

        - "Old P-value"
        - "Old Adjusted P-value"

    Examples
    --------
    >>> libraries = ['GO_Biological_Process_2025', 'GO_Cellular_Component_2025', 'GO_Molecular_Function_2025',
                     'Human_Phenotype_Ontology', 'MSigDB_Hallmark_2020', 'MSigDB_Oncogenic_Signatures',
                     'Reactome_Pathways_2024',
                    ]
    >>> G = networkx.Graph(HPRD)  # A gene-gene interaction graph based on HPRD
    >>> gl = ['TOP2A', 'FEN1', 'BLM', 'MCM7', 'MCM8', 'XPC', 'MCM10', 'BRCA1', 'TTF2', ...]
    >>> enrichr(gene_list=gl, # gene_list, # or "./tests/data/gene_list.txt",
                gene_sets=libraries, # ['MSigDB_Hallmark_2020','KEGG_2021_Human'],
                organism='human',
                background=list(G),  # or "hsapiens_gene_ensembl", or int, or text file, or a list of genes
                outdir=None, # if None, don't write to disk
               )
    """
    res = gp.enrichr(**kwargs)
    results = res.results.drop(columns={'Old P-value', 'Old Adjusted P-value'})
    results[['n', 'N']] = results['Overlap'].str.split('/', expand=True).astype(int)
    results['n/N'] = results[['n', 'N']].apply(lambda x: float(x['n']) / float(x['N']), axis=1)
    return results


def gsea_group_summary(records_full: Union[Dict[Any, pd.DataFrame], pd.DataFrame],
                       metric: Literal["pval", "qval"] = "qval",
                       threshold: float = 0.05,
                       dropna: bool = True,
                       **kwargs,
                       ) -> Dict[Any, pd.DataFrame]:
    """
    Perform GSEA summary of significant genes for each group

    This function consumes the output of `netflow.methods.stats.sig_feats_by_group()`
    in either of its supported formats:

    1) Dictionary format:
       ``records_full[group] -> DataFrame(index=features, columns include metric)``
       where `metric` is either "pval" (p-value) or "qval" (corrected p-value).

    2) Wide DataFrame format:
       A DataFrame indexed by features with MultiIndex columns of the form
       ``(group, metric_name)``, where `metric_name` includes "pval" and/or
       "qval".

    The output is a dictionary mapping each group to a pd.DataFrame with the gsea output
    of its significant genes provided by enrichr()

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
    kwargs : `dict`
        Optional keyword arguments passed to enrichr().

    Returns
    -------
    summary : dict
        Dictionary keyed by group label, where values are dataframes of gsea results.

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
    elif isinstance(records_full, pd.DataFrame):
        if not isinstance(records_full.columns, pd.MultiIndex) or records_full.columns.nlevels < 2:
            raise ValueError(
                "Wide records_full must have MultiIndex columns with (group, metric). "
                "Got non-MultiIndex or insufficient levels."
            )

        groups = list(pd.Index(records_full.columns.get_level_values(0)).unique())
    else:
        raise TypeError("records_full must be either a dict[group -> DataFrame] or a pandas DataFrame.")

    if kwargs is None:
        kwargs = {}
    gene_sets = kwargs.get('gene_sets', ['GO_Biological_Process_2025',
                                         'GO_Cellular_Component_2025',
                                         'GO_Molecular_Function_2025',
                                         # 'Human_Phenotype_Ontology',
                                         'MSigDB_Hallmark_2020',
                                         'MSigDB_Oncogenic_Signatures',
                                         'Reactome_Pathways_2024',
                                         ])
    organism = kwargs.get('organism', 'human')
    outdir = kwargs.get('outdir', None)

    summary = {}
    for group in groups:
        df = records_full[group]
        if metric not in df.columns:
            raise ValueError(f"metric='{metric}' not found in records_full[{g!r}] columns.")

        gl = df.index[df[metric] <= threshold].tolist()
        if len(gl) <= 1:
            continue
        enr = enrichr(gene_list=gl, gene_sets=gene_sets, organism=organism, outdir=outdir, **kwargs)
        summary[group] = enr

        # Don't plot if no significant gene sets:
        if enr['Adjusted P-value'].min() > 0.05:
            continue
        # Show top 5 terms of each gene_set ranked by “Adjusted P-value”
        # categorical scatterplot
        ax = gp.dotplot(enr,
                        column="Adjusted P-value",
                        x='Gene_set',  # set x axis, so you could do a multi-sample/library comparsion
                        size=10,
                        top_term=5,
                        cutoff=0.05,
                        figsize=(6, 10),
                        title=f"Group {group}",
                        xticklabels_rot=45,  # rotate xtick labels
                        show_ring=True,  # set to False to revmove outer ring
                        marker='o',
                        fontsize='small',
                        )

        # categorical scatterplot
        ax = gp.barplot(enr,
                        column="Adjusted P-value",
                        group='Gene_set',  # set group, so you could do a multi-sample/library comparsion
                        size=10,
                        cutoff=0.05,
                        top_term=5,
                        figsize=(5, 9),
                        title=f"Group {group}",
                        color=['darkred', 'yellow', 'orange', 'darkgreen',
                               'darkblue', 'purple', 'brown', 'gray'],  # set colors for group
                        fontsize='small',
                        )

    return summary



