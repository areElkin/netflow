"""
This script includes methods for describing resulting branches / clusters / co-localization on the POSE.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Union
import gseapy as gp
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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


def get_graph_distances(graph_nw, weights=None):
    """ Compute distances between all node pairs on a graph.

    Parameters:
    ----------
        graph_nw: `networkx.Graph`
            Input graph.
        weights: {`None`, `pandas.DataFrame`, (n, n)}
            Dataframe of edge weights between nodes (observations). If `None` unweighted hop count is used.

    Returns:
    ----------
        graph_dist: `numpy.ndarray`, (n, n)
        Matrix of pairwise graph distances between nodes ordered 0,1,...,n-1.
        Returns weighted distance if provided. Otherwise, returns hop distance if `weights` is `None`
    """

    graph = graph_nw.copy()

    if weights is None:
        dist_dict = dict(nx.all_pairs_shortest_path_length(graph))
    else:
        observation_labels = weights.columns.to_list()
        for u, v, d in graph.edges(data=True):
            d['distance'] = weights.at[observation_labels[u], observation_labels[v]]
        dist_dict = dict(nx.all_pairs_dijkstra_path_length(graph, weight='distance'))

    node_list = sorted(graph.nodes())
    num_nodes = len(node_list)
    graph_dist = np.zeros((num_nodes, num_nodes))

    for i in node_list:
        for j in node_list:
            graph_dist[i, j] = dist_dict[i].get(j, np.inf)
    return graph_dist


def get_global_node_order(poser, graph_nw, weights=None):
    """ Order nodes by (weighted) graph distance from the root.

    Parameters:
    ----------
        poser: `netflow.pose.POSER`
            The object used to construct the POSE, containing the root node ID.
        graph_nw: `networkx.Graph`
            Input POSE topology.
        weights: {`None`, `pandas.DataFrame`, (n, n)}
            Dataframe of edge weights between nodes (observations). If `None` unweighted hop count is used.

    Returns:
    ----------
        node_ord_dict: `dict`
        Dictionary mapping node indices to ordering in terms of graph distance from the root.
        Ordering is based on weighted distance if provided. Otherwise, based on hop distance if `weights` is `None`
    """

    src_node = poser.root
    node_ids = list(graph_nw.nodes)
    node_dists = get_graph_distances(graph_nw, weights)
    node_ord = np.argsort(node_dists[src_node, node_ids])
    node_ord_dict = dict(zip(node_ids, node_ord))

    return node_ord_dict


def get_branch_node_order(poser, graph_nw, weights=None, min_branch_size=3):
    """  Order observations on each branch by distance from the branch tip nearest to the root.

    Parameters:
    ----------

        poser: `netflow.pose.POSER`
            The object used to construct the POSE.
        graph_nw : `networkx.Graph`
            The POSE graph.
        weights: {`None`, `pandas.DataFrame`, (n, n)}
            Dataframe of edge weights between nodes (observations). If `None` unweighted hop count is used.
        min_branch_size: {`None`, `int`}
            Skip branches with <= ``min_branch_size`` observations.

    Returns:
    ----------
        branch_ord_dict: `dict`
        Dictionary mapping branch node indices to ordering in terms of graph distance from branch tip nearest root.
        Uses weighted distance if provided. Otherwise, uses hop distance if `weights` is `None`
    """

    root_node = poser.root
    branches = poser.tree.get_leaves()

    node_dists = get_graph_distances(graph_nw, weights)
    branch_ord_dict = {}
    for branch in branches:

        if len(branch.data) < min_branch_size:
            continue

        branch_id = branch.name
        branch_tips = list(branch.tips)
        branch_node_ids = list(branch.data)

        tip_ord = np.argsort(node_dists[root_node, branch_tips])
        src_node = branch_tips[tip_ord[0]]
        branch_node_dists = node_dists[src_node, branch_node_ids]
        branch_node_ord = np.argsort(branch_node_dists)
        branch_ord_dict[branch_id] = dict(zip(branch_node_ids, branch_node_ord))

    return branch_ord_dict


def feature_graph_order_correlation_global(poser, graph_nw, data_df, obs_labels, weights=None):
    """ Compute correlation between features and global node ordering.

    Parameters:
    ----------
        poser: `netflow.pose.POSER`
            The object used to construct the POSE.
        graph_nw: `networkx.Graph`
            The POSE graph.
        data_df: `pandas.DataFrame` (n_features, n_observations)
            Feature matrix.
        obs_labels: `list` of `str`
            List of observation labels corresponding to node IDs.
        weights: {`None`, `pandas.DataFrame`}
            Dataframe of edge weights between nodes (observations) used to compute weighted hop distance if provided.
            Unweighted hop count is used if `None`.

    Returns:
    ----------
        corr_df: `pandas.DataFrame`
            Dataframe containing spearman correlation between features and global ordering of nodes, and associated p-values.
    """

    node_ord_dict = get_global_node_order(poser, graph_nw, weights)
    node_ids = list(node_ord_dict.keys())
    obs_subset = [obs_labels[node_id] for node_id in node_ids]

    feat_arr = np.array(data_df.loc[:, obs_subset])
    feat_labels = data_df.index
    node_ord_arr = np.array(list(node_ord_dict.values()))
    result = spearmanr(feat_arr, node_ord_arr, axis=1)
    corr_df = pd.DataFrame({'corr':result.correlation[:-1, -1], 'p-val':result.pvalue[:-1,-1]},
                           index=feat_labels)

    return corr_df


def feature_graph_order_correlation_local(poser, graph_nw, data_df, obs_labels, weights=None, min_branch_size=None):
    """ Compute correlation between features and branch node ordering.

    Parameters:
    ----------
        poser: `netflow.pose.POSER`
            The object used to construct the POSE.
        graph_nw: `networkx.Graph`
            The POSE graph.
        data_df: `pandas.DataFrame` (n_features, n_observations)
            Feature matrix.
        obs_labels: `list` of `str`
            List of observation labels corresponding to node IDs.
        weights: {`None`, `pandas.DataFrame`, (n, n)}
            Dataframe of edge weights between nodes (observations) used to compute weighted hop distance if provided.
            Unweighted hop count is used if `None`.
        min_branch_size: {`None`, `int`}
            Skip branches with <= ``min_branch_size`` observations.

    Returns:
    ----------
        corr_dict: `dict` of `pandas.DataFrame`.
             Dictionary mapping branch IDs to a `pd.DataFrame`. Each dataframe contains spearman correlation between features
             and global ordering of nodes, and associated p-values.
        
    """
    
    branch_ord_dict = get_branch_node_order(poser, graph_nw, weights=weights, min_branch_size=min_branch_size)
    feat_labels = data_df.index
    corr_dict = {}
    for branch_id, ord_dict in branch_ord_dict.items():

        branch_ord_node_ids = list(ord_dict.keys())
        observation_subset = [obs_labels[node_id] for node_id in branch_ord_node_ids]
        feat_arr = np.array(data_df.loc[:, observation_subset])
        branch_node_ord_arr = np.array(list(ord_dict.values()))
        
        result = spearmanr(feat_arr, branch_node_ord_arr, axis=1)
        corr_dict[branch_id] = pd.DataFrame({'corr':result.correlation[:-1, -1], 'p-val':result.pvalue[:-1,-1]},
                                            index=feat_labels)

    return corr_dict


def ordered_features_correlation_global(poser, graph_nw, data_df, obs_labels, weights=None):
    """ Compute correlation between feature pairs, sorted by global node order.

    Parameters:
    ----------
        poser: `netflow.pose.POSER`
            The object used to construct the POSE.
        graph_nw: `networkx.Graph`
            The POSE graph.
        data_df: `pandas.DataFrame` (n_features, n_observations)
            Feature matrix.
        obs_labels: `list` of `str`
            List of observation labels corresponding to node IDs.
        weights: {`None`, `pandas.DataFrame`, (n, n)}
            Dataframe of edge weights between nodes (observations). If `None` unweighted hop count is used.

    Returns:
    ----------
        corr_arr: `np.ndarray` (n_features, n_features)
            Array of correlations between every pair of features sorted by the global ordering of nodes.
            Node order is based on weighted distance if provided. Otherwise, based on hop distance if `weights` is `None`
    """

    node_ord_dict = get_global_node_order(poser, graph_nw, weights=weights)
    node_ids = list(node_ord_dict.keys())
    obs_ord_labels = [obs_labels[node_id] for node_id in node_ids]
    feat_arr = np.array(data_df.loc[:, obs_ord_labels])
    result = spearmanr(feat_arr, axis=1)
    corr_arr = result.correlation

    return corr_arr


def ordered_features_correlation_branch(poser, graph_nw, data_df, obs_labels, weights=None, min_branch_size=3):
    """ Compute correlations between pairs of features on each branch.

    Parameters:
    ----------
        poser: `netflow.pose.POSER`
            The object used to construct the POSE.
        graph_nw: `networkx.Graph`
            The POSE graph.
        data_df: `pandas.DataFrame` (n_features, n_observations)
            Feature matrix.
        obs_labels: `list` of `str`
            List of observation labels corresponding to node IDs.
        weights: {`None`, `pandas.DataFrame`, (n, n)}
            Dataframe of edge weights between nodes (observations). If `None` unweighted hop count is used.
        min_branch_size: {`None`, `int`}
            Skip branches with <= ``min_branch_size`` observations.

    Returns:
    ----------
        corr_dict: `dict`
            Dictionary of correlations between feature pairs sorted by node order on each branch.
            Node order is based on weighted distance if provided. Otherwise, based on hop distance if `weights` is `None`
    """

    branch_ord_dict = get_branch_node_order(poser, graph_nw, weights=weights, min_branch_size=min_branch_size)
    corr_dict = {}
    for branch_id, ord_dict in branch_ord_dict.items():

        branch_ord_node_ids = list(ord_dict.keys())
        observation_subset = [obs_labels[node_id] for node_id in branch_ord_node_ids]
        feat_arr = np.array(data_df.loc[:, observation_subset])
        result = spearmanr(feat_arr, axis=1)
        corr_dict[branch_id] = result.correlation

    return corr_dict



