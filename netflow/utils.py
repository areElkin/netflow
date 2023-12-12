
import itertools
import fastcluster as fc
from functools import lru_cache
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import Orange.clustering.hierarchical as orange_hier
import pandas as pd
import scipy as sc
import scipy.cluster.hierarchy as sch
import scipy.sparse.csgraph as scg
import scipy.spatial.distance as ssd
import scipy.stats as sc_stats
import seaborn as sns
from sklearn.utils import check_random_state
from typing import (
    TYPE_CHECKING,
    TypedDict,
    Union,
    Optional,
    Any,
    NamedTuple,
    Literal,
)
from tqdm import tqdm


from ._logging import logger

# e.g. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
AnyRandom = Union[None, int, np.random.RandomState]  # maybe in the future random.Generator

EPS = 1e-7
cache_maxsize = 1000000


def interaction_iterator(G: nx.Graph, interactions: object = None):
    """
    Parameters
    ----------
    G : networkx graph
        The graph.
    interactions : {None, iterable, 'all'}
    """
    if isinstance(interactions, str):
        if interactions == 'all':
            N = G.number_of_nodes()
            nargs = int(0.5 * N * (N - 1))
            if nargs > 25000:
                # logger.warning(
                #     'Over 25,000 interactions in network, reducing to {} edges'.format(G.number_of_edges()))
                iterator = iter(G.edges())
                # for interaction in G.edges():
                #     yield interaction
            else:
                iterator = itertools.combinations(G.nodes(), 2)
                # for interaction in itertools.combinations(G.nodes(), 2):
                #     yield interaction
        else:
            raise ValueError("Unrecognized value for interactions, must be one of [list, None, 'all'].")

    elif hasattr(interactions, '__iter__'): # isinstance(interactions, list):
        iterator = interactions
        # for interaction in interactions:
        #     yield interaction

    elif interactions is None:
        iterator = iter(G.edges())
        # for interaction in G.edges():
        #     yield interaction

    else:
        raise ValueError("Unrecognized value for interactions, must be one of [list, None, 'all'].")

    for interaction in iterator:
        yield interaction



def compute_graph_distances(G, weight='weight'):
    """ Returns the weighted hop distance matrix for graph with nodes indexed from
    :math:`0, 1, ..., n-1` where :math:`n` is the number of nodes..

    Parameters
    ----------
    G : `networkx.Graph`
        The graph with nodes assumed to be labeled consecutively from
        :math:`0, 1, ..., n-1` where :math:`n` is the number of nodes.
    weight : `str`, optional
        Edge attribute of weights used for computing the weighted hop distance.
        If `None`, compute the unweighted distance. That is, rather than minimizing the sum of weights over
        all paths between every two nodes, minimize the number of edges.

    Returns
    -------
    dist : numpy ndarray
        An n x n matrix of node-pairwise graph distances between the n nodes.
    """
    # try: 
    #     dist = scg.dijkstra(nx.adjacency_matrix(G, weight=weight, nodelist=list(range(G.number_of_nodes()))),
    #                         directed=True, unweighted=False)
    # except FutureWarning as e:
    #     logger.msg(f"{e.message}: {e.args}")
    # except Exception as e:
    #     logger.msg(f"{e.message}: {e.args}")
    #     raise AssertionError("Distance matrix unable to be computed.")

    # return dist
    return scg.dijkstra(nx.adjacency_matrix(G, weight=weight, nodelist=list(range(G.number_of_nodes()))),
                            directed=True, unweighted=False)

        
    
def heat_kernel(profile, laplacian, timestep):    
    """Compute the action of the matrix exponential of :math:`-timestep * laplacian` on the 
    profile, a.k.a., the result of the action .math::`exp(-tL)*profile`.

    Parameters
    ----------
    profile : `numpy.ndarray`, (n,)
        Feature profile.
    """
    return sc.sparse.linalg.expm_multiply(-timestep * laplacian, profile.T)        

@lru_cache(cache_maxsize)
def pij(G, source, target, n_weight="weight", EPS=1e-7):
    """ Compute the 1-step Markov transition probability of going from source to target node in G
    Note: not a lazy walk (i.e. alpha=0)"""
    # assert G.nodes[source][n_weight] > EPS, f"Node {source} with weight < EPS does not interact with any other nodes."
    if G.nodes[source][n_weight] <= EPS:
        logger.msg(f"Node {source} with weight < EPS does not interact with any other nodes - setting pij = 0.")
        return 0.0
    if target not in G.neighbors(source):
        return 0.0
    if G.nodes[target][n_weight] <= EPS:
        logger.msg(f"Node {source} does not interact with {target} with weight < EPS - setting pij = 0.")
        return 0.0
    w = [G.nodes[nbr][n_weight] for nbr in G.neighbors(source)]    
    if sum(w) > EPS:  # ensure no dividing by zero
        return G.nodes[target][n_weight]/sum(w)
    else:  # ensure no dividing by zero
        raise ValueError(f"Net weight for neighbors of {source} is too small to compute interaction probability with {target}.")


def compute_edge_weights(G: nx.Graph, n_weight="weight", e_weight="weight", e_normalized=True, e_sqrt=True, e_wprob=False):
    """ compute edge weights from given nodal weights 
    e_normalized = True AND e_sqrt = True : w_ij = 1/sqrt([(p_ij + p_ji)/2])
    e_normalized = True AND e_sqrt = False : w_ij = 1/[(p_ij + p_ji)/2]
    e_normalized = False AND e_sqrt = True : w_ij = 1/sqrt(w_i * w_j)
    e_normalized = False AND e_sqrt = False : w_ij = (1/w_i)*(1/w_j)    
    NOTE: w_ij = INF if w_i=0 or w_j=0
    e_wprob = True (then e_normalizezd and e_sqrt are ignored) : w_ij = 1 / (p_ij + p_ji - (p_ij * p_ji))
    """
    assert ~(not nx.get_node_attributes(G, n_weight)), "Node weight not detected in graph."
    
    # compute edge weight
    weights = {}
    if e_normalized or e_wprob:  # normalized
        for i, j in G.edges():
            wij = pij(G, i, j, n_weight)
            wji = pij(G, j, i, n_weight)
            if e_wprob:
                w = wij + wji - (wij * wji)
            else: # normalized
                w = (wij+wji)/2  # d(i,j) = 1/sqrt(w_ij)
                if e_sqrt:  # d(i,j) = 1/sqrt(w_ij)
                    w = np.sqrt(w)
            weights[(i, j)] = 1/w if min([w, wij, wji]) > EPS else np.inf
    else:  # not normalized
        for i, j in G.edges():
            w = G.nodes[i][n_weight]*G.nodes[j][n_weight]  # w = (1/w_i)*(1/w_j)
            if e_sqrt:  # w_ij = 1/sqrt(w_i * w_j)"
                w = np.sqrt(w)
            weights[(i, j)] = 1/w if w > EPS else np.inf
    nx.set_edge_attributes(G, weights, name=e_weight)


def get_times(times=None, t_min=-1.5, t_max=1.0, n_t=20, log_time=True):
    """ get array of simulation time-points

    Parameters
    ----------
    times : {`None`, `numpy.ndarray`}
        Array of times to evaluate the diffusion simulation.
        Note, if given, ``t_min``, ``t_max`` and ``n_t`` are ignored.
    t_min : `float`
        First time point to evaluate the diffusion simulation.
        Note, ``t_min`` is ignored if ``times`` is not `None`.
    t_max : `float`
        Last time point to evaluate the diffusion simulation.
        Note, ``t_max`` must be greater than ``t_min``, i.e, :math:`t_max > t_min`
        and ``t_max`` is ignored if ``times`` is not `None`.
    n_t : `int`
        Number of time points to generate.
        Note, ``n_t`` is ignored if ``times`` is not `None`.
    log_time : `bool`
        If `True`, return ``n_t`` numbers spaced evenly on a log scale, where the time
        sequence starts at ``10 ** t_min``, ends with ``10 ** t_max``, and the
        sequence of times if of the form ``10 ** t`` where ``t`` is the `n_t`
        evenly spaced points between (and including) ``t_min`` and ``t_max``.
        For example, ``_get_times(t_min=1, t_max=3, n_t=3, log_time=True) = array([10 ** 1, 10 ** 2, 10 ** 3]) = array([10., 100., 1000.])``.
        If `False`, return ``n_t`` numbers evenly spaced on a linear scale, where the sequence
        starts at ``t_min`` and ends with ``t_max``.
        For example, ``_get_times(t_min=1, t_max=3, n_t=3, log_time=False) = array([1. ,2., 3.])``.
    """
    if times is not None:
        return times
    else:        
        assert n_t > 0, "At least one time step is required for the dynamic simulation."
    if log_time:
        assert t_min < t_max, "Initial simulation time must be before the final time (t_min < t_max)."
        return np.logspace(t_min, t_max, n_t)
    else:
        assert 0 < t_min < t_max, "Linearly spaced simulation time-points must satisfy 0 < t_min < t_max."
        return np.linspace(t_min, t_max, n_t)



def construct_anisotropic_laplacian_matrix(G, weight, use_spectral_gap=True):
    """ Returns transpose of random walk (rw) Laplacian matrix from Transition matrix
    constructed from node attribute 'weight' for anisotropic diffusion.

    .. note::

       :math:`A` is the binary (symmetric) adjacency matrix,
    
       :math:`w` is the array of node weights,
    
       :math:`D` is the diagonal matrix of the node-weighted degrees where :math:`D_{ii} = \sum_{j~i} w_j`, and
    
       :math:`P = [p_{ij}]` is the transition matrix where :math:`p_{ij}` is the probability associated with
       transitioning from node :math:`i` to node :math:`j` defined as

       .. math::

             p_{ij} = \\frac{w_j}{D_{ii}}, \\, if \\, i \\sim j
    
                                        0, \\, otherwise.

    
    .. note::

       The graph Laplacian (:math:`L`) and graph random-walk Laplacian (:math:`L_{rw}`) are then defined as:
    
       .. math::

           L = D-A

           P = D^{-1}A

           L_{rw} = D^{-1}L = I - D^{-1}A = I - P

    .. note:: The transpose of the random-walk Laplacian is returned, :math:`L_{rw}^T`.

    """
    assert nx.get_node_attributes(G, weight), \
        f"Node attribute '{weight}' not found in graph, Anisotropic adjacency  matrix not constructed."
    nl = list(range(G.number_of_nodes()))
    node_weights = np.array([G.nodes[i][weight] for i in nl])
    A = nx.adjacency_matrix(G, nodelist=nl, weight=None)

    # standard formulation:
    # W = A.multiply(node_weights)
    # degrees = np.asarray(W.sum(axis=1)).squeeze()
    # Dinv = sc.sparse.diags(1.0 / degrees)
    # P = Dinv.dot(W)
    # laplacian = (sc.sparse.eye(G.number_of_nodes()) - P).transpose()

    # transpose formulation:
    W = sc.sparse.diags(node_weights).dot(A)
    degrees = np.asarray(W.sum(axis=0)).squeeze()    
    Dinv = sc.sparse.diags(1.0 / degrees)
    P = W.dot(Dinv)
    laplacian = sc.sparse.eye(G.number_of_nodes()) - P

    if use_spectral_gap and len(G) > 3:
        spectral_gap = abs(sc.sparse.linalg.eigs(laplacian, which="SM", k=2)[0][1])
        logger.debug("Spectral gap = 10^{:.1f}".format(np.log10(spectral_gap)))
        laplacian /= spectral_gap

    return laplacian
    

    
def clustermap(data, observations=None, features=None, transform=True, is_symmetric=True, optimal_nodes_ordering=True,
               linkage_kwargs={'method': 'ward', 'metric': 'euclidean'}, title="",
               vis_kwargs={'cmap': 'RdBu', 'center': 0, 'figsize': (7, 7),
                           'dendrogram_ratio': 0.07, 'cbar_pos': (-0.02, 0.84, 0.02, 0.16)}):
    """
    Parameters
    ----------
    data : `pandas.DataFrame`, (m, n)
        A dataframe with :math:`m` observations that are :math:`n` dimensional.
    observations : `list`
        Rows in ``data`` to use. If `None`, all rows are used.
    features : `list`
        Columns in ``data`` to use. If `None`, all columns are used.
    transform : `bool`
        If `True`, translate ``data`` with specified ``observations`` and ``features``
        by feature mean and divide by feature standard deviation.        
    is_symmetric : `bool`
        If `True`, treat ``data`` as symmetric and use the same clustering on rows and columns.
        Otherwise, if `False`, perform independent clustering and ordering on rows and columns.
    optimal_nodes_ordering : `bool`
        If `True`, optimize node ordering.
    linkage_kwargs : `dict`
        Key-word arguments passed to ``fastcluster.linkage``.
    title : `str`
        Figure title.
    vis_kwargs : `dict`
        Key-word arguments passed to ``seaborn.clustermap``.

    Returns
    -------
    cm : `seaborn.matrix.ClusterGrid`
        The plotted clustermap.
    """
    
    observations = data.index.tolist() if observations is None else observations
    features = data.columns.tolist() if features is None else features

    data = data.loc[observations, features]

    if transform:
        data = (data - data.mean(axis=0)) / data.std(axis=0)

    obs_linkage = fc.linkage(data, **linkage_kwargs)
    if optimal_nodes_ordering:
        tree = orange_hier.tree_from_linkage(obs_linkage)
        tree = orange_hier.optimal_leaf_ordering(tree, ssd.squareform(ssd.pdist(data)))
        obs_linkage = orange_hier.linkage_from_tree(tree)

    if is_symmetric:
        feat_linkage=obs_linkage
    else:
        feat_linkage = fc.linkage(data.transpose(), **linkage_kwargs)
        if optimal_nodes_ordering:
            tree = orange_hier.tree_from_linkage(feat_linkage)
            tree = orange_hier.optimal_leaf_ordering(tree, ssd.squareform(ssd.pdist(data.transpose())))
            feat_linkage = orange_hier.linkage_from_tree(tree)

    cm = sns.clustermap(data.transpose(), row_linkage=feat_linkage, col_linkage=obs_linkage, **vis_kwargs)
    plt.gcf().suptitle(title, y=1.05)

    return cm

    
def spearmanr_(data, **kwargs):
    """ Calculate a Spearman correlation coefficient with associated p-value using scipy.stats.spearmanr.

    Parameters
    ----------
    data : `numpy.ndarray`, (n_observations, n_features)
        2-D array containing multiple variables and observations, where each column represents
        a variable, with observations in the rows.
    **kwargs : `dict`
        Optional key-word arguments passed to ``scipy.stats.spearmanr``.

    Returns
    -------
    R : `pandas.DataFrame`
        Spearman correlation matrix. The correlation matrix is square with
        length equal to total number of variables (columns or rows).
    pvalue : `float`
        The p-value for a hypothesis test whose null hypotheisis
        is that two sets of data are uncorrelated. See documentation for scipy.stats.spearmanr
        for alternative hypotheses. ``pvalue`` has the same
        shape as ``R``.
    """
    stats = sc_stats.spearmanr(data, axis=0, **kwargs)
    R = pd.DataFrame(data=stats.correlation, index=data.columns.copy(), columns=data.columns.copy())
    p = pd.DataFrame(data=stats.pvalue, index=data.columns.copy(), columns=data.columns.copy())
    return R, p

def kendall_tau_(data, **kwargs):
    """ Calculate a Kendall's tau correlation coefficient with associated p-value using ``scipy.stats.kendalltau``.

    Parameters
    ----------
    data : `numpy.ndarray`, (n_observations, n_features)
        2-D array containing multiple variables and observations, where each column represents
        a variable, with observations in the rows.
    **kwargs : `dict`
        Optional key-word arguments passed to ``scipy.stats.spearmanr``.

    Returns
    -------
    R : `pandas.DataFrame`
        Spearman correlation matrix. The correlation matrix is square with
        length equal to total number of variables (columns or rows).
    pvalue : `float`
        The p-value for a hypothesis test whose null hypotheisis
        is that two sets of data are uncorrelated. See documentation for ``scipy.stats.kendalltau``.
        for alternative hypotheses. ``pvalue`` has the same
        shape as ``R``.
    """
    R = {k: {} for k in data.columns}
    p = {k: {} for k in data.columns}

    for k in data.columns:
        R[k][k] = 1.
        p[k][k] = 0.

    for i, j in tqdm(itertools.combinations(data.columns, 2), total=data.shape[1]*(data.shape[1]-1)/2, leave=False):
        stats = sc_stats.kendalltau(data[i].values, data[j].values)
        R[i][j] = R[j][i] = stats.correlation
        p[i][j] = p[j][i] = stats.pvalue

    R = pd.DataFrame.from_dict(R)
    R = R.loc[data.columns, data.columns]
    p = pd.DataFrame.from_dict(p)
    p = p.loc[data.columns, data.columns]
    
    return R, p


def stack_triu_(df, name=None):
    """ Stack the upper triangular entries of the dataframe above the diagonal
    .. note::

       Useful for symmetric dataframes like correlations or distances.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Dataframe to stack. 
        Note, upper triangular entries are taken from ``df`` as provided,
        with no check that the rows and columns are symmetric.
    name : `str`
        Optional name of pandas Series output ``df_stacked``.

    Returns
    -------
    df_stacked : `pandas.Series`
        The stacked upper triangular entries above the diagonal of the dataframe.
    """
    df_stacked = df.stack()[np.triu(np.ones(df.shape).astype(bool), 1).reshape(df.size)]
    df_stacked.name = name
    return df_stacked


def stack_triu_where_(df, condition, name=None):
    """ Stack the upper triangular entries of the dataframe above the diagonal where the condition is True
    .. note::

       Useful for symmetric dataframes like correlations or distances.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Dataframe to stack. 
        Note, upper triangular entries are taken from ``df`` as provided,
        with no check that the rows and columns are symmetric.
    condition : `pandas.DataFrame`
        Boolean dataframe of the same size and order of rows and columns as ``df`` indicating 
        values, where `True`, to include in the stacked dataframe.
    name : `str`
        Optional name of pandas Series output ``df_stacked``.

    Returns
    -------
    df_stacked : `pandas.Series`
        The stacked upper triangular entries above the diagonal of the dataframe,
        where ``condition`` is `True`.        
    """        
    df_stacked = df.stack()[np.triu(condition.astype(bool), 1).reshape(df.size)]
    df_stacked.name = name
    return df_stacked


def unstack_triu_(series, diag=0., index=None):
    """ Unstack pandas Series with upper triangular entries to symmetric matrix.

    Parameters
    ----------
    series : `pandas.Series`
        The stacked upper triangular entries to be unstacked.
    diag : `float`
        The value to be used on the diagonal.
    index : list-like, optional
        If provided, return unstacked matrix with rows and columns sorted by index.
        If `None`, rows and columns are alphebetically sorted.

    Returns
    -------
    M : `pandas.DataFrame`
        Symmetric unstacked matrix with ``diag`` on the diagonal.
    """
    index = sorted(set(itertools.chain(*series.index))) if index is None else index
    M = pd.concat([series, pd.Series(data=[diag]*len(index), index=[(k,k) for k in index]),
                   pd.Series(data=series.values, index=[(k[1], k[0]) for k in series.index])], axis=0).unstack().loc[index, index]
    return M


def dispersion_(data, axis=0):
    """ Data dispersion computed as the absolute value of the variance-to-mean ratio where the 
    variance and mean is computed on the values over the requested axis.

    Parameters
    ----------
    data : `pandas.DataFrame`
        Data used to compute dispersion.
    axis : {0, 1}
        Axis on which the variance and mean is applied on computed.

        Options : 

        - 0 : for each column, apply function to the values over the index
        - 1 : for each index, apply function to the values over the columns

    Returns
    -------
    vmr : `pandas.Series`
        Variance-to-mean ratio (vmr) quantifying the disperion.
    """
    vmr = np.abs(data.var(axis=axis) / data.mean(axis=axis))
    return vmr


def compute_eigen(T,
                  n_comps: int = 15,
                  sort: Literal["decrease", "increase"] = "decrease",
                  random_state: AnyRandom = 0,
                  ):
    """ Compute eigen decomposition of (transition) matrix.

    Code taken from ``scanpy.neighbors.__init__.py``
    
    Parameters
    ----------
    T : `numpy.ndarray`, (n, n)
        Matrix that eigen decomposition will be computed on (likely the transition matrix).
    n_comps
        Number of eigenvalues/vectors to be computed, set ``n_comps = 0`` to compute the whole spectrum.
        Alternatively, if set ``n_comps >= n``, the whole spectrum will be computed.
    sort : {"decrease", "increase"}
        Order to sort eigenvalues.
    random_state
        A numpy random seed.

    Returns
    -------
    eigen_values : numpy.ndarray
        Eigenvalues of transition matrix.
    eigen_basis : numpy.ndarray
         Matrix of eigenvectors (stored in columns).  ``.eigen_basis`` is
         projection of data matrix on right eigenvectors, that is, the
         projection on the diffusion components.  these are simply the
         components of the right eigenvectors and can directly be used for
         plotting.
    """
    if (n_comps == 0) or (n_comps >= T.shape[0]):
        evals, evecs = sc.linalg.eigh(T)
        logger.debug("Computing full eigen spectrum.")
    else:
        n_comps = min(T.shape[0] - 1, n_comps)
        logger.debug(f"Computing {n_comps} components of the eigen spectrum.")

        # it pays off to increase the stability with a bit more precision
        T = T.astype(np.float64)

        # Setting the random initial vector
        random_state = check_random_state(random_state)
        v0 = random_state.standard_normal(T.shape[0])

        which = "LM" if sort == "decrease" else "SM"
        
        evals, evecs = sc.sparse.linalg.eigsh(T, k=n_comps, which=which,
                                              ncv=None, v0=v0)
        evals, evecs = evals.astype(np.float32), evecs.astype(np.float32)

    if sort == "decrease":
        evals = evals[::-1]
        evecs = evecs[:, ::-1]

    logger.info(
        "    eigenvalues of matrix\n"
        "    {}".format(str(evals).replace("\n", "\n    "))
    )

    # if self._number_connected_components > len(evals) / 2:
    #     logg.warning("Transition matrix has many disconnected components!")
    return evals, evecs


def gauss_window(window_size=5, smoothness=2.5):
    """ Gaussian window of width ``window_size``.

    Parameters
    ----------
    window_size : `int`, default = 5
        Window size.
    smoothness : `float`, default = 2.5
        Smoothness of curve.

    Returns
    -------
    gauss : `numpy.ndarray`
        Gaussian window.
    """
    window = np.arange(-(window_size-1), (window_size-1)+1, 2)    
    gauss = np.exp(-0.5 * np.power(smoothness/window_size * window, 2))
    return gauss


def gauss_conv(array, window_size=5, smoothness=2.5):
    """ Smooth an array using Gaussian kernel.

    Parameters
    ----------
    array : `numpy.ndarray`
        Array that will be smoothed.
    window_size : `int`, default = 5
        Window size.
    smoothness : `float`, default = 2.5
        Smoothness of curve.

    Returns
    -------
    smoothed_array : `numpy.ndarray`
        The smoothed array.
    """
    gauss_filter = gauss_window(window_size=window_size, smoothness=smoothness)
    gauss_filter = gauss_filter / gauss_filter.sum() # normalize

    smoothed_array = np.convolve(array, gauss_filter, mode='full')
    return smoothed_array
    
    
    
    
