from functools import lru_cache, partial
import itertools
import multiprocessing as mp
import numpy as np
import cvxpy as cvx
import ot
import pandas as pd
import heapq
from scipy.optimize import linprog
import scipy.spatial as ss
from scipy import sparse
import scipy

import netflow.utils as utl
from .._logging import logger


linprog_status = {0:'Optimization proceeding nominally.',
                  1:'Iteration limit reached.',
                  2:'Problem appears to be infeasible.',
                  3:'Problem appears to be unbounded.',
                  4:'Numerical difficulties encountered.'}



def optimal_transportation_distance(x, y, d, solvr=None):    
    """ Compute the optimal transportation distance (OTD) of the given density distributions by CVXPY.

    Parameters
    ----------
    x : `numpy.ndarray`, (m,)
        Source's density distributions, includes source and source's neighbors.
    y : `numpy.ndarray`, (n,) 
        Target's density distributions, includes source and source's neighbors.
    d : `numpy.ndarray`, (m, n) 
        Shortest path matrix.

    Returns
    -------
    m : `float`
        Optimal transportation distance.
    """
    rho = cvx.Variable((len(y), len(x)))  # the transportation plan rho
    # objective function d(x,y) * rho * x, need to do element-wise multiply here
    obj = cvx.Minimize(cvx.sum(cvx.multiply(np.multiply(d.T, x.T), rho)))
    # \sigma_i rho_{ij}=[1,1,...,1]
    source_sum = cvx.sum(rho, axis=0, keepdims=True)
    # constrains = [rho * x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    constrains = [rho @ x == y, source_sum == np.ones((1, (len(x)))), 0 <= rho, rho <= 1]
    prob = cvx.Problem(obj, constrains)
    if solvr is None:
        m = prob.solve()
    else:
        m = prob.solve(solver=solvr)
        # m = prob.solve(solver='ECOS', abstol=1e-6,verbose=True)
        # m = prob.solve(solver='OSQP', max_iter=100000,verbose=False)
    return m


def OTD(x, y, d, solvr=None, flag=None):
    """ Compute the optimal transportation distance (OTD) of the given density distributions
    trying first with POT package and then by CVXPY.

    Parameters
    ----------
    x : (m,) numpy.ndarray
        Source's density distributions, includes source and source's neighbors.
    y : (n,) numpy.ndarray
        Target's density distributions, includes source and source's neighbors.
    d : (m, n) numpy.ndarray
        Shortest path matrix.
    flag : str, optional
        Optional flag included in logger output to identify inputs.
        

    Returns
    -------
    m : float
        Optimal transportation distance.
    """

    flag = "" if flag is None else flag+": "

    try:
        wasserstein_distance, lg = ot.emd2(x, y, d, log=True)
        if lg['warning'] is not None:
            logger.msg(f"{flag}POT library failed: warning = {lg['warning']}, retry with explicit computation")
            wasserstein_distance = optimal_transportation_distance(x, y, d)
    except cvx.error.SolverError:
        logger.msg(f"{flag}OTD failed, retry with SCS solver")
        wasserstein_distance = optimal_transportation_distance(x, y, d, solvr='SCS')
    return wasserstein_distance





########


def wass_distance(observations, profiles, graph_distances, measure_cutoff=1e-6,
                  solvr=None, flag=None):
    """ Compute Wasserstein distance between profiles of two observations.

    .. note: Here, the ``profiles`` include many observations, not just those of interest.
             Use ``observations`` to specify which two observations in ``profiles`` the
             the distance should be computed between.

             TO DO: This requires a lot more memory, could be modified to only take
             profiles of interest. 


    Parameters
    ----------
    observations : 2-tuple
        Profile columns names referring to the two observations that the Wasserstein distance should be computed between.
    profiles : `pandas.DataFrame`
        Observation profiles with features as rows and observations as columns.
    graph_distances : `numpy.ndarray`, (n', n')
        A matrix of node-pairwise graph distances between the :math:`n'` nodes ordered by the rows in ``profiles``.
    measure_cutoff : `float`
        Threshold for treating values in profiles as zero, default = 1e-6.

    Returns
    -------
    wasserstein_distance : `float`
        The Wasserstein distance.
    """
    observation_a, observation_b = observations
    if flag is None:
        flag = f"{observation_a} - {observation_b}: "
    else:
        flag = flag + ': ' + f"{observation_a} - {observation_b}: "
    m_a, m_b = profiles[observation_a].values, profiles[observation_b].values

    Na = np.where(m_a >= measure_cutoff)[0] #  * np.max(m_a))[0]
    Nb = np.where(m_b >= measure_cutoff)[0] #  * np.max(m_y))[0]

    if Na.shape[0] == Nb.shape[0] == 0:
        logger.msg(f"{flag}Profiles are both zero, returning Wasserstein distance = 0.")
        return 0.
    # elif Na.shape[0] == Nb.shape[0] == 1:
    #     logger.msg(f"{flag}Both profiles with only one feature, returning Wasserstein distance = 0.")
    #     return 0.
    elif (Na.shape[0] == 0) or (Nb.shape[0] == 0):
        logger.msg(f"{flag}Cannot compute Wasserstein distance between a zero-profile, returning Wasserstein distance = nan.")
        return np.nan

    m_a, m_b = m_a[Na], m_b[Nb]
    m_a /= m_a.sum()
    m_b /= m_b.sum()

    # distances_ab = graph_distances[np.ix_(profiles.index.tolist(), profiles.index.tolist())]
    # distances_ab = distances_ab[np.ix_(Na, Nb)]
    distances_ab = graph_distances[np.ix_(Na, Nb)]

    wasserstein_distance = OTD(m_a, m_b, distances_ab, solvr=solvr, flag=flag)

    return wasserstein_distance


def pairwise_observation_euc_distances(profiles, metric='euclidean', **kwargs):
     """ Compute observation-pairwise Euclidean distances between the profiles.
     
     Parameters
     ----------
     profiles : `pandas.DataFrame`, (n_features, n_observations)
         Profiles that Euclidean distance is computed between.
     metrics : `str` or callable, optional
         The distance metric to use passed to `scipy.spatial.distance.cdist`.
     **kwargs : `dict`, optional
         Extra arguments to metric, passed to `scipy.spatial.distance.cdist`.

     Returns
     -------
     ed : pandas Series
         Euclidean distances between pairwise observations
     """
     n = profiles.shape[1]     
     eds = pd.DataFrame(data=ss.distance.cdist(profiles.T.values, profiles.T.values, metric=metric, **kwargs),
                        index=profiles.columns.tolist(), columns=profiles.columns.tolist())
     eds = eds.stack()[np.triu(np.ones(eds.shape), 1).astype(bool).reshape(eds.size)]
     eds = eds.reset_index().rename(columns={'level_0': 'observation_a', 'level_1': 'observation_b'}).set_index(['observation_a', 'observation_b'])
     eds = eds[0]

     return eds


def pairwise_observation_wass_distances(profiles, graph_distances, proc=mp.cpu_count(), chunksize=None,
                                   measure_cutoff=1e-6, solvr=None, flag=None):
    """ Compute observation-pairwise Wasserstein distances between the profiles on a fixed weighted graph.

    Parameters
    ----------
    profiles : `pandas.DataFrame`, (n_features, n_observations)
        Profiles that are normalized and treated as probability distributions for computing
        Wasserstein distance.
    graph_distances : `numpy.ndarray`, (n, n)
        A matrix of node-pairwise graph distances between the n nodes (ordered from 0, 1, ..., n-1).
    measure_cutoff : `float`
        Threshold for treating values in profiles as zero, default = 1e-6.
    proc : `int`
        Number of processor used for multiprocessing. (Default value = cpu_count()). 
    chunksize : `int`
        Chunksize to allocate for multiprocessing.
    solvr : `str`
        Solver to pass to POT library for computing Wasserstein distance.

    Returns
    -------
    wd : pandas Series
        Wasserstein distances between pairwise observations
    """
    n = profiles.shape[1]
    # wds = np.zeros([1, int(0.5 * n * (n-1))])
    # print(f"wds = {wds.shape}.")
    # print(f"wds[0] = {wds[0].shape}.")
    with mp.Pool(proc) as pool:
        if chunksize is None:            
            # logger.msg(f"Profiles shape = {n}")
            chunksize = int(np.round(max(1, 0.5 * n * (n-1) / proc)))
            # chunksize, extra = divmod(0.5 * n * (n-1), proc * 4)
            # if extra:
            #     chunksize += 1
                
            # logger.msg(f"Chunksize = {chunksize}.")

        # def wass_distance(observations, profiles, graph_distances, measure_cutoff=1e-6,
        #           solvr=None):
        wds = pool.map(partial(wass_distance,
                               profiles=profiles,
                               graph_distances=graph_distances,
                               measure_cutoff=measure_cutoff,
                               solvr=solvr,
                               flag=flag),
                       list(itertools.combinations(profiles.columns.tolist(), 2)),
                       chunksize=chunksize)

        # logger.msg(f"wds = {wds}.")
    wd = pd.Series(data=wds,
                   index=pd.MultiIndex.from_tuples(itertools.combinations(profiles.columns.tolist(), 2),
                                                   names=['observation_a', 'observation_b']))

    return wd

                                               
# def pairwise_observation_neighborhood_wass_distance(features, graph_distances,
#                                                proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000):
#     for a,b in itertools.:
#         with mp.get_context('fork').Pool(processes=_proc) as pool:
#         chunksize, extra = divmod(nargs, _proc * 4)
#         if extra:
#             chunksize += 1
#         result = pool.imap_unordered(_wrap_compute_wasserstein_edge, args, chunksize=chunksize)
#         pool.close()
#         pool.join()

#     output = {}
#     for K in result:
#         for k in list(K.keys()):
#             output[(k[0], k[1])] = K[k]

#     return output
    

# used to be def norm_features_
def norm_features_(X, method='L1'):
    """ Norm of multi-feature data points.

    Intended to compute the norm of pairwise distances between observations :math:`s_q` 
    and :math:`s_r` for all features
    :math:`f_i \in F`\: :math:`D_{qr}^{(F)} = [d_{qr}^{(f_1)}, ..., d_{qr}^{(f_m)}]`

    The multi-feature distances array is of the form :math:`D^{(F)} = [d_{ij}]` of size
    (n_observations * (n_observations - 1) / 2, n_features).
    :math:`F` is the set of features where :math:`|F| = ` n_features, with a multi-index of size 2
    of the form :math:`(obs_i, obs_j)` and :math:`d_{ij}` is the distance between
    pairwise-observations
    :math:`i = (obs_p, obs_q)` with respect to feature :math:`j`.

    Parameters
    ----------
    X : `pandas.DataFrame`, (n_observations * (n_observations - 1) / 2, n_features)
        The matrix of  multiple pairwise-observation distances.
        The norm is computed on the rows of ``X``.
    method : {'L1', 'L2', 'inf', 'mean', 'median'}
        Indicate which norm to compute. For each row of the form :math:`x = [x_1, x_2, ..., x_n]` :

        Options:

        - 'L1' : :math:`\sum_{i=1}^n abs(x_i)`
        - 'L2' : :math:`\sqrt{\sum_{i=1}^n (x_i)^2}`
        - 'inf' : :math:`max_i abs(x_i)`
        - 'mean' : Mean of :math:`x`
        - 'median' : Median of :math:`x`

    Returns
    -------
    n : `float` or array-like 
        Norm of the row(s).
        When not a `float`, If ``X`` is a `pandas.DataFrame`, ``n`` is a `pandas.Series`.
        Otherwise, if ``X`` is a `numpy.ndarray`, ``n`` is a `numpy.ndarray` vector.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        index = X.index.copy()
        X = X.values
    else:
        index = None
        
    if method == 'L1':
        n = np.linalg.norm(X, axis=1, ord=1, keepdims=False)
    elif method == 'L2':
        n = np.linalg.norm(X, axis=1, ord=2, keepdims=False)
    elif method == 'inf':
        n = np.linalg.norm(X, axis=1, ord=np.inf, keepdims=False)
    elif method == 'mean':
        n = np.mean(X, axis=1, keepdims=False)
    elif method == 'median':
        n = np.median(X, axis=1, keepdims=False)
    else:
        msg = f"{method!r} not recognized for value of method, must be one of ['L1', 'L2', 'inf', 'mean', 'median']."
        raise AssertionError(msg)
    logger.debug("computed norm")

    if index is not None:
        n = pd.Series(data=n, index=index)
    return n

    
def norm_features(keeper, key, features=None, method='L1', label=None):
    """ Norm of multi-feature data points in keeper.

    Intended to compute the norm of pairwise distances between observations :math:`s_q` 
    and :math:`s_r` for all features
    :math:`f_i \in F`\: :math:`D_{qr}^{(F)} = [d_{qr}^{(f_1)}, ..., d_{qr}^{(f_m)}]`

    The multi-feature distances array is of the form :math:`D^{(F)} = [d_{ij}]` of size
    (n_observations * (n_observations - 1) / 2, n_features).
    :math:`F` is the set of features where :math:`|F| = ` n_features, with a multi-index of size 2
    of the form :math:`(obs_i, obs_j)` and :math:`d_{ij}` is the distance between
    pairwise-observations
    :math:`i = (obs_p, obs_q)` with respect to feature :math:`j`.

    If ``label`` is provided, the resulting norm is stored in ``keeper.misc[label]``.
    Otherwise, if ``label`` is `None`, the resulting norm is returned.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object that stores the multiple pairwise-observation distances.
    key : `str`
        The label used to reference the multiple pairwise-observation distances stored in ``keeper.misc``,
        of size (n_observations * (n_observations - 1) / 2, n_features).
        The norm is computed on the rows of ``keeper.misc[key]``.    
    features : {`None`, `List` [`str`]}
        Subset of features to include. If provided, restrict to norm over columns corresponding to features
        in the specified list. If `None`, use all columns.
    method : {'L1', 'L2', 'inf', 'mean', 'median'}
        Indicate which norm to compute. For each row of the form :math:`x = [x_1, x_2, ..., x_n]` :
    
        Options:

        - 'L1' : :math:`\sum_{i=1}^n abs(x_i)`
        - 'L2' : :math:`\sqrt{\sum_{i=1}^n (x_i)^2}`
        - 'inf' : :math:`max_i abs(x_i)`
        - 'mean' : Mean of :math:`x`
        - 'median' : Median of :math:`x`
    label : {`None`, `str`}
        Label used to store resulting norm in ``keeper.misc``. If `None`, the resulting norm is returned
        instead of storing it in the keeper.

    Returns
    -------
    n : `pandas.Series`
        Norm of the row(s) of length n_observations * (n_observations - 1) / 2..
    """
    X = keeper.misc[key]
    if features is not None:
        X = X[features]

    Xnorm = norm_features_(X, method=method)

    if label is None:
        return Xnorm

    keeper.add_misc(Xnorm, label)
    return None
    

# used to be def compute_norm_features(self) in organization.pseudo
def norm_features_as_sym_dist(keeper, key, label, features=None, method='L1',
                              is_distance=True):
    """ Construct symmetric distance matrix between observations from the norm of
    multi-feature pairwise-observation distances.

    Intended to compute the norm of pairwise distances between observations :math:`s_q` 
    and :math:`s_r` for all features
    :math:`f_i \in F`\: :math:`D_{qr}^{(F)} = [d_{qr}^{(f_1)}, ..., d_{qr}^{(f_m)}]`

    The multi-feature distances array is of the form :math:`D^{(F)} = [d_{ij}]` of size
    (n_observations * (n_observations - 1) / 2, n_features).
    :math:`F` is the set of features where :math:`|F| = ` n_features, with a multi-index of size 2
    of the form :math:`(obs_i, obs_j)` and :math:`d_{ij}` is the distance between
    pairwise-observations
    :math:`i = (obs_p, obs_q)` with respect to feature :math:`j`.

    The multi-feature pairwise-observations are by default treated as distances,
    but they could be similarities. If they are similarities, set ``is_distance``
    to `False`. Note: similarities will have 1. along the diagonal.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object that stores the multiple pairwise-observation distances.
    key : `str`
        The label used to reference the multiple pairwise-observation distances stored in ``keeper.misc``,
        of size (n_observations * (n_observations - 1) / 2, n_features).
        The norm is computed on the rows of ``keeper.misc[key]``.
    label : `str`
        Label used to store resulting distance matrix. If ``is_distance`` is `True`, the matrix is stored in
        ``keeper.distances[label]``. Otherwise, it is a similarity matrix stored in ``keeper.similarity[label]``.
    features : {`None`, `List` [`str`]}
        Subset of features to include. If provided, restrict to norm over columns corresponding to features
        in the specified list. If `None`, use all columns.
    method : {'L1', 'L2', 'inf', 'mean', 'median'}
        Indicate which norm to compute. For each row of the form :math:`x = [x_1, x_2, ..., x_n]` :
    
        Options:

        - 'L1' : :math:`\sum_{i=1}^n abs(x_i)`
        - 'L2' : :math:`\sqrt{\sum_{i=1}^n (x_i)^2}`
        - 'inf' : :math:`max_i abs(x_i)`
        - 'mean' : Mean of :math:`x`
        - 'median' : Median of :math:`x`
    is_distance : `bool`
        Indicate if the multi-feature pairwise-observations are distances or similarities.
        If `True`, they are treated as distances. If `False, they are treated as similarities.

    """

    Xnorm = norm_features(keeper, key, features=features, method=method, label=None)

    if is_distance:        
        Xdist = utl.unstack_triu_(Xnorm, index=keeper.observation_labels)
        keeper.add_distance(Xdist, label)
    else:  # similarity
        Xdist = utl.unstack_triu_(Xnorm, diag=1., index=keeper.observation_labels)
        keeper.add_similarity(Xdist, label)
                              

