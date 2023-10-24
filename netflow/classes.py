from pathlib import Path

from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt
import multiprocessing as mp
import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial as ss
import scipy.stats as sc_stats
from tqdm import tqdm

from ._logging import logger
from .checks import *
from .metrics import OTD

import netflow.utils as utl
from importlib import reload
reload(utl)
kendall_tau_ = utl.kendall_tau_
# clustermap = utl.clustermap

from .utils import compute_graph_distances, heat_kernel, compute_edge_weights, get_times, \
    construct_anisotropic_laplacian_matrix, clustermap, spearmanr_, kendall_tau_, stack_triu_, \
    stack_triu_where_, dispersion_

import netflow.utils as utl
from importlib import reload
reload(utl)
# heat_kernel = utl.heat_kernel
construct_anisotropic_laplacian_matrix = utl.construct_anisotropic_laplacian_matrix

def wass_distance(samples, profiles, graph_distances, measure_cutoff=1e-6,
                  solvr=None, flag=None):
    """ Compute Wasserstein distance between profiles of two samples.

    Parameters
    ----------
    samples : 2-tuple
        Profile columns names referring to the two samples that the Wasserstein distance should be computed between.
    profiles : pandas DataFrame
        Sample profiles with features as rows and samples as columns.
    graph_distances : numpy ndarray
        An n' x n' matrix of node-pairwise graph distances between the n' nodes ordered by the rows in `profiles`.
    measure_cutoff : float
        Threshold for treating values in profiles as zero, default = 1e-6.

    Returns
    -------
([([(    """
    
    sample_a, sample_b = samples
    if flag is None:
        flag = f"{sample_a} - {sample_b}: "
    else:
        flag = flag + ': ' + f"{sample_a} - {sample_b}: "
    m_a, m_b = profiles[sample_a].values, profiles[sample_b].values

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


def pairwise_sample_euc_distances(profiles, metric='euclidean', **kwargs):
     """ Compute sample-pairwise Euclidean distances between the profiles.
     Parameters
     ----------
     profiles : pandas DataFrame
         Profiles that Euclidean distance is computed between
         where rows are features and columns are samples.
     metrics : tr or callable, optional
         The distance metric to use passed to scipy.spatial.distance.cdist.
     **kwargs : dict, optional
         Extra arguments to metric, passed to scipy.spatial.distance.cdist.

     Returns
     -------
     ed : pandas Series
         Euclidean distances between pairwise samples
     """
     
     n = profiles.shape[1]     
     eds = pd.DataFrame(data=ss.distance.cdist(profiles.T.values, profiles.T.values, metric=metric, **kwargs),
                        index=profiles.columns.tolist(), columns=profiles.columns.tolist())
     eds = eds.stack()[np.triu(np.ones(eds.shape), 1).astype(bool).reshape(eds.size)]
     eds = eds.reset_index().rename(columns={'level_0': 'sample_a', 'level_1': 'sample_b'}).set_index(['sample_a', 'sample_b'])
     eds = eds[0]

     return eds


def pairwise_sample_wass_distances(profiles, graph_distances, proc=mp.cpu_count(), chunksize=None,
                                   measure_cutoff=1e-6, solvr=None, flag=None):
    """ Compute sample-pairwise Wasserstein distances between the profiles.

    Parameters
    ----------
    profiles : pandas DataFrame
        Profiles that are normalized and treated as probability distributions for computing Wasserstein distance,
        where rows are features and columns are samples.
    graph_distances : numpy ndarray
        An n x n matrix of node-pairwise graph distances between the n nodes (ordered from 0, 1, ..., n-1).
    measure_cutoff : float
        Threshold for treating values in profiles as zero, default = 1e-6.
    proc : int
        Number of processor used for multiprocessing. (Default value = cpu_count()). 
    chunksize : int
        Chunksize to allocate for multiprocessing.
    solvr : str
        Solver to pass to POT library for computing Wasserstein distance.


    Returns
    -------
    wd : pandas Series
        Wasserstein distances between pairwise samples
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

        # def wass_distance(samples, profiles, graph_distances, measure_cutoff=1e-6,
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
                                                   names=['sample_a', 'sample_b']))

    return wd

                                               
# def pairwise_sample_neighborhood_wass_distance(features, graph_distances,
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
    

class InfoNet:
    """ A class to compute information flow on a network and correlation between network modules

    Parameters
    ----------
    G : networkx graph
        Simple, connected, undirected graph.
    data : pandas DataFrame
        Data with features as rows and samples as columns.
    outdir : {Path, str}
        Path to store results
    """
    def __init__(self, G, data, outdir):
        check_connected_graph(G)
        check_graph_no_self_loops(G)

        if not nx.get_node_attributes(G, name='name'):
            G = nx.convert_node_labels_to_integers(G, label_attribute='name')

        self.G = G.copy()
        self.name2node = {v: k for k, v in nx.get_node_attributes(self.G, "name").items()}

        self.data = data.rename(index=self.name2node)
        self.features = data.index.tolist()[:]
        self.samples = data.columns.tolist()[:]

        self.meta = {}  # can be used to reference data to be used for subsequent analysis

        self.outdir = outdir if isinstance(outdir, Path) else Path(outdir)
        if self.outdir.is_dir():  # store list of names of saved files
            self.filenames = [k.name for k in self.outdir.iterdir()]
        else:
            logger.msg(f"Creating directory {self.outdir}.")
            self.outdir.mkdir()
            self.filenames = []

    def spearmanr(self, data, **kwargs):
        """ Calculate a Spearman correlation coefficient with associated p-value using scipy.stats.spearmanr.

        Parameters
        ----------
        data : 2D array_like
            2-D array containing multiple variables and observations, where each column represents
            a variable, with observations in the rows.
        **kwargs : dict
            Optional key-word arguments passed to scipy.stats.spearmanr.

        Returns
        -------
        R : pandas DataFrame
            Spearman correlation matrix. The correlation matrix is square with
            length equal to total number of variables (columns or rows).
        pvalue : float
            The p-value for a hypothesis test whose null hypotheisis
            is that two sets of data are uncorrelated. See documentation for scipy.stats.spearmanr
            for alternative hypotheses. `p` has the same
            shape as `R`.
        """
        # stats = sc_stats.spearmanr(data, axis=0, **kwargs)
        # R = pd.DataFrame(data=stats.correlation, index=data.columns.copy(), columns=data.columns.copy())
        # p = pd.DataFrame(data=stats.pvalue, index=data.columns.copy(), columns=data.columns.copy())
        R, p = spearmanr_(data, **kwargs)
        return R, p

    def kendall_tau(self, data, **kwargs):
        """ Calculate Kendall's tau correlation coefficient with associated p-value using scipy.stats.kendalltau.

        Parameters
        ----------
        data : 2D array_like
            2-D array containing multiple variables and observations, where each column represents
            a variable, with observations in the rows.
        **kwargs : dict
            Optional key-word arguments passed to scipy.stats.kendalltau.

        Returns
        -------
        R : pandas DataFrame
            Spearman correlation matrix. The correlation matrix is square with
            length equal to total number of variables (columns or rows).
        pvalue : float
            The p-value for a hypothesis test whose null hypotheisis
            is that two sets of data are uncorrelated. See documentation for scipy.stats.spearmanr
            for alternative hypotheses. `p` has the same
            shape as `R`.
        """
        R, p = kendall_tau_(data, **kwargs)
        return R, p


    def stack_triu(self, df, name=None):
        """ Stack the upper triangular entries of the dataframe above the diagonal
        .. note:: Useful for symmetric dataframes like correlations or distances.

        Parameters
        ----------
        df : pandas DataFrame
            Dataframe to stack. 
            .. note:: upper triangular entries are taken from `df` as provided, with no check that the rows and columns are symmetric.
        name : str
            Optional name of pandas Series output `df_stacked`.

        Returns
        -------
        df_stacked : pandas Series
            The stacked upper triangular entries above the diagonal of the dataframe.
        """
        # df_stacked = df.stack()[np.triu(np.ones(df.shape).astype(bool), 1).reshape(df.size)]
        # df_stacked.name = name
        df_stacked = stack_triu_(df, name=name)
        return df_stacked

    def stack_triu_where(self, df, condition, name=None):
        """ Stack the upper triangular entries of the dataframe above the diagonal where the condition is True
        .. note:: Useful for symmetric dataframes like correlations or distances.

        Parameters
        ----------
        df : pandas DataFrame
            Dataframe to stack. 
            .. note:: upper triangular entries are taken from `df` as provided, with no check that the rows and columns are symmetric.
        condition : pandas DataFrame
            Boolean dataframe of the same size and order of rows and columns as `df` indicating values, where `True`, to include
            in the stacked dataframe.
        name : str
            Optional name of pandas Series output `df_stacked`.

        Returns
        -------
        df_stacked : pandas Series
            The stacked upper triangular entries above the diagonal of the dataframe, where `condition` is `True`.        
        """        
        # df_stacked = df.stack()[np.triu(condition.astype(bool), 1).reshape(df.size)]
        # df_stacked.name = name
        df_stacked = stack_triu_where_(df, condition, name=name)
        return df_stacked


    def dispersion(self, data, axis=0):
        """ Data dispersion computed as the absolute value of the variance-to-mean ratio where the variance and mean is computed on
        the values over the requested axis.

        Parameters
        ----------
        data : pandas DataFrame
            Data used to compute dispersion.
        axis : {0, 1}
            Axis on which the variance and mean is applied on computed.

            Options
            -------
            0 : for each column, apply function to the values over the index
            1 : for each index, apply function to the values over the columns

        Returns
        -------
        vmr : pandas Series
            Variance-to-mean ratio (vmr) quantifying the disperion.
        """
        # vmr = np.abs(data.var(axis=axis) / data.mean(axis=axis))
        vmr = dispersion_(data, axis=axis)
        return vmr


    def value_counter(self, values):
        """ returns dictionary with the number of times each value appears.

        Parameters
        ----------
        values : iterable
            List of values.

        Returns
        -------
        counter : defaultdict(int)
            Dictionary of the form {value : count} with the number of times each value appears in the iterable.
        """
        counter = defaultdict(int)
        for k in values:
            counter[k] += 1
        return counter

        
    def weighted_sample_network(self, sample, weight='weight', data=None, **kwargs):
        """ return weighted graph by sample feature

        Parameters
        ----------
        sample : str
            Name of sample to use.
        weight : str
            Name of node attribute that the sample feature is saved to in the returned graph, default = 'weight'
        data : {None, str, pandas DataFrame}
            Specify which data the sample weights should be taken from.

            data options
            ------------
            None : The original data is used.
            str : `data` is is expected to be a key in `self.meta` and the sample weights are taken from the
                data in the correesponding dict-value.
            pandas DataFrame : The dataframe is used to select the sample weights, where the sample is expected to be
                one of the columns and the rows are expected to be labelled by the node indices, :math:`0, 1, ..., n-1` where
                :math:`n` is the number of nodes in the graph.
        kwargs : dict, optional
            Specify arguments passed to computing edge weights.

        Returns
        -------
        G : networkx graph
            The sample-specific node-weighted graph.
        """
        G = self.G.copy()
        if data is None:
            s_data = self.data[sample].to_dict()
        elif isinstance(data, str):
            s_data = self.meta[data][sample].to_dict()
        else:  # pandas DataFrame
            s_data = data[sample].to_dict()
        nx.set_node_attributes(G, s_data, name=weight)

        compute_edge_weights(G, n_weight=weight, e_weight=weight, **kwargs)
        return G

    def compute_graph_distances(self, G=None, weight='weight'):
        """ compute graph distances

        Parameters
        ----------
        G : nx.Graph
            The graph with nodes assumed to be labeled consecutively from `0, 1, ..., n-1` where `n` is the number of nodes.
        weight : str, optional
            Edge attribute of weights used for computing the weighted hop distance.
            If `None`, compute the unweighted distance. That is, rather than minimizing the sum of weights over
            all paths between every two nodes, minimize the number of edges.

        Returns
        -------
        dist : numpy ndarray
            An n x n matrix of node-pairwise graph distances between the n nodes.
        """
        G = self.G if G is None else G
        return compute_graph_distances(G, weight=weight)



    def neighborhood(self, node, include_self=False):
        """ return neighborhood of the node.

        Parameters
        ----------
        node : int
            Node of interest.
        include_self : bool
            If `True`, include `node` in its neighborhood.

        Returns
        -------
        neighborhood : list
            List of nodes in the neighborhood of `node`.
        """
        neighborhood = list(self.G.neighbors(node))
        if include_self:
            neighborhood = neighborhood + [node]
        return neighborhood
        

        
    def neighborhood_profiles(self, node, include_self=False, data=None):
        """ return profiles on the neighborhood of the node.

        Parameters
        ----------
        node : int
            Node in the graph to compute the neighborhood on.
        include_self : bool
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution over the neighborhood.
        data : {None, str, pandas DataFrame}
            Specify which data the profiles should be taken from.

            data options
            ------------
            None : The original data is used.
            str : `data` is is expected to be a key in `self.meta` and the profiles are taken from the
                data in the correesponding dict-value.
            pandas DataFrame : The dataframe is used to select the profiles, where columns are samples and the rows are
                expected to be labelled by the node indices, :math:`0, 1, ..., n-1` where
                :math:`n` is the number of nodes in the graph.

        Returns
        -------
        sub_profiles : pandas DataFrame
            The profiles over the node neighborhood.
        """
        # neighborhood = list(self.G.neighbors(node))
        # if include_self:
        #     neighborhood = neighborhood + [node]

        if data is None:
            s_data = self.data
        elif isinstance(data, str):
            s_data = self.meta[data]
        else:  # pandas DataFrame
            s_data = data
            
        neighborhood = self.neighborhood(node, include_self=include_self)
        sub_profiles = s_data.loc[neighborhood].copy()
        return sub_profiles


    def pairwise_sample_neighborhood_wass_distance(self, node, include_self=False, data=None, graph_distances=None,
                                                   proc=mp.cpu_count(), chunksize=None,
                                                   measure_cutoff=1e-6, solvr=None):
        """ Compute sample-pairwise Wasserstein distances between the profiles over node neighborhood.

        Parameters
        ----------
        node : int
            Node in the graph to compute the neighborhood on.
        include_self : bool
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution over the neighborhood.
        data : {None, str, pandas DataFrame}
            Specify which data the profiles should be taken from.

            data options
            ------------
            None : The original data is used.
            str : `data` is is expected to be a key in `self.meta` and the profiles are taken from the
                data in the correesponding dict-value.
            pandas DataFrame : The dataframe is used to select the profiles, where columns are samples and the rows are
                expected to be labelled by the node indices, :math:`0, 1, ..., n-1` where
                :math:`n` is the number of nodes in the graph.
        graph_distances : numpy ndarray
            An n x n matrix of node-pairwise graph distances between the n nodes (ordered from 0, 1, ..., n-1).
            If `None`, use hop distance.
        measure_cutoff : float
            Threshold for treating values in profiles as zero, default = 1e-6.
        proc : int
            Number of processor used for multiprocessing. (Default value = cpu_count()). 
        chunksize : int
            Chunksize to allocate for multiprocessing.
        solvr : str
            Solver to pass to POT library for computing Wasserstein distance.


        Returns
        -------
        wd : pandas Series
            Wasserstein distances between pairwise samples
        """

        profiles = self.neighborhood_profiles(node, include_self=include_self, data=data)
        if graph_distances is None:
            logger.msg("Computing graph hop distances.")
            graph_distances = self.compute_graph_distances(weight=None)

        graph_distances = graph_distances[np.ix_(profiles.index.tolist(), profiles.index.tolist())]

        wd = pairwise_sample_wass_distances(profiles, graph_distances, proc=proc, chunksize=chunksize,
                                                 measure_cutoff=measure_cutoff, solvr=solvr, flag=f"node {node}")

        return wd

    def pairwise_sample_neighborhood_euc_distance(self, node, include_self=False, metric='euclidean',
                                                  data=None, normalize=False, **kwargs):
        """ Compute sample-pairwise Euclidean distances between the profiles over node neighborhood.

        Parameters
        ----------
        profiles : pandas DataFrame
            Profiles that are normalized and treated as probability distributions for computing Wasserstein distance,
            where rows are features and columns are samples.
        node : int
            Node in the graph to compute the neighborhood on.
        include_self : bool
            If `True`, include node in neighborhood.
            If `False`, node is not included in neighborhood.
        data : {None, str, pandas DataFrame}
            Specify which data the profiles should be taken from.
        normalize : bool
            If `True`, normalize neighborhood profiles to sum to 1.
        **kwargs : dict
            Extra arguments to metric, passed to scipy.spatial.distance.cdist.

            data options
            ------------
            None : The original data is used.
            str : `data` is is expected to be a key in `self.meta` and the profiles are taken from the
                data in the correesponding dict-value.
            pandas DataFrame : The dataframe is used to select the profiles, where columns are samples and the rows are
                expected to be labelled by the node indices, :math:`0, 1, ..., n-1` where
                :math:`n` is the number of nodes in the graph.

        Returns
        -------
        ed : pandas Series
            Euclidean distances between pairwise samples        
        """

        profiles = self.neighborhood_profiles(node, data=data, include_self=include_self)
        if normalize:
            profiles = profiles / profiles.sum(axis=0)        
        return pairwise_sample_euc_distances(profiles, metric=metric, **kwargs)


    def anisotropic_laplacian_matrix(self, sample=None, use_spectral_gap=False, data=None):
        """ returns transpose of the  anisotropic random-walk  Laplacian matrix

        Parameters
        ----------
        sample : {None, str}
            If provided, use sample-weighted graph to construct the Laplacian. Otherwise, if `None`,
            The Laplacian is constructed from the unweighted graph, treated with uniform weights equal to 1.
        use_spectral_gap : bool
            Option to use spectral gap.
        data : {None, str, pandas DataFrame}
            Specify which data the sample profile should be taken from.
            .. note: This is ignored if `sample` is `None`.

            data options
            ------------
            None : The original data is used.
            str : `data` is is expected to be a key in `self.meta` and the sample profile is taken from the
                data in the correesponding dict-value.
            pandas DataFrame : The dataframe is used to select the sample profile, where columns are samples and the rows are
                expected to be labelled by the node indices, :math:`0, 1, ..., n-1` where
                :math:`n` is the number of nodes in the graph.

        Returns
        -------
        Lrw :
            Transpose of the random-walk graph Laplacian matrix.
        """

        G = self.G.copy()
        if sample is None:
            nx.set_node_weights(G, 1., name='weight')
        else:
            G = self.weighted_sample_network(sample, weight='weight', data=data)
        laplacian = construct_anisotropic_laplacian_matrix(G, 'weight', use_spectral_gap=use_spectral_gap)
        return laplacian
    
    
    def diffuse_profile(self, sample, times=None, t_min=-1.5, t_max=2.5, n_t=10,
                        log_time=True, # graph_distances=None,
                        laplacian=None, do_save=True):
        """ diffuse profile from original data

        # get laplacian for sample or hop distance for all

        Parameters
        ----------
        sample : str
            Sample profile to use.
        times : {None, array}
            Array of times to evaluate the diffusion simulation.
            .. note:: If given, `t_min`, `t_max` and `n_t` are ignored.
        t_min : float
            First time point to evaluate the diffusion simulation.
            .. note:: `t_min` is ignored if `times` is not `None`.
        t_max : float
            Last time point to evaluate the diffusion simulation.
            .. note:: `t_max` must be greater than `t_min`, i.e, `t_max` > `t_min`.
            .. note:: 't_max` is ignored if `times` is not `None`.
        n_t : int
            Number of time points to generate.
            .. note:: `n_t` is ignored if `times` is not `None`.
        log_time : bool
            If `True`, return `n_t` numbers spaced evenly on a log scale, where the time
            sequence starts at ``10 ** t_min``, ends with ``10 ** t_max``, and the
            sequence of times if of the form ``10 ** t`` where ``t`` is the `n_t`
            evenly spaced points between (and including) `t_min` and `t_max`.
            For example, `_get_times(t_min=1, t_max=3, n_t=3, log_time=True) = array([10 ** 1, 10 ** 2, 10 ** 3])
                                                                             = array([10., 100., 1000.])`.
            If `False`, return `n_t` numbers evenly spaced on a linear scale, where the sequence
            starts at `t_min` and ends with `t_max`.
            For example, `_get_times(t_min=1, t_max=3, n_t=3, log_time=False) = array([1. ,2., 3.])`.
        graph_distances : numpy ndarray
            An `n` x `n` matrix of node-pairwise graph distances between the `n` nodes ordered by the rows in `object.profiles`.
        laplacian : numpy ndarray
            The `n` x `n` transpose of the graph Laplacian matrix where the `n` rows and columns  ordered by the rows in `object.profiles`.
        filename : str
            If not `None`,
        do_save : bool
            If `True`, save diffused profile to `self.outdir` / 'diffused_profiles' / 'diffused_profile_{`sample`}.csv'

        Returns
        -------
        profiles : pandas DataFrame
            Diffused profiles where each row is a time and each column is a feature name
        """

        if do_save:
            if not (self.outdir / 'diffused_profiles').is_dir():
                (self.outdir / 'diffused_profiles').mkdir()

        times = get_times(times=times, t_min=t_min, t_max=t_max, n_t=n_t, log_time=log_time)
        times_with_zero = np.insert(times, 0, 0.0)
        profile = self.data[sample].loc[list(range(len(self.G)))].values
        # profile = profile[:, 0]
        # logger.msg(f"Profile shape = {profile.shape}.")
        profiles = np.inf * np.ones([len(times_with_zero), len(self.G)]) 
        profiles[0] = profile
        for time_index in tqdm(range(len(times)), desc="Diffusing profile", colour="green", leave=False):
            logger.debug("---------------------------------")
            logger.debug("Step %s / %s", str(time_index), str(len(times)))
            logger.debug("Computing diffusion time 10^{:.1f}".format(np.log10(times[time_index])))

            logger.debug("Computing measures")
            profile = heat_kernel(profile, laplacian, times_with_zero[time_index + 1] - times_with_zero[time_index])
            profiles[time_index+1] = profile

        profiles = pd.DataFrame(data=profiles, index=times_with_zero,
                                # columns=list(range(len(self.G))),
                                columns=[self.G.nodes[k]['name'] for k in range(len(self.G))])
                                
        if do_save:
            profiles.to_csv(self.outdir / 'diffused_profiles' / 'diffused_profile_{}.csv'.format(sample),
                            header=True, index=True)
        return profiles

    def diffuse_multiple_profiles(self, samples=None, times=None, t_min=-1.5, t_max=2.5, n_t=10,
                                  log_time=True, # graph_distances=None,
                                  laplacian=None, use_spectral_gap=False, do_plot=False, **plot_kwargs):
        """ diffuse multiple sample profiles from original data

        # get laplacian for sample or hop distance for all

        Parameters
        ----------
        samples : {`None`, list}
            Samples to iterate over. If `None`, use all samples in data.
        times : {None, array}
            Array of times to evaluate the diffusion simulation.
            .. note:: If given, `t_min`, `t_max` and `n_t` are ignored.
        t_min : float
            First time point to evaluate the diffusion simulation.
            .. note:: `t_min` is ignored if `times` is not `None`.
        t_max : float
            Last time point to evaluate the diffusion simulation.
            .. note:: `t_max` must be greater than `t_min`, i.e, `t_max` > `t_min`.
            .. note:: 't_max` is ignored if `times` is not `None`.
        n_t : int
            Number of time points to generate.
            .. note:: `n_t` is ignored if `times` is not `None`.
        log_time : bool
            If `True`, return `n_t` numbers spaced evenly on a log scale, where the time
            sequence starts at ``10 ** t_min``, ends with ``10 ** t_max``, and the
            sequence of times if of the form ``10 ** t`` where ``t`` is the `n_t`
            evenly spaced points between (and including) `t_min` and `t_max`.
            For example, `_get_times(t_min=1, t_max=3, n_t=3, log_time=True) = array([10 ** 1, 10 ** 2, 10 ** 3])
                                                                             = array([10., 100., 1000.])`.
            If `False`, return `n_t` numbers evenly spaced on a linear scale, where the sequence
            starts at `t_min` and ends with `t_max`.
            For example, `_get_times(t_min=1, t_max=3, n_t=3, log_time=False) = array([1. ,2., 3.])`.
        graph_distances : numpy ndarray
            An `n` x `n` matrix of node-pairwise graph distances between the `n` nodes ordered by the rows in `object.profiles`.
        laplacian : numpy ndarray
            The `n` x `n` transpose of the graph Laplacian matrix where the `n` rows and columns  ordered by the rows in `object.profiles`.
            If `None`, the sample-specific Laplacian is used.
        use_spectral_gap : bool
            Option to use spectral gap.
            .. note: This is ignored if `laplacian` is provided and not `None`.
        filename : str
            If not `None`,
        do_plot : bool
            If `True`, plot diffused profiles for each sample.
        **plot_kwargs : dict
            Key-word arguments passed to `plot_profiles` (should not include `title`).
        

        Side-effects
        ------------
        - Saves pandas DataFrame of the diffused profiles, where each row is a time and each column is a feature name,
            for each sample to the file name `self.outdir` / 'diffused_profiles' / 'diffused_profile_{`sample`}.csv'
        - If `do_plot` is `True`, plots the diffused profile for each sample.
        """

        if samples is None:
            samples = self.data.columns.tolist()

        for sample in tqdm(samples, desc='Diffusing sample profiles'):
            if laplacian is None:
                s_laplacian = self.anisotropic_laplacian_matrix(sample=sample, use_spectral_gap=use_spectral_gap)
            else:
                s_laplacian = laplacian

            profiles = self.diffuse_profile(sample, times=times, t_min=t_min, t_max=t_max, n_t=n_t,
                                            log_time=log_time, laplacian=s_laplacian, do_save=True)

            if do_plot:
                ax = self.plot_profiles(profiles, title=sample+": diffused profile", **plot_kwargs)


    def load_diffused_profile(self, sample):
        """ loads sample diffused profile

        Parameters
        ----------
        sample : str
            Sample profile to load

        Returns
        -------
        diffused_profiles : pandas DataFrame
            Diffused profiles where each row is a time and each column is a feature name
        """
        diffused_profiles = pd.read_csv(self.outdir / 'diffused_profiles' / 'diffused_profile_{}.csv'.format(sample),
                                        header=0, index_col=0)

        return diffused_profiles


    def load_diffused_timepoint_profile(self, time, samples=None):
        """ loads sample diffused profile

        Parameters
        ----------
        time : float
            Timepoint in diffusion simulation to select.
        samples : {`None`, list}
            List of samples to iterate over. If `None`, all samples in `self.data` are used.

        Returns
        -------
        diffused_profiles : pandas DataFrame
            Diffused profiles at `time`, for all `samples`, where rows are nodes and columns are samples.
        """

        if samples is None:
            samples = self.data.columns.tolist()

        diffused_profiles = []
        for sample in tqdm(samples, desc='Loading diffused profiles at specified time'):
            profile = self.load_diffused_profile(sample)
            profile = profile.loc[time]
            profile.name = sample
            diffused_profiles.append(profile)

        diffused_profiles = pd.concat(diffused_profiles, axis=1)
        return diffused_profiles            
        

    def plot_profiles(self, profiles, ylog=False, ax=None, figsize=(5.3, 4), title="", lw=1.3, marker_size=2, **plot_kwargs):
        """ plot profiles, with rows as times and columns as features """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.get_figure()

        times = profiles.index.tolist()[:]
        if times[0] < 1e-6:
            times[0] = 1e-6
        for profile in profiles.values.T:
            if all(profile > 0):
                color = "olive"  # "C0"
            else:
                color = "tan"  # "C1"
            ax.plot(np.log10(times), profile, '-o', c=color, lw=lw, ms=marker_size, **plot_kwargs)

        if ylog:
            ax.set_xscale("symlog")
        ax.axhline(0, ls="--", c="k")
        ax.axis([np.log10(times[0]), np.log10(times[-1]), np.min(profiles.values), np.max(profiles.values)])
        ax.set_xlabel(r"$log_{10}(t)$")
        ax.set_ylabel("diffused value")
        ax.set_title(title)

        return ax


    def multiple_pairwise_sample_neighborhood_wass_distance(self, nodes=None, include_self=False, data=None,
                                                            graph_distances=None, desc='Computing pairwise 1-hop distances',
                                                            profiles_desc='t0',
                                                            proc=mp.cpu_count(), chunksize=None,
                                                            measure_cutoff=1e-6, solvr=None):
        """ Compute sample-pairwise Wasserstein distances between the profiles over node neighborhood for all nodes.

        Parameters
        ----------
        nodes : {None, list(int)}
            List of nodes to compute neighborhood distances on. If `None`, all genes with at least 2 neighbors is used.
        include_self : bool
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution over the neighborhood.
        data : {None, str, pandas DataFrame}
            Specify which data the profiles should be taken from.

            data options
            ------------
            None : The original data is used.
            str : `data` is is expected to be a key in `self.meta` and the profiles are taken from the
                data in the correesponding dict-value.
            pandas DataFrame : The dataframe is used to select the profiles, where columns are samples and the rows are
                expected to be labelled by the node indices, :math:`0, 1, ..., n-1` where
                :math:`n` is the number of nodes in the graph.
        graph_distances : numpy ndarray
            An n x n matrix of node-pairwise graph distances between the n nodes (ordered from 0, 1, ..., n-1).
            If `None`, use hop distance.
        desc : str
            Description for progress bar.
        profiles_desc : str, default = "t0"
            Description of profiles used in name of file to store results.
        measure_cutoff : float
            Threshold for treating values in profiles as zero, default = 1e-6.
        proc : int
            Number of processor used for multiprocessing. (Default value = cpu_count()). 
        chunksize : int
            Chunksize to allocate for multiprocessing.
        solvr : str
            Solver to pass to POT library for computing Wasserstein distance.

        Returns
        -------
        wds : pandas DataFrame
            Wasserstein distances between pairwise samples where rows are sample-pairs and columns are node names.

        .. note:: If `object.outdir` is not `None`, Wasserstein distances are saved to file every 10 iterations.
            Before starting the computation, check if the file exists. If so, load and remove already computed
            nodes from the iteration. Wasserstein distances are computed for the remaining nodes, combined with
            the previously computed and saved results before saving and returning the combined results.

        To do: specify if nodes in input are ids or node names and check that loaded data has correct type int or str for nodes
        """

        if nodes is None:
            nodes = [k for k in self.G if len(list(self.G.neighbors(k)))>1]
        else:
            nodes = [self.name2node[k] for k in nodes if self.G.degree(self.name2node[k]) > 1]

        if self.outdir is None:
            fname = None
            wds_prior = None
        else:
            # fname = self.outdir / f"wass_dist_sample_pairwise_1hop_nbhd_profiles_{profiles_desc}_with{'' if include_self else 'out'}_self.csv"
            fname = f"wass_dist_sample_pairwise_1hop_nbhd_profiles_{profiles_desc}_with{'' if include_self else 'out'}_self.csv"
            self.filenames.append(fname)
            if (self.outdir / fname).is_file():
                
                wds_prior = pd.read_csv(self.outdir / fname, header=0, index_col=(0, 1))
                logger.msg(f"Loaded saved sample pairwise 1-hop neighborhood Wasserstein distances from {profiles_desc} profiles of size {wds_prior.shape}.")
                n_orig = len(nodes)
                nodes = [k for k in nodes if self.G.nodes[k]['name'] not in wds_prior.columns]
                n_update = len(nodes)
                if n_update < n_orig:
                    if n_update == 0:
                        # logger.msg(f"Loading sample pairwise 1-hop neighborhood Wasserstein distances from {profiles_desc} profiles.")
                        return wds_prior
                    
                    logger.msg(f"Computing sample pairwise 1-hop neighborhood Wasserstein distances from {profiles_desc} profiles on {n_update}/{n_orig} nodes.")
            else:
                wds_prior = None

        wds = []

        # logger.msg(f"Computing Wasserstein distances on {len(nodes)} neighborhoods.")
        for ix, node in tqdm(enumerate(nodes), desc=desc, colour='yellow', total=len(nodes)):
            tmp = self.pairwise_sample_neighborhood_wass_distance(node, include_self=include_self, data=data,
                                                                  graph_distances=graph_distances,
                                                                  proc=proc, chunksize=chunksize,
                                                                  measure_cutoff=measure_cutoff, solvr=solvr)
            tmp.name = node
            wds.append(tmp)

            if (ix % 10 == 0) and (self.outdir is not None):
                wds_tmp = pd.concat(wds, axis=1)
                wds_tmp = wds_tmp.rename(columns=nx.get_node_attributes(self.G, 'name'))
                if wds_prior is not None:
                    wds_prior = pd.concat([wds_prior, wds_tmp], axis=1)
                else:
                    wds_prior = wds_tmp
                wds_prior.to_csv(str(self.outdir / fname), header=True, index=True)
                wds = []

        if wds:        
            wds = pd.concat(wds, axis=1)
            wds = wds.rename(columns=nx.get_node_attributes(self.G, 'name'))
        else:
            wds = None
        if wds_prior is not None:
            wds = pd.concat([wds_prior, wds], axis=1)
        if self.outdir is not None:
            wds.to_csv(str(self.outdir / fname), header=True, index=True)
            logger.msg(f"Sample pairwise 1-hop neighborhood Wasserstein distances on {profiles_desc} saved to {str(fname)}.")
        return wds


    def multiple_pairwise_sample_neighborhood_euc_distance(self, nodes=None, include_self=False, data=None,
                                                           desc='Computing pairwise 1-hop distances', profiles_desc='t0',
                                                           metric='euclidean', normalize=False, **kwargs):
        """ Compute sample-pairwise Euclidean distances between the profiles over node neighborhood for all nodes.

        Parameters
        ----------
        nodes : {None, list(int)}
            List of nodes to compute neighborhood distances on. If `None`, all genes with at least 2 neighbors is used.
        include_self : bool
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution over the neighborhood.
        data : {None, str, pandas DataFrame}
            Specify which data the profiles should be taken from.

            data options
            ------------
            None : The original data is used.
            str : `data` is is expected to be a key in `self.meta` and the profiles are taken from the
                data in the correesponding dict-value.
            pandas DataFrame : The dataframe is used to select the profiles, where columns are samples and the rows are
                expected to be labelled by the node indices, :math:`0, 1, ..., n-1` where
                :math:`n` is the number of nodes in the graph.
        desc : str
            Description for progress bar.
        profiles_desc : str, default = "t0"
            Description of profiles used in name of file to store results.
        normalize : bool
            If `True`, normalize neighborhood profiles to sum to 1.
        **kwargs : dict
            Extra arguments to metric, passed to scipy.spatial.distance.cdist.

        Returns
        -------
        eds : pandas DataFrame
            Euclidean distances between pairwise samples where rows are sample-pairs and columns are node names.

        .. note:: If `object.outdir` is not `None`, Euclidean distances are saved to file.
            Before starting the computation, check if the file exists. If so, load and remove already computed
            nodes from the iteration. Wasserstein distances are computed for the remaining nodes, combined with
            the previously computed and saved results before saving and returning the combined results.
        """

        if nodes is None:
            nodes = [k for k in self.G if len(list(self.G.neighbors(k)))>1]
        else:
            nodes = [self.name2node[k] for k in nodes if self.G.degree(self.name2node[k]) > 1]

        if self.outdir is None:
            fname = None
            eds_prior = None
        else:
            if normalize:
                fname = f"euc_dist_sample_pairwise_1hop_nbhd_profiles_{profiles_desc}_with{'' if include_self else 'out'}_self_normalized.csv"
            else:
                fname = f"euc_dist_sample_pairwise_1hop_nbhd_profiles_{profiles_desc}_with{'' if include_self else 'out'}_self.csv"
            self.filenames.append(fname)
            if (self.outdir / fname).is_file():                
                eds_prior = pd.read_csv(self.outdir / fname, header=0, index_col=(0, 1))
                logger.msg(f"Loaded saved sample pairwise 1-hop neighborhood Euclidean distances from {profiles_desc} profiles of size {eds_prior.shape}.")
                n_orig = len(nodes)
                nodes = [k for k in nodes if self.G.nodes[k]['name'] not in eds_prior.columns]
                n_update = len(nodes)
                if n_update < n_orig:
                    if n_update == 0:
                        return eds_prior
                    logger.msg(f"Computing sample pairwise 1-hop neighborhood Euclidean distances from {profiles_desc} profiles on {n_update}/{n_orig} nodes.")
            else:
                eds_prior = None

        eds = []

        for node in tqdm(nodes, desc=desc, colour='yellow', total=len(nodes)):
            tmp = self.pairwise_sample_neighborhood_euc_distance(node, include_self=include_self,
                                                                 metric=metric, data=data, normalize=normalize, **kwargs)
            tmp.name = node
            eds.append(tmp)
        
        if eds:
            eds = pd.concat(eds, axis=1)
            eds = eds.rename(columns=nx.get_node_attributes(self.G, 'name'))
        else:
            eds = None

        if eds_prior is not None:
            eds = pd.concat([eds_prior, eds], axis=1)
        if self.outdir is not None:
            eds.to_csv(self.outdir / fname, header=True, index=True)
            logger.msg(f"Sample pairwise 1-hop neighborhood Euclidean distances on {profiles_desc} saved to {str(fname)}.")
        return eds

            


        

        


        

        

    



    

    
    


    

if __name__ == '__main__':
    G = nx.karate_club_graph()
    data = pd.DataFrame(data=np.abs(np.random.randn(len(G), 9)) + 0.001,
                        # data=np.array([[1.]*len(G), [2.]*len(G), [3.]*len(G),
                        #                [4.]*len(G), [5.]*len(G), [6.]*len(G),
                        #                [7.]*len(G), [8.]*len(G), [9.]*len(G)]).transpose(), index=list(G),
                        columns=[f"x{k}" for k in range(9)])

    logger.msg(f"Data is size {data.shape[0]} x {data.shape[1]}.")
    inet = InfoNet(G, data, './')
    g = inet.weighted_sample_network('x2', weight='weight')
    print(nx.get_node_attributes(g, 'weight'))


    dhop = inet.compute_graph_distances(weight=None)
    print(dhop.shape)
    # print(inet.neighborhood_profiles(16, include_self=False))

    # logger.msg(f"Wass = {wass_distance(('x1', 'x2'), data, dhop, measure_cutoff=1e-6)}.")

    # wds16 = inet.pairwise_sample_neighborhood_wass_distance(16, include_self=False, graph_distances=dhop) # , graph_distances=dhop, proc=mp.cpu_count(), chunksize=None)
    

    # print(wds16)

    wds, eds = [], []

    
    for gn in [k for k in G if len(list(G.neighbors(k)))>1]:
        tmp = inet.pairwise_sample_neighborhood_wass_distance(gn, include_self=False, graph_distances=dhop, measure_cutoff=1e-6)
        tmp.name = gn
        wds.append(tmp)

        tmp = inet.pairwise_sample_neighborhood_euc_distance(gn, include_self=False, metric='euclidean')
        tmp.name=gn
        eds.append(tmp)

    wds = pd.concat(wds, axis=1)
    eds = pd.concat(eds, axis=1)
    print("ranges: ", wds.max().max(), wds.min().min(), eds.max().max(), eds.min().min())
    print(wds.head())


    wds_a = inet.multiple_pairwise_sample_neighborhood_wass_distance(data=None,
                                                                     graph_distances=dhop, desc='Computing pairwise 1-hop distances',
                                                                     profiles_desc='t0',
                                                                     proc=mp.cpu_count(), chunksize=None,
                                                                     measure_cutoff=1e-6, solvr=None)
    print(f"And automated wds: ")
    print("ranges: ", wds_a.max().max(), wds_a.min().min())
    print(wds_a.head())
    
    # print(wds.corr(method='spearman'))

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.clustermap(wds.corr(method='spearman'), method='ward', cmap='RdBu', center=0)
    # plt.gcf().suptitle('Spearman corr wds');
    
    # sns.clustermap(eds.corr(method='spearman'), method='ward', cmap='RdBu', center=0)
    # plt.gcf().suptitle('Spearman corr eds');

    # sns.clustermap(1. - (wds / eds), method='ward', cmap='RdBu', center=0)    
    # plt.gcf().suptitle('Sample curvatures');


    # controlled distance ranges
    # wds2 = wds / wds.max().max()
    # eds2 = eds / eds.max().max()
    # sns.clustermap(1. - (wds2 / eds2), method='ward', cmap='RdBu', center=0)    
    # plt.gcf().suptitle('Sample scaled curvatures');

    plt.show()


    # compute feature distance
    import seaborn as sns
    import netflow.pseudotime as nfp
    import netflow.utils as nfu
    print("Computing feature distances:")
    wf = pd.Series(data=nfp.norm_features(wds, method='L2'),
                   index=wds.index.copy())
    print(wf.shape, wf, sep='\n')

    A_wf = nfu.unstack_triu_(wf, index=data.columns.tolist())
    print(A_wf)

    sns.heatmap(A_wf)
    plt.show()
