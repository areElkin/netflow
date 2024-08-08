"""
Classes to compute distances.
"""

from pathlib import Path

from collections import defaultdict
from functools import partial
from itertools import combinations
import matplotlib.pyplot as plt
import multiprocessing as mp
import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial as ss
import scipy.stats as sc_stats
from tqdm import tqdm

# from .._logging import logger
from .._logging import _gen_logger, set_verbose
from ..checks import *
from .metrics import pairwise_observation_euc_distances, pairwise_observation_wass_distances

import netflow.utils as utl
# from importlib import reload
# reload(utl)
# kendall_tau_ = utl.kendall_tau_
# clustermap = utl.clustermap

# from ..utils import compute_graph_distances, heat_kernel, compute_edge_weights, get_times, \
#     construct_anisotropic_laplacian_matrix, clustermap, spearmanr_, kendall_tau_, stack_triu_, \
#     stack_triu_where_, dispersion_

# import netflow.utils as utl
# from importlib import reload
# reload(utl)
# # heat_kernel = utl.heat_kernel
# construct_anisotropic_laplacian_matrix = utl.construct_anisotropic_laplacian_matrix

logger = _gen_logger(__name__)

class InfoNet:
    """ A class to compute information flow on a network and correlation between network modules

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object that stores the data of size (n_features, n_observations).
    graph_key : 'str'
        The key to the graph in the graph keeper that should be used.
        (Does not have to include all features in the data)
    layer : `str`
        The key to the data in the data keeper that should be used.
    """
    
    def __init__(self, keeper, graph_key, layer,
                 verbose=None):

        # logger.msg("INFONET HAS BEEN ENTERED")

        if verbose is not None:
            set_verbose(logger, verbose)

        self.data = keeper.data[layer]
        check_matrix_no_nan(self.data.data)        
        logger.debug(f"Loaded data: {self.data.data.shape}")

        if graph_key is not None:
            G = keeper.graphs[graph_key].copy()  # keeper.misc[graph_key].copy()
            check_connected_graph(G)
            check_graph_no_self_loops(G)
            logger.debug(f"Loaded graph: {G}")

            if not nx.get_node_attributes(G, name='name'):
                # G = nx.convert_node_labels_to_integers(G, label_attribute='name')
                # Note: need nodes to line up with order in data
                nx.set_node_attributes(G, {k: k for k in G}, name='name')
                # nx.relabel_nodes(G, {k: ix for ix, k in enumerate(self.data.feature_labels)}, copy=False)
                nx.relabel_nodes(G, dict(zip(self.data.feature_labels, range(self.data.num_features))), copy=False)

            self.G = G
            self.name2node = {v: k for k, v in nx.get_node_attributes(self.G, "name").items()}

            # self.data = data.rename(index=self.name2node)
            # self.features = data.index.tolist()[:]
            # self.observations = data.columns.tolist()[:]
            self.features = [self.name2node[i] for i in self.data.feature_labels if i in self.name2node] # HERE!!! 
            # assert self.features == list(range(len(self.G))), "Graph features are not ordered according to data features."
            assert all([self.data.feature_labels[i] == self.G.nodes[i]['name'] for i in self.G]), \
                "Features in the graph are not ordered according to the data features."
            assert max(list(self.G)) == len(self.features) - 1, f"Features in the graph are not consecutively ordered."
            
        else:
            self.name2node = dict(zip(self.data.feature_labels, range(len(self.data.feature_labels))))
            self.features = self.data.feature_labels

        self.keeper = keeper
        self.observations = self.data.observation_labels

        self.meta = {}  # can be used to reference data to be used for subsequent analysis

        self.outdir = keeper.outdir # outdir if (outdir is None or isinstance(outdir, Path)) else Path(outdir)
        if self.outdir is not None:
            if self.outdir.is_dir():  # store list of names of saved files
                self.filenames = [k.name for k in self.outdir.iterdir()]
            else:
                logger.msg(f"Creating directory {self.outdir}.")
                self.outdir.mkdir()
                self.filenames = []


    def invariant_measures(self, label='IM'):
        """ Compute the invariant measure for each observation using data as node weights.

        Parameters
        ----------
        label : `str`
            Label of key used to store invariant measures in the data keeper.

        Returns
        -------
        Makes the invariant measures attribute of size (n_observations, n_observations)
        in ``keeper.data[label]`` available.
        """
        check_matrix_nonnegative(self.data.data)
        IM = utl.invariant_measure(self.data.data, G=self.G)
        self.keeper.add_data(IM, label)
        

        raise NotImplementedError("Not yet implemented")
        

            
    def spearmanr(self, data, **kwargs):
        """ Calculate a Spearman correlation coefficient with associated p-value using scipy.stats.spearmanr.

        Parameters
        ----------
        data : `numpy.ndarray`, (n_observations, n_features)
            2-D array containing multiple variables and observations.
        **kwargs : `dict`
            Optional key-word arguments passed to `scipy.stats.spearmanr`.

        Returns
        -------
        R : `pandas.DataFrame`
            Spearman correlation matrix. The correlation matrix is square with
            length equal to total number of variables (columns or rows).
        pvalue : `float`
            The p-value for a hypothesis test whose null hypotheisis
            is that two sets of data are uncorrelated. See documentation for scipy.stats.spearmanr
            for alternative hypotheses. ``p`` has the same
            shape as ``R``.
        """
        # stats = sc_stats.spearmanr(data, axis=0, **kwargs)
        # R = pd.DataFrame(data=stats.correlation, index=data.columns.copy(), columns=data.columns.copy())
        # p = pd.DataFrame(data=stats.pvalue, index=data.columns.copy(), columns=data.columns.copy())
        R, p = utl.spearmanr_(data, **kwargs)
        return R, p

    def kendall_tau(self, data, **kwargs):
        """ Calculate Kendall's tau correlation coefficient with associated p-value using scipy.stats.kendalltau.

        Parameters
        ----------
        data : `numpy.ndarray`, (n_observations, n_features)
            2-D array containing multiple variables and observations.
        **kwargs : `dict`
            Optional key-word arguments passed to `scipy.stats.kendalltau`.

        Returns
        -------
        R : `pandas.DataFrame`
            Spearman correlation matrix. The correlation matrix is square with
            length equal to total number of variables (columns or rows).
        pvalue : `float`
            The p-value for a hypothesis test whose null hypotheisis
            is that two sets of data are uncorrelated. See documentation for `scipy.stats.spearmanr`
            for alternative hypotheses. ``p`` has the same
            shape as ``R``.
        """
        R, p = utl.kendall_tau_(data, **kwargs)
        return R, p


    def stack_triu(self, df, name=None):
        """ Stack the upper triangular entries of the dataframe above the diagonal

        Note, this is useful for symmetric dataframes like correlations or distances.

        Parameters
        ----------
        df : pandas DataFrame
            Dataframe to stack. 
            Note, upper triangular entries are taken from ``df`` as provided, with no check that the rows and columns are symmetric.
        name : str
            Optional name of pandas Series output ``df_stacked``.

        Returns
        -------
        df_stacked : pandas Series
            The stacked upper triangular entries above the diagonal of the dataframe.
        """
        # df_stacked = df.stack()[np.triu(np.ones(df.shape).astype(bool), 1).reshape(df.size)]
        # df_stacked.name = name
        df_stacked = utl.stack_triu_(df, name=name)
        return df_stacked

    
    def stack_triu_where(self, df, condition, name=None):
        """ Stack the upper triangular entries of the dataframe above the diagonal where the condition is `True`
        Note, this is useful for symmetric dataframes like correlations or distances.

        Parameters
        ----------
        df : pandas DataFrame
            Dataframe to stack. 
            Note, upper triangular entries are taken from ``df`` as provided, with no check that the rows and columns are symmetric.
        condition : pandas DataFrame
            Boolean dataframe of the same size and order of rows and columns as `df` indicating values, where `True`, to include
            in the stacked dataframe.
        name : `str`, optional
            Name of pandas Series output ``df_stacked``.

        Returns
        -------
        df_stacked : pandas Series
            The stacked upper triangular entries above the diagonal of the dataframe, where ``condition`` is `True`.        
        """        
        # df_stacked = df.stack()[np.triu(condition.astype(bool), 1).reshape(df.size)]
        # df_stacked.name = name
        df_stacked = utl.stack_triu_where_(df, condition, name=name)
        return df_stacked


    def dispersion(self, data, axis=0):
        """ Data dispersion computed as the absolute value of the variance-to-mean ratio where the variance and mean is computed on
        the values over the requested axis.

        Parameters
        ----------
        data : `pandas.DataFrame`
            Data used to compute dispersion.
        axis : {0, 1}
            Axis on which the variance and mean is applied on computed.

            Options:

            - 0 : for each column, apply function to the values over the index
            - 1 : for each index, apply function to the values over the columns

        Returns
        -------
        vmr : `pandas.Series`
            Variance-to-mean ratio (vmr) quantifying the disperion.
        """
        raise NotImplementedError
    
        # vmr = np.abs(data.var(axis=axis) / data.mean(axis=axis))
        vmr = utl.dispersion_(data, axis=axis)
        return vmr


    def value_counter(self, values):
        """ returns dictionary with the number of times each value appears.

        Parameters
        ----------
        values : iterable
            List of values.

        Returns
        -------
        counter : `defaultdict` [value, `int`]
            Dictionary of the form {value : count} with the number of times each value appears in the iterable.
        """
        counter = defaultdict(int)
        for k in values:
            counter[k] += 1
        return counter

        
    def weighted_observation_network(self, observation, weight='weight', data=None, **kwargs):
        """ return weighted graph by observation feature.

        Parameters
        ----------
        observation : `str`
            Name of observation to use.
        weight : `str`
            Name of node attribute that the observation feature is saved to in the returned graph, default = 'weight'
        data : {`None`, `str`, `pandas.DataFrame`}
            Specify which data the observation weights should be taken from.

            data options:

            - `None` : The original data is used.
            - `str` : ``data`` is is expected to be a key in ``self.meta`` and the observation weights are taken from the
                data in the correesponding dict-value.
            - `pandas.DataFrame` : The dataframe is used to select the observation weights, where the observation is expected to be
                one of the columns and the rows are expected to be labelled by the node indices, :math:`0, 1, ..., n-1` where
                :math:`n` is the number of nodes in the graph.
        kwargs : `dict`, optional
            Specify arguments passed to computing edge weights.

        Returns
        -------
        G : networkx graph
            The observation-specific node-weighted graph.
        """
        raise NotImplementedError
    
        G = self.G.copy()
        if data is None:
            s_data = self.data[observation].to_dict()
        elif isinstance(data, str):
            s_data = self.meta[data][observation].to_dict()
        else:  # pandas DataFrame
            s_data = data[observation].to_dict()
        nx.set_node_attributes(G, s_data, name=weight)

        utl.compute_edge_weights(G, n_weight=weight, e_weight=weight, **kwargs)
        return G

    
    def compute_graph_distances(self, G=None, weight='weight'):
        """ compute graph distances

        Parameters
        ----------
        G : `networkx.Graph`
            The graph with nodes assumed to be labeled consecutively from :math:`0, 1, ..., n-1` where :math:`n` is the number of nodes.
        weight : `str`, optional
            Edge attribute of weights used for computing the weighted hop distance.
            If `None`, compute the unweighted distance. That is, rather than minimizing the sum of weights over
            all paths between every two nodes, minimize the number of edges.

        Returns
        -------
        dist : `numpy.ndarray`, (n, n)
            A matrix of node-pairwise graph distances between the :math:`n` nodes.
        """
        G = self.G if G is None else G
        dist = utl.compute_graph_distances(G, weight=weight)
        return dist


    def neighborhood(self, node, include_self=False):
        """ return neighborhood of the node.

        Parameters
        ----------
        node : `int`
            Node of interest.
        include_self : `bool`
            If `True`, include ``node`` in its neighborhood.

        Returns
        -------
        neighborhood : `list`
            List of nodes in the neighborhood of ``node``.
        """
        neighborhood = list(self.G.neighbors(node))
        if include_self:
            neighborhood = neighborhood + [node]
        return neighborhood
        
        
    def neighborhood_profiles(self, node, include_self=False):
        """ return profiles on the neighborhood of the node.

        Parameters
        ----------
        node : `int`
            Node in the graph to compute the neighborhood on.
        include_self : `bool`
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution over the neighborhood.

        Returns
        -------
        neighborhood : `list`
            List of nodes in the neighborhood of ``node``.
        sub_profiles : `pandas.DataFrame`
            The profiles over the node neighborhood.
        """
        # neighborhood = list(self.G.neighbors(node))
        # if include_self:
        #     neighborhood = neighborhood + [node]

        # if data is None:
        #     s_data = self.data
        # elif isinstance(data, str):
        #     s_data = self.meta[data]
        # else:  # pandas DataFrame
        #     s_data = data
        # s_data = self.data.data if data is None else data
            
        neighborhood = self.neighborhood(node, include_self=include_self)
        # sub_profiles = s_data.loc[neighborhood].copy()
        sub_profiles = self.data.data[neighborhood, :].copy()
        return neighborhood, sub_profiles


    def pairwise_observation_neighborhood_wass_distance(self, node, include_self=False,
                                                        graph_distances=None,
                                                        proc=mp.cpu_count(), chunksize=None,
                                                        measure_cutoff=1e-6, solvr=None):
        """ Compute observation-pairwise Wasserstein distances between the profiles over node neighborhood.

        Parameters
        ----------
        node : `int`
            Node in the graph to compute the neighborhood on.
        include_self : `bool`
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution over the neighborhood.
        graph_distances : `numpy.ndarray`, (n, n)
            A matrix of node-pairwise graph distances between the n nodes (ordered from :math:`0, 1, ..., n-1`).
            If `None`, use hop distance.
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
        wd : `pandas.Series`
            Wasserstein distances between pairwise observations
        """
        # check_matrix_nonnegative(self.data.data)
        nbhd, profiles = self.neighborhood_profiles(node, include_self=include_self)
        if graph_distances is None:
            logger.debug("Computing graph hop distances.")
            graph_distances = self.compute_graph_distances(weight=None)

        # graph_distances = graph_distances[np.ix_(profiles.index.tolist(), profiles.index.tolist())]
        graph_distances = graph_distances[np.ix_(nbhd, nbhd)]

        wd = pairwise_observation_wass_distances(pd.DataFrame(data=profiles, columns=self.observations), # (n_features, n_observations)
                                            graph_distances, proc=proc, chunksize=chunksize,
                                            measure_cutoff=measure_cutoff, solvr=solvr, flag=f"node {node}")

        return wd

    
    def pairwise_observation_neighborhood_euc_distance(self, node, include_self=False, metric='euclidean',
                                                  normalize=False, **kwargs):
        """ Compute observation-pairwise Euclidean distances between the profiles over node neighborhood.

        Parameters
        ----------
        profiles : `pandas.DataFrame` (n_features, n_observations)
            Profiles that Euclidean distance is computed between. 
        node : `int`
            Node in the graph to compute the neighborhood on.
        include_self : `bool`
            If `True`, include node in neighborhood.
            If `False`, node is not included in neighborhood.
        metric : `str`
            The metric used to compute the distance, passed to scipy.spatial.distance.cdist.
        normalize : `bool`
            If `True`, normalize neighborhood profiles to sum to 1.
        **kwargs : `dict`
            Extra arguments to metric, passed to scipy.spatial.distance.cdist.

        Returns
        -------
        ed : `pandas.Series`
            Euclidean distances between pairwise observations        
        """
        nbhd, profiles = self.neighborhood_profiles(node,include_self=include_self)
        if normalize:
            profiles = profiles / profiles.sum(axis=0)        
        ed = pairwise_observation_euc_distances(pd.DataFrame(data=profiles, columns=self.observations),
                                                metric=metric, **kwargs)
        return ed


    def anisotropic_laplacian_matrix(self, observation=None, use_spectral_gap=False, data=None):
        """ returns transpose of the  anisotropic random-walk  Laplacian matrix

        Parameters
        ----------
        observation : {`None`, `str`}
            If provided, use observation-weighted graph to construct the Laplacian. Otherwise, if `None`,
            The Laplacian is constructed from the unweighted graph, treated with uniform weights equal to 1.
        use_spectral_gap : `bool`
            Option to use spectral gap.
        data : {`None`, `str`, `pandas.DataFrame`}
            Specify which data the observation profile should be taken from.
            Note, this is ignored if ``observation`` is `None`.

            data options:

            - `None` : The original data is used.
            - `str` : `data` is is expected to be a key in ``self.meta`` and the observation profile is taken from the
                data in the correesponding dict-value.
            - `pandas.DataFrame` : The dataframe is used to select the observation profile, where columns are observations and the rows are
                expected to be labelled by the node indices, :math:`0, 1, ..., n-1` where
                :math:`n` is the number of nodes in the graph.

        Returns
        -------
        Lrw 
            Transpose of the random-walk graph Laplacian matrix.
        """
        G = self.G.copy()
        if observation is None:
            nx.set_node_weights(G, 1., name='weight')
        else:
            G = self.weighted_observation_network(observation, weight='weight', data=data)
        laplacian = utl.construct_anisotropic_laplacian_matrix(G, 'weight', use_spectral_gap=use_spectral_gap)
        return laplacian
    
    
    def diffuse_profile(self, observation, times=None, t_min=-1.5, t_max=2.5, n_t=10,
                        log_time=True, # graph_distances=None,
                        laplacian=None, do_save=True):
        """ diffuse profile from original data.

        ..todo:  get laplacian for observation or hop distance for all

        Parameters
        ----------
        observation : `str`
            Observation profile to use.
        times : {`None`, array}
            Array of times to evaluate the diffusion simulation.
            Note, If given, ``t_min``, ``t_max`` and ``n_t`` are ignored.
        t_min : `float`
            First time point to evaluate the diffusion simulation.
            Note, ``t_min`` is ignored if ``times`` is not `None`.
        t_max : `float`
            Last time point to evaluate the diffusion simulation.
            Note, ``t_max`` must be greater than ``t_min``, i.e, ``t_max`` > ``t_min``.
            Note, ``t_max`` is ignored if ``times`` is not `None`.
        n_t : `int`
            Number of time points to generate.
            Note, ``n_t`` is ignored if ``times`` is not `None`.
        log_time : `bool`
            If `True`, return ``n_t`` numbers spaced evenly on a log scale, where the time
            sequence starts at ``10 ** t_min``, ends with ``10 ** t_max``, and the
            sequence of times if of the form ``10 ** t`` where ``t`` is the ``n_t``
            evenly spaced points between (and including) ``t_min`` and ``t_max``.
            For example,
            ``_get_times(t_min=1, t_max=3, n_t=3, log_time=True) = array([10 ** 1, 10 ** 2, 10 ** 3]) = array([10., 100., 1000.])``.
            If `False`, return ``n_t`` numbers evenly spaced on a linear scale, where the sequence
            starts at ``t_min`` and ends with ``t_max``.
            For example, ``_get_times(t_min=1, t_max=3, n_t=3, log_time=False) = array([1. ,2., 3.])``.
        graph_distances : `numpy.ndarray`, (n, n)
            A matrix of node-pairwise graph distances between the :math:`n` nodes ordered by the rows in ``object.profiles``.
        laplacian : `numpy.ndarray`, (n, n)
            The transpose of the graph Laplacian matrix where the :math:`n` rows and columns  ordered by the rows in ``object.profiles``.
        filename : `str`
            If not `None`, save results.
        do_save : `bool`
            If `True`, save diffused profile to `self.outdir` / 'diffused_profiles' / 'diffused_profile_{``observation``}.csv'

        Returns
        -------
        profiles : `pandas.DataFrame`
            Diffused profiles where each row is a time and each column is a feature name.
        """
        if do_save:
            if not (self.outdir / 'diffused_profiles').is_dir():
                (self.outdir / 'diffused_profiles').mkdir()

        times = utl.get_times(times=times, t_min=t_min, t_max=t_max, n_t=n_t, log_time=log_time)
        times_with_zero = np.insert(times, 0, 0.0)
        profile = self.data[observation].loc[list(range(len(self.G)))].values
        # profile = profile[:, 0]
        # logger.msg(f"Profile shape = {profile.shape}.")
        profiles = np.inf * np.ones([len(times_with_zero), len(self.G)]) 
        profiles[0] = profile
        for time_index in tqdm(range(len(times)), desc="Diffusing profile", colour="green", leave=False):
            logger.debug("---------------------------------")
            logger.debug("Step %s / %s", str(time_index), str(len(times)))
            logger.debug("Computing diffusion time 10^{:.1f}".format(np.log10(times[time_index])))

            logger.debug("Computing measures")
            profile = utl.heat_kernel(profile, laplacian, times_with_zero[time_index + 1] - times_with_zero[time_index])
            profiles[time_index+1] = profile

        profiles = pd.DataFrame(data=profiles, index=times_with_zero,
                                # columns=list(range(len(self.G))),
                                columns=[self.G.nodes[k]['name'] for k in range(len(self.G))])
                                
        if do_save:
            profiles.to_csv(self.outdir / 'diffused_profiles' / 'diffused_profile_{}.csv'.format(observation),
                            header=True, index=True)
        return profiles

    
    def diffuse_multiple_profiles(self, observations=None, times=None, t_min=-1.5, t_max=2.5, n_t=10,
                                  log_time=True, # graph_distances=None,
                                  laplacian=None, use_spectral_gap=False, do_plot=False, **plot_kwargs):
        """ diffuse multiple observation profiles from original data

        # get laplacian for observation or hop distance for all

        Parameters
        ----------
        observations : {`None`, `list`}
            Observations to iterate over. If `None`, use all observations in data.
        times : {`None`, array}
            Array of times to evaluate the diffusion simulation.
            Note, If given, ``t_min``, ``t_max`` and ``n_t`` are ignored.
        t_min : `float`
            First time point to evaluate the diffusion simulation.
            Note, ``t_min`` is ignored if ``times`` is not `None`.
        t_max : `float`
            Last time point to evaluate the diffusion simulation.
            Note, ``t_max`` must be greater than ``t_min``, i.e, ``t_max`` > ``t_min``.
            Note, ``t_max`` is ignored if ``times`` is not `None`.
        n_t : `int`
            Number of time points to generate.
            Note, ``n_t`` is ignored if ``times`` is not `None`.
        log_time : `bool`
            If `True`, return ``n_t`` numbers spaced evenly on a log scale, where the time
            sequence starts at ``10 ** t_min``, ends with ``10 ** t_max``, and the
            sequence of times if of the form ``10 ** t`` where ``t`` is the ``n_t``
            evenly spaced points between (and including) ``t_min`` and ``t_max``. For example,            
            ``_get_times(t_min=1, t_max=3, n_t=3, log_time=True) = array([10 ** 1, 10 ** 2, 10 ** 3]) = array([10., 100., 1000.])``.
            If `False`, return ``n_t`` numbers evenly spaced on a linear scale, where the sequence
            starts at ``t_min`` and ends with ``t_max``.
            For example, ``_get_times(t_min=1, t_max=3, n_t=3, log_time=False) = array([1. ,2., 3.])``.
        graph_distances : `numpy.ndarray`, (n, n)
            A matrix of node-pairwise graph distances between the :math:`n` nodes ordered by the rows in ``object.profiles``.
        laplacian : `numpy.ndarray`, (n, n)
            The transpose of the graph Laplacian matrix where the :math:`n` rows and columns  ordered by the rows in ``object.profiles``.
            If `None`, the observation-specific Laplacian is used.
        use_spectral_gap : `bool`
            Option to use spectral gap.
            Note, This is ignored if ``laplacian`` is provided and not `None`.
        filename : `str`
            If not `None`,
        do_plot : `bool`
            If `True`, plot diffused profiles for each observation.
        **plot_kwargs : `dict`
            Key-word arguments passed to ``plot_profiles`` (should not include ``title``).
        

        Notes
        -----
        Side-effects :
        
        - Saves pandas DataFrame of the diffused profiles, where each row is a time and each column is a feature name,
            for each observation to the file name ``self.outdir`` / 'diffused_profiles' / 'diffused_profile_{``observation``}.csv'
        - If ``do_plot`` is `True`, plots the diffused profile for each observation.
        
        """
        if observations is None:
            observations = self.data.columns.tolist()

        for observation in tqdm(observations, desc='Diffusing observation profiles'):
            if laplacian is None:
                s_laplacian = self.anisotropic_laplacian_matrix(observation=observation, use_spectral_gap=use_spectral_gap)
            else:
                s_laplacian = laplacian

            profiles = self.diffuse_profile(observation, times=times, t_min=t_min, t_max=t_max, n_t=n_t,
                                            log_time=log_time, laplacian=s_laplacian, do_save=True)

            if do_plot:
                ax = self.plot_profiles(profiles, title=observation+": diffused profile", **plot_kwargs)


    def load_diffused_profile(self, observation):
        """ loads observation diffused profile

        Parameters
        ----------
        observation : `str`
            Observation profile to load.

        Returns
        -------
        diffused_profiles : `pandas.DataFrame`
            Diffused profiles where each row is a time and each column is a feature name.
        """
        diffused_profiles = pd.read_csv(self.outdir / 'diffused_profiles' / 'diffused_profile_{}.csv'.format(observation),
                                        header=0, index_col=0)

        return diffused_profiles


    def load_diffused_timepoint_profile(self, time, observations=None):
        """ loads observation diffused profile.

        Parameters
        ----------
        time : `float`
            Timepoint in diffusion simulation to select.
        observations : {`None`, `list`}
            List of observations to iterate over. If `None`, all observations in ``self.data`` are used.

        Returns
        -------
        diffused_profiles : `pandas.DataFrame`
            Diffused profiles at ``time``, for all ``observations``, where rows are nodes and columns are observations.
        """
        if observations is None:
            observations = self.data.columns.tolist()

        diffused_profiles = []
        for observation in tqdm(observations, desc='Loading diffused profiles at specified time'):
            profile = self.load_diffused_profile(observation)
            profile = profile.loc[time]
            profile.name = observation
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


    def multiple_pairwise_observation_neighborhood_wass_distance(self, nodes=None, include_self=False,
                                                                 graph_distances=None, label='pw_obs_nbhd_wass_dist',
                                                                 desc='Computing pairwise 1-hop distances',
                                                                 profiles_desc='t0',
                                                                 proc=mp.cpu_count(), chunksize=None,
                                                                 measure_cutoff=1e-6, solvr=None):
        """ Compute observation-pairwise Wasserstein distances between the profiles over node neighborhood for all nodes.

        Parameters
        ----------
        nodes : {`None`, `list` [`str`])}
            List of nodes to compute neighborhood distances on. If `None`, all genes with at least 2 neighbors is used.
        include_self : `bool`
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution over the neighborhood.
        graph_distances : `numpy.ndarray`, (n, n)
            A matrix of node-pairwise graph distances between the :math:`n` nodes (ordered from :math:`0, 1, ..., n-1`).
            If `None`, use hop distance.
        label : str
            Label that resulting Wasserstein distances are saved in ``keeper.misc``.
        desc : `str`
            Description for progress bar.
        profiles_desc : str, default = 't0'
            Description of profiles used in name of file to store results.
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
        wds : `pandas.DataFrame`
            Wasserstein distances between pairwise observations where rows are observation-pairs and columns are node names.
            This is saved in ``keeper.misc`` with the key ``label``.

        Notes
        -----
        If ``object.outdir`` is not `None`, Wasserstein distances are saved to file every 10 iterations.
        Before starting the computation, check if the file exists. If so, load and remove already computed
        nodes from the iteration. Wasserstein distances are computed for the remaining nodes, combined with
        the previously computed and saved results before saving and returning the combined results.

        Only nodes with at least 2 neighbors are included, as leaf nodes will all have the same Wassserstein distance
        and do not provide any further information.

        To do: specify if nodes in input are ids or node names and check that loaded data has correct type int or str for nodes
        """
        # check_matrix_nonnegative(self.data.data)
        if nodes is None:
            nodes = [k for k in self.G if len(list(self.G.neighbors(k)))>1]
        else:
            nodes = [self.name2node[k] for k in nodes if self.G.degree(self.name2node[k]) > 1]

        if self.outdir is None:
            fname = None
            wds_prior = None
        else:
            # fname = self.outdir / f"wass_dist_observation_pairwise_1hop_nbhd_profiles_{profiles_desc}_with{'' if include_self else 'out'}_self.csv"
            # fname = f"wass_dist_observation_pairwise_1hop_nbhd_profiles_{profiles_desc}_with{'' if include_self else 'out'}_self.csv"
            fname = f"{label}.csv"
            self.filenames.append(fname)
            if (self.outdir / fname).is_file():
                
                wds_prior = pd.read_csv(self.outdir / fname, header=0, index_col=(0, 1))                
                # logger.msg(f"Loaded saved observation pairwise 1-hop neighborhood Wasserstein distances from {profiles_desc} profiles of size {wds_prior.shape}.")
                logger.msg(f"Loaded saved observation pairwise 1-hop neighborhood Wasserstein distances from {label} profiles of size {wds_prior.shape}.")
                n_orig = len(nodes)
                nodes = [k for k in nodes if self.G.nodes[k]['name'] not in wds_prior.columns]
                n_update = len(nodes)
                if n_update < n_orig:
                    if n_update == 0:
                        # logger.msg(f"Loading observation pairwise 1-hop neighborhood Wasserstein distances from {profiles_desc} profiles.")
                        # return wds_prior
                        self.keeper.add_misc(wds_prior, label)
                        return None                    
                    # logger.msg(f"Computing observation pairwise 1-hop neighborhood Wasserstein distances from {profiles_desc} profiles on {n_update}/{n_orig} nodes.")
                    logger.msg(f"Computing observation pairwise 1-hop neighborhood Wasserstein distances from {label} profiles on {n_update}/{n_orig} nodes.")
            else:
                wds_prior = None

        wds = []

        # logger.msg(f"Computing Wasserstein distances on {len(nodes)} neighborhoods.")
        for ix, node in tqdm(enumerate(nodes), desc=desc, colour='yellow', total=len(nodes)):
            tmp = self.pairwise_observation_neighborhood_wass_distance(node, include_self=include_self, 
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
            # logger.msg(f"Observation pairwise 1-hop neighborhood Wasserstein distances on {profiles_desc} saved to {str(fname)}.")
            logger.msg(f"Observation pairwise 1-hop neighborhood Wasserstein distances on {label} saved to {str(fname)}.")            

        self.keeper.add_misc(wds, label)
        # return wds
        return None


    def multiple_pairwise_observation_neighborhood_euc_distance(self, nodes=None, include_self=False, label='pw_obs_nbhd_euc_dist',
                                                                desc='Computing pairwise 1-hop distances', profiles_desc='t0',
                                                                metric='euclidean', normalize=False, **kwargs):
        """ Compute observation-pairwise Euclidean distances between the profiles over node neighborhood for all nodes.

        Parameters
        ----------
        nodes : {`None`, `list`, [`int`]}
            List of nodes to compute neighborhood distances on. If `None`, all nodes with at least 2 neighbors are used.
        include_self : `bool`
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution over the neighborhood.
        label : str
            Label that resulting Euclidean distances are saved in ``keeper.misc``.
        desc : `str`
            Description for progress bar.
        profiles_desc : `str`, default = "t0"
            Description of profiles used in name of file to store results.
        metric : `str`
            The metric used to compute the distance, passed to scipy.spatial.distance.cdist.
        normalize : `bool`
            If `True`, normalize neighborhood profiles to sum to 1.
        **kwargs : `dict`
            Extra arguments to metric, passed to `scipy.spatial.distance.cdist`.

        Returns
        -------
        eds : `pandas.DataFrame`
            Euclidean distances between pairwise observations where rows are observation-pairs and columns are node names.
            This is saved in ``keeper.misc`` with the key ``label``.

        Notes
        -----
        If ``object.outdir`` is not `None`, Euclidean distances are saved to file.
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
            # if normalize:
            #     fname = f"euc_dist_observation_pairwise_1hop_nbhd_profiles_{profiles_desc}_with{'' if include_self else 'out'}_self_normalized.csv"
                
            # else:
            #     fname = f"euc_dist_observation_pairwise_1hop_nbhd_profiles_{profiles_desc}_with{'' if include_self else 'out'}_self.csv"
            fname = f"{label}.csv"
            self.filenames.append(fname)
            if (self.outdir / fname).is_file():                
                eds_prior = pd.read_csv(self.outdir / fname, header=0, index_col=(0, 1))
                # logger.msg(f"Loaded saved observation pairwise 1-hop neighborhood Euclidean distances from {profiles_desc} profiles of size {eds_prior.shape}.")
                logger.msg(f"Loaded saved observation pairwise 1-hop neighborhood Euclidean distances from {label} profiles of size {eds_prior.shape}.")
                n_orig = len(nodes)
                nodes = [k for k in nodes if self.G.nodes[k]['name'] not in eds_prior.columns]
                n_update = len(nodes)
                if n_update < n_orig:
                    if n_update == 0:
                        # return eds_prior
                        self.keeper.add_misc(eds_prior, label)
                        return None
                    # logger.msg(f"Computing observation pairwise 1-hop neighborhood Euclidean distances from {profiles_desc} profiles on {n_update}/{n_orig} nodes.")
                    logger.msg(f"Computing observation pairwise 1-hop neighborhood Euclidean distances from {label} profiles on {n_update}/{n_orig} nodes.")
            else:
                eds_prior = None

        eds = []

        for node in tqdm(nodes, desc=desc, colour='yellow', total=len(nodes)):
            tmp = self.pairwise_observation_neighborhood_euc_distance(node, include_self=include_self,
                                                                 metric=metric, normalize=normalize, **kwargs)
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
            # logger.msg(f"Observation pairwise 1-hop neighborhood Euclidean distances on {profiles_desc} saved to {str(fname)}.")
            logger.msg(f"Observation pairwise 1-hop neighborhood Euclidean distances on {label} saved to {str(fname)}.")

        self.keeper.add_misc(eds, label)
        # return eds
        return None


    def pairwise_observation_profile_wass_distance(self, features=None,
                                                   graph_distances=None, label='wass_dist_observation_pairwise_profiles_t0',
                                                   desc='Computing pairwise profile Wasserstein distances',
                                                   proc=mp.cpu_count(), chunksize=None,
                                                   measure_cutoff=1e-6, solvr=None):
        """ Compute observation-pairwise Wasserstein distances between the profiles over selected features.

        Parameters
        ----------
        features : {`None`, `list` [`str`])}
            List of features to compute profile distances on. If `None`, all features are used.
        graph_distances : `numpy.ndarray`, (n, n)
            A matrix of node-pairwise graph distances between the :math:`n` nodes (ordered from :math:`0, 1, ..., n-1`).
            If `None`, use hop distance.
        label : str
            Label that resulting Wasserstein distances are saved in ``keeper.distances`` and
            name of file to store stacked results.
        desc : `str`
            Description for progress bar.
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
        wds : `pandas.DataFrame`
            Wasserstein distances between pairwise profiles where rows are observation-pairs and columns are node names.
            This is saved in ``keeper.distances`` with the key ``label``.

        Notes
        -----
        If ``object.outdir`` is not `None`, Wasserstein distances are saved to file every 10 iterations.
        Before starting the computation, check if the file exists. If so, load and remove already computed
        nodes from the iteration. Wasserstein distances are computed for the remaining nodes, combined with
        the previously computed and saved results before saving and returning the combined results.

        Only nodes with at least 2 neighbors are included, as leaf nodes will all have the same Wassserstein distance
        and do not provide any further information.

        To do: specify if nodes in input are ids or node names and check that loaded data has correct type int or str for nodes

        SAVED TO DISTANCES
        """
        # check_matrix_nonnegative(self.data.data)
        if features is None:
            features = list(self.G)
        else:
            features = [self.name2node[k] for k in features]
        
        pw_obs = list(combinations(self.observations, 2))

        logger.info(f">>> computing {len(pw_obs)} pw-obs distances")

        if self.outdir is None:
            fname = None
            wds_prior = None
        else:
            # fname = self.outdir / f"wass_dist_observation_pairwise_1hop_nbhd_profiles_{profiles_desc}_with{'' if include_self else 'out'}_self.csv"
            fname = f"{label}.csv"
            self.filenames.append(fname)
            if (self.outdir / fname).is_file():
                
                wds_prior = pd.read_csv(self.outdir / fname, header=0, index_col=(0, 1))
                logger.msg(f"Loaded saved observation pairwise profile Wasserstein distances from {label} profiles of size {wds_prior.shape}.")

                n_orig = len(pw_obs)
                
                # logger.msg(f">>> n_orig = {n_orig}")
                # pw_obs = [k for k in pw_obs if k not in set(wds_prior.index)]
                # n_update = len(pw_obs)
                # logger.msg(f">>> n_update / n_orig = {n_update} / {n_orig}.")
                # if n_update < n_orig:
                #     if n_update == 0:
                if wds_prior.shape[0] == len(pw_obs):
                    if True:
                        # logger.msg(f"Loading observation pairwise profile Wasserstein distances from {label}.csv.")
                        # return wds_prior
                        # self.keeper.add_misc(wds_prior, label)
                        logger.msg(f">>> WD TO UPDATE")
                        if isinstance(wds_prior, pd.DataFrame):
                            wds_prior = wds_prior[wds_prior.columns[0]]
                        logger.msg(f">>> WD to stacked distance")
                        self.keeper.add_stacked_distance(wds_prior, label)
                        logger.msg(f">>> stacked distance updated - should return now.")
                        return None
                    
                    # logger.msg(f"Computing observation pairwise profile Wasserstein distances between {n_update}/{n_orig} pairwise observations.")
            else:
                wds_prior = None

        profiles = self.data.subset(features=features)

        if graph_distances is None:
            logger.msg("Computing graph hop distances.")
            # TO DO: SHOULDN"T COMPUTE ALL DISTANCES HERE
            graph_distances = self.compute_graph_distances(weight=None)            
        graph_distances = graph_distances[np.ix_(features, features)]

        # logger.msg(f">>> {len(pw_obs)} pw-obs left to compute")

        wds = pairwise_observation_wass_distances(profiles,
                                                  graph_distances, proc=proc, pairwise_obs_list=pw_obs,
                                                  chunksize=chunksize,
                                                  measure_cutoff=measure_cutoff, solvr=solvr, flag=f"pairwise-profiles")
        wds.name = 'WD'


        if wds_prior is not None:
            wds = pd.concat([wds_prior, wds], axis=0)
            # only save if new wds were computed:
            if self.outdir is not None: 
                wds.to_csv(str(self.outdir / fname), header=True, index=True)
                logger.msg(f"Observation pairwise profile Wasserstein distances saved to {str(fname)}.")

        # logger.msg(f">>> wds made it to here with shape {wds.shape}.")

        ## self.keeper.add_misc(wds, label)

        # ensure wds is a Series and not a DataFrame
        # logger.msg(f">>> wds is a {type(wds)}")
        if isinstance(wds, pd.DataFrame):
            wds = wds[wds.columns[0]]
        # logger.msg(f">>> wds is now a {type(wds)}")

        # logger.msg(f">>> ABOUT TO SAVE WDS TO KEEPER AS STACKED DISTANCE: type={type(wds)}, size={wds.shape}.")

        # logger.msg(f">>> {help(self.keeper.add_stacked_distance)}.")
            
        self.keeper.add_stacked_distance(wds, label)
        # return wds
        return None


    def pairwise_observation_profile_euc_distance(self, features=None, label='euc_dist_observation_pairwise_profiles_t0',
                                                  desc='Computing pairwise profile Euclidean distances',
                                                  metric='euclidean', normalize=False, **kwargs):
        """ Compute observation-pairwise Euclidean distances between the profiles over selected features.

        Parameters
        ----------
        features : {`None`, `list`, [`int`]}
            List of features to compute profile distances on. If `None`, all features are used.
        label : str
            Label that resulting Euclidean distances are saved in ``keeper.distances`` and
            name of file to store stacked results.
        desc : `str`
            Description for progress bar.
        metric : `str`
            The metric used to compute the distance, passed to scipy.spatial.distance.cdist.
        normalize : `bool`
            If `True`, normalize neighborhood profiles to sum to 1.
        **kwargs : `dict`
            Extra arguments to metric, passed to `scipy.spatial.distance.cdist`.

        Returns
        -------
        eds : `pandas.DataFrame`
            Euclidean distances between pairwise observations where rows are observation-pairs and columns are node names.
            This is saved in ``keeper.distances`` with the key ``label``.

        Notes
        -----
        If ``object.outdir`` is not `None`, Euclidean distances are saved to file.
        Before starting the computation, check if the file exists. If so, load and remove already computed
        nodes from the iteration. Euclidean distances are computed for the remaining nodes, combined with
        the previously computed and saved results before saving and returning the combined results.

        SAVES TO DISTANCES
        """
        if features is None:
            features = list(self.features) # list(self.G)
        else:
            features = [self.name2node[k] for k in features]

        pw_obs = list(combinations(self.observations, 2))

        if self.outdir is None:
            fname = None
            eds_prior = None
        else:
            if normalize:
                fname = f"{label}_self_normalized.csv"
            else:
                fname = f"{label}.csv"
            self.filenames.append(fname)
            if (self.outdir / fname).is_file():                
                eds_prior = pd.read_csv(self.outdir / fname, header=0, index_col=(0, 1))
                logger.msg(f"Loaded saved observation pairwise profile Euclidean distances from {label} profiles of size {eds_prior.shape}.")

                n_orig = len(pw_obs)
                # pw_obs = [k for k in pw_obs if k not in set(eds_prior.index)]
                # n_update = len(pw_obs)
                
                # if n_update < n_orig:
                #     if n_update == 0:
                if eds_prior.shape[0] == n_orig:
                    if True:
                        # return eds_prior
                        # self.keeper.add_misc(eds_prior, label)
                        if isinstance(eds_prior, pd.DataFrame):
                            eds_prior = eds_prior[eds_prior.columns[0]]
                            self.keeper.add_stacked_distance(eds_prior, label)
                        return None

                    # logger.msg(f"Computing observation pairwise profile Euclidean distances between {n_update}/{n_orig} pairwise observations.")
            # else:
            #     eds_prior = None

        # if len(pw_obs)>0:
        profiles = self.data.subset(features=features)

        if normalize:
            profiles = profiles / profiles.sum(axis=0)

        eds = pairwise_observation_euc_distances(profiles,
                                                 metric=metric, **kwargs)

        # line 1025
        eds.name = 'ED'

        # if self.outdir is not None:
        #     eds.to_csv(self.outdir / fname, header=True, index=True)
        #     logger.msg(f"Observation pairwise profile Euclidean distances saved to {str(fname)}.")
            
        # if eds_prior is not None:
        #     eds = pd.concat([eds_prior, eds], axis=0)
        # if self.outdir is not None:
        #     eds.to_csv(self.outdir / fname, header=True, index=True)
        #     logger.msg(f"Observation pairwise profile Euclidean distances saved to {str(fname)}.")


        # ensure wds is a Series and not a DataFrame
        if isinstance(eds, pd.DataFrame):
            eds = eds[eds.columns[0]]

        # logger.msg(f">>> Euc - type(eds) = {type(eds)}")
            
        # logger.msg(f">>> ABOUT TO SAVE EUC TO KEEPER AS STACKED DISTANCE: type={type(eds)}, size={eds.shape}.")
        self.keeper.add_stacked_distance(eds, label)        
        # return eds
        return None



            




    
# if __name__ == '__main__':
#     G = nx.karate_club_graph()
#     data = pd.DataFrame(data=np.abs(np.random.randn(len(G), 9)) + 0.001,
#                         # data=np.array([[1.]*len(G), [2.]*len(G), [3.]*len(G),
#                         #                [4.]*len(G), [5.]*len(G), [6.]*len(G),
#                         #                [7.]*len(G), [8.]*len(G), [9.]*len(G)]).transpose(), index=list(G),
#                         columns=[f"x{k}" for k in range(9)])

#     logger.msg(f"Data is size {data.shape[0]} x {data.shape[1]}.")
#     inet = InfoNet(G, data, './')
#     g = inet.weighted_observation_network('x2', weight='weight')
#     print(nx.get_node_attributes(g, 'weight'))


#     dhop = inet.compute_graph_distances(weight=None)
#     print(dhop.shape)
#     # print(inet.neighborhood_profiles(16, include_self=False))

#     # logger.msg(f"Wass = {wass_distance(('x1', 'x2'), data, dhop, measure_cutoff=1e-6)}.")

#     # wds16 = inet.pairwise_observation_neighborhood_wass_distance(16, include_self=False, graph_distances=dhop) # , graph_distances=dhop, proc=mp.cpu_count(), chunksize=None)
    

#     # print(wds16)

#     wds, eds = [], []

    
#     for gn in [k for k in G if len(list(G.neighbors(k)))>1]:
#         tmp = inet.pairwise_observation_neighborhood_wass_distance(gn, include_self=False, graph_distances=dhop, measure_cutoff=1e-6)
#         tmp.name = gn
#         wds.append(tmp)

#         tmp = inet.pairwise_observation_neighborhood_euc_distance(gn, include_self=False, metric='euclidean')
#         tmp.name=gn
#         eds.append(tmp)

#     wds = pd.concat(wds, axis=1)
#     eds = pd.concat(eds, axis=1)
#     print("ranges: ", wds.max().max(), wds.min().min(), eds.max().max(), eds.min().min())
#     print(wds.head())


#     wds_a = inet.multiple_pairwise_observation_neighborhood_wass_distance(data=None,
#                                                                      graph_distances=dhop, desc='Computing pairwise 1-hop distances',
#                                                                      profiles_desc='t0',
#                                                                      proc=mp.cpu_count(), chunksize=None,
#                                                                      measure_cutoff=1e-6, solvr=None)
#     print(f"And automated wds: ")
#     print("ranges: ", wds_a.max().max(), wds_a.min().min())
#     print(wds_a.head())
    
#     # print(wds.corr(method='spearman'))

#     # import seaborn as sns
#     # import matplotlib.pyplot as plt
#     # sns.clustermap(wds.corr(method='spearman'), method='ward', cmap='RdBu', center=0)
#     # plt.gcf().suptitle('Spearman corr wds');
    
#     # sns.clustermap(eds.corr(method='spearman'), method='ward', cmap='RdBu', center=0)
#     # plt.gcf().suptitle('Spearman corr eds');

#     # sns.clustermap(1. - (wds / eds), method='ward', cmap='RdBu', center=0)    
#     # plt.gcf().suptitle('Observation curvatures');


#     # controlled distance ranges
#     # wds2 = wds / wds.max().max()
#     # eds2 = eds / eds.max().max()
#     # sns.clustermap(1. - (wds2 / eds2), method='ward', cmap='RdBu', center=0)    
#     # plt.gcf().suptitle('Observation scaled curvatures');

#     plt.show()


#     # compute feature distance
#     import seaborn as sns
#     import netflow.pseudotime as nfp
#     import netflow.utils as nfu
#     print("Computing feature distances:")
#     wf = pd.Series(data=nfp.norm_features(wds, method='L2'),
#                    index=wds.index.copy())
#     print(wf.shape, wf, sep='\n')

#     A_wf = nfu.unstack_triu_(wf, index=data.columns.tolist())
#     print(A_wf)

#     sns.heatmap(A_wf)
#     plt.show()
