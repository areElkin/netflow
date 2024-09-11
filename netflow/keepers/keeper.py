"""
keeper
======

Classes used for data storage and manipulation. 
"""

from pathlib import Path

import multiprocessing as mp
import networkx as nx
import numpy as np
import pandas as pd

from .. import checks
from .._utils import _docstring_parameter, _desc_distance, \
    _desc_data_distance, load_from_file, fuse_labels
from ..utils import unstack_triu_
from ..methods.classes import InfoNet
from ..pose import similarity as nfs
from ..pose.organization import compute_transitions, dpt_from_augmented_sym_transitions, POSER
# import netflow.InfoNet as InfoNet
from .._logging import _gen_logger, set_verbose



logger = _gen_logger(__name__)


class DataKeeper:
    """ A class to store and handle multiple data sets.

    Parameters
    ----------
    data : {`numpy.ndarray`, `pandas.DataFrame`, `dict` [`str`, `numpy.ndarray`], `dict` [`str`, `pandas.DataFrame`]}
        One or multiple feature data sets, where each data set is size (num_features, num_observations).

        Feature data set(s) may be provided in multiple ways:

        - `numpy.ndarray` : A single feature data set.

          * Observation labels may be specified in ``observation_labels``.
          * To include feature labels, ``data`` should be a `pandas.DataFrame`.
          * The array ``data`` is placed in a `dict` with default label of the form
            ``{'data' : data}``. 
        - `pandas.DataFrame` : A single feature data set.
    
          * The index indicates feature labels.
          * Columns indicate observation labels.
          * If ``observation_labels`` is provided, it should match the columns of the dataframe,
            that will be ordered according to the ``observation_labels``.
          * The array ``data.values`` is placed in a `dict` with default label of the form
            ``{'data' : data.values}``. 
        - `dict` [`str`, `numpy.ndarray`] : A single or mutliple feature data set(s) may be provided as 
           the value(s) of a `dict` keyed by a `str` that serves as the feature data descriptor or label 
           for each input of the form ``{'data_label' : `numpy.ndarray`}``.

          * All arrays are expected to have the same number of columns corresponding to the number
            of observations with the same ordering.
          * Observation labels may be specified in ``observation_labels``.
          * To include feature labels, pass a `dict` of `pandas.DataFrame`\s instaed.
        - `dict` [`str`, `pandas.DataFrame`] : A single or mutliple feature data set(s) may be provided as 
           the value(s) of a `dict` keyed by a `str` that serves as the feature data descriptor or label 
           for each input of the form ``{'data_label' : `pandas.DataFrame`}``.

          * All dataframes are expected to have the same number of columns corresponding to the number
            of observations with the same columns provided in the same order.
          * The index of each input indicates feature labels.
          * Columns indicate observation labels.
          * If ``observation_labels`` is provided, it should match the columns of the dataframe(s),
            that will be ordered according to the ``observation_labels``.
    observation_labels : `list` [`str`], optional
        List of labels for each observation with length equal to num_observations.
        If provided when ``data`` is a `pandas.DataFrame` or `dict` [`str`, `pandas.DataFrame`],
        it should match the columns of the dataframe(s), which will be ordered according to
        ``observation_labels``.
    
        If not provided and ``data`` is a :

        - `pandas.DataFrame` or `dict` [`str`, `pandas.DataFrame`],
          then the columns of the (first) dataframe are used.
        - `numpy.ndarray` or `dict` [`str`, `numpy.ndarray`],
          then the columns are labeled by their row index from :math:`0, 1, ..., num\_observations - 1`.

    Notes
    -----
    All data sets are assumed to contain the same set of data points in the same order.
    """

    def __init__(self, data=None, observation_labels=None):        
        self._data = {}
        if observation_labels is not None:
            if len(observation_labels) != len(set(observation_labels)):
                raise ValueError("Observation labels must be unique")
        self._observation_labels = observation_labels
        self._num_observations = None if observation_labels is None else len(observation_labels)
        self._features_labels = {}
        self._num_features = {}

        if isinstance(data, (pd.DataFrame, np.ndarray)):
            self.add_data(data, 'data')
        elif isinstance(data, dict):
            for label, cur_data in data.items():
                self.add_data(cur_data, label)
        elif data is None:
            pass
        else:
            raise TypeError("Unrecognized type for data, must be one of [pandas.DataFrame, numpy.ndarray, dict].")            


    def __getitem__(self, key):
        # return self._data[key]
        return DataView(dkeeper=self, label=key)


    def __contains__(self, key):
        return key in self._data


    def __iter__(self):
        for key in self._data.keys():
            yield DataView(dkeeper=self, label=key)


    @property
    def data(self):
        """ A dictionary of all data sets. """
        return self._data


    @property
    def observation_labels(self):
        """ Labels for each observation. """
        return self._observation_labels


    def observation_index(self, observation_label):
        """ Return index of observation.

        Parameters
        ----------
        observation_label : `str`
            The observation label.

        Returns
        -------
        observation_index : `int`
            Index of observation in list of observation labels.
        """
        observation_index = self._observation_labels.index(observation_label)
        return observation_index


    # @observation_labels.setter
    # def observation_labels(self, labels):
    #     if self._observation_labels is not None:
    #         raise ValueError("Observation labels cannot be changed.")

    #     if (self._num_observations is not None) and (self._num_observations != len(labels)):
    #         raise ValueError("Inconsisstent size, number of labels must match the number of observations.")
            
    #     self._observation_labels = labels.copy()


    @property
    def num_observations(self):
        """ Number of observations. """
        return self._num_observations


    @property
    def features_labels(self):
        """ A dictionary of feature labels for each data set. """
        return self._features_labels


    @property
    def num_features(self):
        """ A dictionary with the number of features in each data set. """
        return self._num_features 


    def add_data(self, data, label):
        """ Add a feature data set to the keeper.

        .. Note::

           Observation labels may be set after object initialization.
           This may require upkeep in :class:`Keeper` to update and coordinate
           observation labels among the different keepers.
            
        Parameters
        ----------
        data : {`numpy.ndarray`, `pandas.DataFrame`}
            The data set of size (num_features, num_observations).
        label : `str`
            Reference label describing the data set.        
        """
        # Check that label is not already in the keeper
        if label in self._data:
            raise KeyError(f"Dubplicate label detected, {label} already exists in the keeper.")
        
        # Check that data is numpy ndarray or pandas DataFrame:
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("Unrecognized type, data must be numpy ndarray or pandas DataFrame.")
        
        # If no data in keeper yet, initialize:
        if len(self._data) == 0:
            self._num_observations = data.shape[1] if self._num_observations is None else self._num_observations
            if (self._observation_labels is not None) and (len(self._observation_labels) != self._num_observations):
                raise ValueError("Inconsistent data size, must be the same as num_observations.")
            
            # if isinstance(data, pd.DataFrame):
        #         self._observation_labels = data.columns.tolist()  # overrides pre-specified observation_labels
        #     else:  # np.ndarray
        #         self._observation_labels = self._observation_labels if self._observation_labels is not None \
        #             else list(range(self._num_observations))
            
        # else:  # Check that the data has the right number of columns        
        #     if data.shape[1] != self._num_observations:
        #         raise ValueError("Inconsistent data size, must be the same as num_observations")

        # Check that the data has the right number of columns        
        if data.shape[1] != self._num_observations:
            raise ValueError("Inconsistent data size, must be the same as num_observations")
        
        if isinstance(data, pd.DataFrame):
            if self._observation_labels is None:
                if len(set(data.columns)) != data.shape[1]:
                    raise ValueError("Observation labels must be unique")
                self._observation_labels = data.columns.tolist()
                logger.debug("Set observation labels.")
            if len(set(data.index)) != data.shape[0]:
                raise ValueError("feature labels must be unique")
            feature_labels = data.index.tolist()
            data = data[self._observation_labels].values
        else:  # np.ndarray
            feature_labels = None        
        
        # checks.check_matrix_no_nan(data)
        # checks.check_matrix_nonnegative(data)
        self._data[label] = data
        self._features_labels[label] = feature_labels
        self._num_features[label] = data.shape[0]

        logger.debug(f"Data {label} with {data.shape[0]} features added to keeper.")


    def subset(self, observations):
        """ Return a new instance of DataKeeper restricted to subset of observations

        Parameters
        ----------
        observations : `list`, 
            List of observations to include in the subset.
            This is treated differently depending on the type of observation labels :

            - If ``self._observation_labels`` is `List` [`str`], ``observations`` can be of the form:
        
              * `List` [`str`], to reference observations by their `str` label or;
              * `List` [`int`], to reference observations by their location index.

            - If ``self._observation_labels`` is `None` or `List` [`int`], ``observations`` must 
              be of the form `List` [`int`], to reference observations by their location index.

        Returns
        -------
        data_subset : `DataKeeper`
            A DataKeeper object restricted to the selected observations.
        """
        data_subset = DataKeeper(data=None, observation_labels=observations)
        for data in self:
            data_subset.add_data(data.subset(observations=observations), data.label)
        return data_subset    
            

class DataView:
    """ A class to extract single data view from DataKeeper for analysis.
    
    Parameters
    ----------
    dkeeper : `DataKeeper`
        The object to extract data from.        
    label : `str`
        The identifier of the data to be extracted from ``dkeeper``.
    """

    def __init__(self, dkeeper=None, label=None,
                 # data=None, observation_labels=None, feature_labels=None,
                 ):
        if label not in dkeeper.data:
            raise KeyError(f"No {label} in dkeeper.")

        self._label = label
        self._data = dkeeper.data[label]
        self._observation_labels = dkeeper.observation_labels        
        self._num_observations = dkeeper.num_observations
        self._feature_labels = dkeeper.features_labels[label]
        self._num_features = dkeeper.num_features[label]
        

    @property
    def label(self):
        """ The data label. """
        return self._label

    
    @property
    def data(self):
        """ The data set. """
        return self._data


    @property
    def observation_labels(self):
        """ Labels for each observation. """
        return self._observation_labels


    @property
    def num_observations(self):
        """ Number of observations. """
        return self._num_observations


    @property
    def feature_labels(self):
        """ Feature labels. """
        return self._feature_labels


    @property
    def num_features(self):
        """ The number of features in the data set. """
        return self._num_features 



    def subset(self, observations=None, features=None):
        """ Return data for specified subset of observations and/or features.

        Parameters
        ----------
        observations : `list`, 
            List of observations to include in the subset.
            This is treated differently depending on the type of observation labels :

            - If ``self._observation_labels`` is `List` [`str`], ``observations`` can be of the form:
        
              * `List` [`str`], to reference observations by their `str` label or;
              * `List` [`int`], to reference observations by their location index.

            - If ``self._observation_labels`` is `None` or `List` [`int`], ``observations`` must 
              be of the form `List` [`int`], to reference observations by their location index.
        features : `list`
            List of features to include in the subset.
            The list type depends on the type of ``self._feature_labels``, analogous to
            ``self._observation_labels`` for ``observations`` above.

        Returns
        -------
        data : `pandas.DataFrame`, (``len(observations)``, ``len(features)``)
            The data subset where the :math:`ij^{th}` entry is the value of the
            :math:`i^{th}` feature in ``features`` for the  :math:`j^{th}`
            observation in ``observations``.
        """
        if (observations is None) and (features is None):
            raise ValueError("Either observations or features must be provided.")

        data = self._data
        if observations is None:
            observations = self.observation_labels            
        # if observations is not None:
        #     if isinstance(observations[0], str):  # convert to location index
        #         observations = [self._observation_labels.index(k) for k in observations]
        # data = data[:, observations]
        if isinstance(observations[0], str):  # convert to location index
            observations_ix = [self._observation_labels.index(k) for k in observations]
        else:
            observations_ix = observations
        data = data[:, observations_ix]

        if features is None:
            features = self.feature_labels
        # if features is not None:
        #     if isinstance(features[0], str):  # convert to location index
        #         features = [self._feature_labels.index(k) for k in features]
        #     data = data[features, :]
        if isinstance(features[0], str):  # convert to location index
            features_ix = [self._feature_labels.index(k) for k in features]
        else:
            features_ix = features
        data = data[features_ix, :]

        return pd.DataFrame(data=data, columns=observations, index=features)


    def observation_index(self, observation_label):
        """ Return index of observation.

        Parameters
        ----------
        observation_label : `str`
            The observation label.

        Returns
        -------
        observation_index : `int`
            Index of observation in list of observation labels.
        """
        observation_index = self._observation_labels.index(observation_label)
        return observation_index


    def feature_index(self, feature_label):
        """ Return index of feature.

        Parameters
        ----------
        feature_label : `str`
            The feature label.

        Returns
        -------
        feature_index : `int`
            Index of feature in list of feature labels.
        """
        feature_index = self._feature_labels.index(feature_label)
        return feature_index


    def to_frame(self):
        """ Return data as a pandas DataFrame """
        df = pd.DataFrame(data=self.data,
                          columns=self.observation_labels,
                          index=self.feature_labels)
        return df
    

        
class DistanceKeeper:
    """ A class to store and handle multiple distances.

    This class may also be used to store and handle similarity matrices.
    Distance may be interchanged with similarity, but distance is used for simplicity.
    

    Parameters
    ----------
    data : {`numpy.ndarray`, `pandas.DataFrame`, `dict` [`str`, `numpy.ndarray`], `dict` [`str`, `pandas.DataFrame`]}
        One or multiple symmetric distance(s) between observations, where each distance matrix is size
        (num_observations, num_observations).

        Distance(s) may be provided in multiple ways:

        - `numpy.ndarray` : A single distance matrix.

          * Observation labels may be specified in ``observation_labels``.
          * To include feature labels, ``data`` should be a `pandas.DataFrame`.
          * The array ``data`` is placed in a `dict` with default label of the form ``{'distance' : data}``. 
        - `pandas.DataFrame` : A single distance matrix.
    
          * The index should be the same as the columns, which indicate observation labels.
          * If ``observation_labels`` is provided, it should match the rows and columns of the dataframe,
            that will be ordered according to the ``observation_labels``.
          * The array ``data.values`` is placed in a `dict` with default label of the form
            ``{'distance' : data.values}``. 
        - `dict` [`str`, `numpy.ndarray`] : A single or mutliple distance(s) may be provided as the value(s) 
           of a `dict` keyed by a `str` that serves as the distance descriptor or label for each
           input of the form ``{'distance_label' : `numpy.ndarray`}``.

          * All arrays are expected to have the same number of columns and rows corresponding to the number
            of observations with the same ordering.
          * Observation labels may be specified in ``observation_labels``.
        - `dict` [`str`, `pandas.DataFrame`] : A single or mutliple distance(s) may be provided as the value(s) 
           of a `dict` keyed by a `str` that serves as the distance descriptor or label for each
           input of the form ``{'distance_label' : `pandas.DataFrame`}``.

          * All dataframes are expected to have the same number of rows and columns corresponding 
            to the number of observations with the same columns provided in the same order.
          * The index and columns indicate observation labels.
          * If ``observation_labels`` is provided, it should match the columns of the dataframe(s),
            that will be ordered according to the ``observation_labels``.
    observation_labels : `list` [`str`]
        List of labels for each observation with length equal to num_observations.
        If provided when ``data`` is a `pandas.DataFrame` or `dict` [`str`, `pandas.DataFrame`],
        it should match the columns of the dataframe(s), which will be ordered according to
        ``observation_labels``.
    
        If not provided and ``data`` is a :

        - `pandas.DataFrame` or `dict` [`str`, `pandas.DataFrame`],
          then the columns of the (first) dataframe are used.
        - `numpy.ndarray` or `dict` [`str`, `numpy.ndarray`],
          then the columns are labeled by their row index from :math:`0, 1, ..., num_observations - 1`.

    Notes
    -----
    All sets are assumed to contain the same set of data points in the same order.
    """

    def __init__(self, data=None, observation_labels=None):
        self._data = {}
        if observation_labels is not None:
            if len(observation_labels) != len(set(observation_labels)):
                raise ValueError("Observation labels must be unique")
        self._observation_labels = observation_labels
        self._num_observations = None if observation_labels is None else len(observation_labels)

        if isinstance(data, (pd.DataFrame, np.ndarray)):
            self.add_data(data, 'distance')
        elif isinstance(data, dict):
            for label, cur_distance in data.items():
                self.add_data(cur_distance, label)
        elif data is None:
            pass
        else:
            raise TypeError("Unrecognized type for data, must be one of [pandas.DataFrame, numpy.ndarray, dict].")


    def __getitem__(self, key):
        # return self._data[key]
        return DistanceView(dkeeper=self, label=key)


    def __contains__(self, key):
        return key in self._data


    def __iter__(self):
        for key in self._data.keys():
            yield DistanceView(dkeeper=self, label=key)

    
    @property
    def data(self):
        """\ A dictionary of all distances. """
        return self._data


    @property
    def observation_labels(self):
        """ Labels for each observation. """
        return self._observation_labels


    @property
    def num_observations(self):
        """ Number of observations. """
        return self._num_observations


    @_docstring_parameter(desc=_desc_distance, desc_data=_desc_data_distance)
    def add_data(self, data, label):
        """\
        {desc}

        Parameters
        ----------
        {desc_data}
        label : `str`
            Reference label describing the input.
        """
        # Check that label is not already in the keeper
        if label in self._data:
            raise KeyError(f"Dubplicate label detected, {label} already exists in the keeper.")
        
        # Check that distance is numpy ndarray or pandas DataFrame:
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("Unrecognized type, data must be numpy ndarray or pandas DataFrame")
        
        # If no distance in keeper yet, initialize:
        if len(self._data) == 0:
            self._num_observations = data.shape[1] if self._num_observations is None else self._num_observations
            if (self._observation_labels is not None) and (len(self._observation_labels) != self._num_observations):
                raise ValueError("Inconsistent data size, must be the same as num_observations.")
            
        # Check that the distance has the right number of rows and columns        
        if (data.shape[1] != self._num_observations) or (data.shape[0] != self._num_observations):
            raise ValueError("Inconsistent distance matrix size, should be (num_observations, num_observations).")

        if isinstance(data, pd.DataFrame):
            if self._observation_labels is None:
                if len(set(data.columns)) != data.shape[1]:
                    raise ValueError("Observation labels must be unique")
                self._observation_labels = data.columns.tolist()
                logger.debug("Set observation labels.")
            data = data.loc[self._observation_labels, self._observation_labels].values


        checks.check_matrix_no_nan(data)
        checks.check_distance_matrix(data)

        self._data[label] = data

        logger.debug(f"Distance {label} between {data.shape[0]} observations added to keeper.")

    def add_stacked_data(self, data, label, diag=0.):
        """ Add a symmetric distance from stacked Series to the keeper.

        Parameters
        ----------
        data : `pandas.Series`
            The stacked distances of size (num_observations * (num_observations - 1) / 2,)
            with a 2-multi-index of the pairwise observation labels.
        label : `str`
            Reference label describing the input.
        diag : `float`
            Value used on diagonal. 
        """
        # Check that label is not already in the keeper
        if label in self._data:
            raise KeyError(f"Dubplicate label detected, {label} already exists in the keeper.")

        data = unstack_triu_(data, diag=diag, index=self.observation_labels)

        self.add_data(data, label)
        

    def observation_index(self, observation_label):
        """ Return index of observation.

        Parameters
        ----------
        observation_label : `str`
            The observation label.

        Returns
        -------
        observation_index : `int`
            Index of observation in list of observation labels.
        """
        observation_index = self._observation_labels.index(observation_label)
        return observation_index


    def subset(self, observations):
        """ Return a new instance of DistanceKeeper restricted to subset of observations

        Parameters
        ----------
        observations : `list`, 
            List of observations to include in the subset.
            This is treated differently depending on the type of observation labels :

            - If ``self._observation_labels`` is `List` [`str`], ``observations`` can be of the form:
        
              * `List` [`str`], to reference observations by their `str` label or;
              * `List` [`int`], to reference observations by their location index.

            - If ``self._observation_labels`` is `None` or `List` [`int`], ``observations`` must 
              be of the form `List` [`int`], to reference observations by their location index.

        Returns
        -------
        distance_subset : `DistanceKeeper`
            A DistnaceKeeper object restricted to the selected observations.
        """
        distance_subset = DistanceKeeper(data=None, observation_labels=observations)
        for data in self:
            distance_subset.add_data(data.subset(observations_a=observations), data.label)
        return distance_subset



class DistanceView:
    """ A class to extract single distance matrix view from DistanceKeeper for analysis.

    This class may also be used to extract a similarity matrix.
    Distance may be interchanged with similarity, but distance is used for simplicity.    
    
    Parameters
    ----------
    dkeeper : `DistanceKeeper`
        The object to extract the distance from.
    label : `str`
        The identifier of the distance matrix to be extracted from ``dkeeper``.
    """

    def __init__(self, dkeeper=None, label=None):
        if label not in dkeeper.data:
            raise KeyError(f"No {label} in dkeeper.")

        self._label = label
        self._data = dkeeper.data[label]
        self._observation_labels = dkeeper.observation_labels        
        self._num_observations = dkeeper.num_observations


    @property
    def label(self):
        """ The distance label. """
        return self._label

    
    @property
    def data(self):
        """ The distance matrix. """
        return self._data


    @property
    def observation_labels(self):
        """ Labels for each observation. """
        return self._observation_labels


    @property
    def num_observations(self):
        """ Number of observations. """
        return self._num_observations


    def subset(self, observations_a, observations_b=None):
        """ Return subset of distances between ``observations_a`` and ``observations_b``.

        Parameters
        ----------
        observations_a : `list`
            Subset of observations to extract distances from, that make up the rows of
            the returned sub-distance matrix.
            This is treated differently depending on the type of observation labels :

            - If ``self._observation_labels`` is `List` [`str`], ``observations_a`` can
              be of the form:
        
              * `List` [`str`], to reference observations by their `str` label or;
              * `List` [`int`], to reference observations by their location index.

            - If ``self._observation_labels`` is `None` or `List` [`int`], ``observations``
              must be of the form `List` [`int`], to reference observations by their
              location index.
        observations_b : {`None`, `list`}, optional
            Subset of observations to extract distances computed from ``observations_a``.
            The list type depends on the type of ``self._feature_labels``, analogous to
            ``self._observation_labels`` for ``observations_a`` above.
            If `None`, ``observations_a`` is used to create symmetric matrix.

        Returns
        -------
        distance : `pandas.DataFrame`, (``len(observations_a)``, ``len(observations_b)``)
            The sub-matrix of distances where the :math:`ij^{th}` entry is the distance
            between the :math:`i^{th}` observation in ``observations_a`` and the
            :math:`j^{th}` observation in ``observations_b``.
        """

        distance = self._data

        if isinstance(observations_a[0], str):  # convert to location index
            observations_a_ix = [self._observation_labels.index(k) for k in observations_a]
        else:
            observations_a_ix = observations_a

        if observations_b is None:
            observations_b = observations_a
            observations_b_ix = observations_a_ix
        elif isinstance(observations_b[0], str):  # convert to location index
            observations_b_ix = [self._observation_labels.index(k) for k in observations_b]
        else:
            observations_b_ix = observations_b

        distance = distance[np.ix_(observations_a_ix, observations_b_ix)]
        return pd.DataFrame(data=distance, index=observations_a, columns=observations_b)


    def observation_index(self, observation_label):
        """ Return index of observation.

        Parameters
        ----------
        observation_label : `str`
            The observation label.

        Returns
        -------
        observation_index : `int`
            Index of observation in list of observation labels.
        """
        observation_index = self._observation_labels.index(observation_label)
        return observation_index


    def to_frame(self):
        """ Return data as a pandas DataFrame """
        df = pd.DataFrame(data=self.data,
                          columns=self.observation_labels,
                          index=self.observation_labels)
        return df


    def density(self):
        """ Return density of each observation

        The density of an observation is its net distance to all other
        observations. This should be minimized for distances and maximized
        for similarities.
        
        Returns
        -------
        d : `pandas.Series`
            The densities indexed by the observations.
        """
        d = self.to_frame().sum(axis=0)
        return d
            
            
        
        

class GraphKeeper:
    """ A class to store and handle multiple netowrks.

    Parameters
    ----------
    graphs : {`networkx.Graph`, `dict` [`str', `networkx.Graph`]}
        One or multiple networks.

        The network(s) may be provided in multiple ways:

        - `networkx.Graph` : A single network.
          
          * The network is placed in a `dict` with default label of the form
            ``{'graph' : graphs}``.
          * To use a customized label for the network, provide the network
            as a `dict`, shown next.
        - `dict` [`str`, `networkx.Graph`] : A single or multiple networks may
          be provided as the value(s) of a `dict` keyed by a `str` that serves as
          the netwwork descriptor or label for each input of the form
          ``{'graph_label' : `networkx.Graph`}``.


    Notes
    -----
    The graph label is also set to the graph's name.
    """

    def __init__(self, graphs=None):
        self._graphs = {}

        if isinstance(graphs, nx.Graph):
            self.add_graph(graphs, 'graph')
        elif isinstance(graphs, dict):
            for label, cur_graph in graphs.items():
                self.add_graph(cur_graph, label)
        elif graphs is None:
            pass
        else:
            raise TypeError("Unrecognized type for graphs, must be one of [networkx.Graph or dict].")


    def __getitem__(self, key):
        return self._graphs[key]


    def __contains__(self, key):
        return key in self._graphs


    def __iter__(self):
        for key, graph in self._graphs.items():
            yield graph


    @property
    def graphs(self):
        """ A dictionary of all the graphs. """
        return self._graphs


    def add_graph(self, graph, label):
        """ Add a network to the keeper.

        Parameters
        ----------
        graph : `networkx.Graph`
            The network.
        label : `str`
            Reference label describing the network.
        """
        # Check that the label is not already in the keeper
        if label in self._graphs:
            raise KeyError(f"Dupblicate label detected, {label} already exists in the keeper.")

        # Check that the graph is a networkx.Graph instance
        if not isinstance(graph, nx.Graph):
            raise ValueError("Unrecognized type, graph must be networkx Graph.")

        graph.name = label

        self._graphs[label] = graph

    
class Keeper:
    """ A class to store and handle data, distances and similarities between data points
    (or observations), and miscellaneous related results.

    Parameters
    ----------
    data : {`numpy.ndarray`, `pandas.DataFrame`, `dict` [`str`, `numpy.ndarray`], `dict` [`str`, `pandas.DataFrame`]}
        One or multiple feature data sets, where each data set is size (num_features, num_observations).

        Feature data set(s) may be provided in multiple ways:

        - `numpy.ndarray` : A single feature data set.

          * Observation labels may be specified in ``observation_labels``.
          * To include feature labels, ``data`` should be a `pandas.DataFrame`.
          * The array ``data`` is placed in a `dict` with default label of the form ``{'data' : data}``. 
        - `pandas.DataFrame` : A single feature data set.
    
          * The index indicates feature labels.
          * Columns indicate observation labels.
          * If ``observation_labels`` is provided, it should match the columns of the dataframe,
            that will be ordered according to the ``observation_labels``.
          * The array ``data.values`` is placed in a `dict` with default label of the form ``{'data' : data.values}``. 
        - `dict` [`str`, `numpy.ndarray`] : A single or mutliple feature data set(s) may be provided as the value(s) 
           of a `dict` keyed by a `str` that serves as the feature data descriptor or label for each
           input of the form ``{'data_label' : `numpy.ndarray`}``.

          * All arrays are expected to have the same number of columns corresponding to the number
            of observations with the same ordering.
          * Observation labels may be specified in ``observation_labels``.
          * To include feature labels, pass a `dict` of `pandas.DataFrame`\s instaed.
        - `dict` [`str`, `pandas.DataFrame`] : A single or mutliple feature data set(s) may be provided as the value(s) 
           of a `dict` keyed by a `str` that serves as the feature data descriptor or label for each
           input of the form ``{'data_label' : `pandas.DataFrame`}``.

          * All dataframes are expected to have the same number of columns corresponding to the number
            of observations with the same columns provided in the same order.
          * The index of each input indicates feature labels.
          * Columns indicate observation labels.
          * If ``observation_labels`` is provided, it should match the columns of the dataframe(s),
            that will be ordered according to the ``observation_labels``.
    distances : {`numpy.ndarray`, `pandas.DataFrame`, `dict` [`str`, `numpy.ndarray`], `dict` [`str`, `pandas.DataFrame`]}
        One or multiple symmetric distance(s) between observations, where each distance matrix is size
        (num_observations, num_observations).

        Distance(s) may be provided in multiple ways:

        - `numpy.ndarray` : A single distance matrix.

          * Observation labels may be specified in ``observation_labels``.
          * To include feature labels, ``distances`` should be a `pandas.DataFrame`.
          * The array ``distances`` is placed in a `dict` with default label of the form
            ``{'distance' : distances}``. 
        - `pandas.DataFrame` : A single distance matrix.
    
          * The index should be the same as the columns, which indicate observation labels.
          * If ``observation_labels`` is provided, it should match the rows and columns of the dataframe,
            that will be ordered according to the ``observation_labels``.
          * The array ``distances.values`` is placed in a `dict` with default label of the form
            ``{'distance' : distances.values}``. 
        - `dict` [`str`, `numpy.ndarray`] : A single or mutliple distance(s) may be provided as the value(s) 
           of a `dict` keyed by a `str` that serves as the distance descriptor or label for each
           input of the form ``{'distance_label' : `numpy.ndarray`}``.

          * All arrays are expected to have the same number of columns and rows corresponding to the number
            of observations with the same ordering.
          * Observation labels may be specified in ``observation_labels``.
        - `dict` [`str`, `pandas.DataFrame`] : A single or mutliple distance(s) may be provided as the value(s) 
           of a `dict` keyed by a `str` that serves as the distance descriptor or label for each
           input of the form ``{'distance_label' : `pandas.DataFrame`}``.

          * All dataframes are expected to have the same number of rows and columns corresponding 
            to the number of observations with the same columns provided in the same order.
          * The index and columns indicate observation labels.
          * If ``observation_labels`` is provided, it should match the columns of the dataframe(s),
            that will be ordered according to the ``observation_labels``.
    similarities : {`numpy.ndarray`, `pandas.DataFrame`, `dict` [`str`, `numpy.ndarray`], `dict` [`str`, `pandas.DataFrame`]}
        One or multiple symmetric similarit(y/ies) between observations, where each similarity
        matrix is size (num_observations, num_observations).
        Similarit(y/ies) may be provided in multiple ways, analogous to ``distances``.
    graphs : : {`networkx.Graph`, `dict` [`str', `networkx.Graph`]}
        One or multiple networks.

        The network(s) may be provided in multiple ways:

        - `networkx.Graph` : A single network.
          
          * The network is placed in a `dict` with default label of the form
            ``{'graph' : graphs}``.
          * To use a customized label for the network, provide the network
            as a `dict`, shown next.
        - `dict` [`str`, `networkx.Graph`] : A single or multiple networks may
          be provided as the value(s) of a `dict` keyed by a `str` that serves as
          the netwwork descriptor or label for each input of the form
          ``{'graph_label' : `networkx.Graph`}``.
    misc : `dict`
        Miscellaneous data or results.
    observation_labels : `list` [`str`], optional
        List of labels for each observation with length equal to num_observations.

        Labels will be set depending on the input format accordingly :
    
        - If ``observation_labels`` is provided and ``data``, ``distances``, or ``similarities``  is a :

          * `pandas.DataFrame` or `dict` [`str`, `pandas.DataFrame`], then ``observation_labels`` should 
            match the columns of the dataframe(s), which will be ordered according to ``observation_labels``.
          * `numpy.ndarray` or `dict` [`str`, `numpy.ndarray`], then the array(s) is (are) assumed to be
            ordered according to ``observation_labels``.
    
        - If ``observation_labels`` is not provided and ``data``, ``distances``, or ``similarities``  is a :

          * `pandas.DataFrame` or `dict` [`str`, `pandas.DataFrame`], then all dataframes are expected to
            have the same column names, which is used as the ``observation_labels``.
          * `numpy.ndarray` or `dict` [`str`, `numpy.ndarray`], then the array(s) is (are) assumed to have
            columns corresponding to the observations in the same order. Default values are used for
            ``observation_labels`` of the form 'X0', 'X1', ... and so on.
    outdir : {`None`, `str` `pathlib.Path`}
        Global path where any results will be saved. If `None`, no results will be saved.

    Notes
    -----
    All data sets, distances and similarities are assumed to contain the same set of data points in the same order.

    Subsets of a data set, distance or similarity should be stored in ``Keeper.misc`` as a `pandas.DataFrame`
    to keep track of the subset of observations (and features).
        
    """
    
    def __init__(self, data=None, distances=None, similarities=None,
                 graphs=None, misc=None, observation_labels=None,
                 outdir=None, verbose=None):

        if verbose is not None:
            set_verbose(logger, verbose)

        # if (data is None) and (distances is None) and (similarities is None):
        #     raise ValueError("At least one of data, distances, or similarities must be provided.")
        
        # check if observation labels given in data, distances or similarities:
        if observation_labels is None:
            num_observations = None
            tmp = DataKeeper(data=data, observation_labels=observation_labels)
            observation_labels = tmp.observation_labels
            if tmp.num_observations is not None:
                num_observations = tmp.num_observations
            # if observation_labels are still None, try distances:
            if observation_labels is None:
                tmp = DistanceKeeper(data=distances,
                                     observation_labels=observation_labels)
                observation_labels = tmp.observation_labels
                if (num_observations is None) and (tmp.num_observations is not None):
                    num_observations = tmp.num_observations
                # if observation_labels are still None, try similarities:
                if observation_labels is None:
                    tmp = DistanceKeeper(data=similarities,
                                         observation_labels=observation_labels)
                    observation_labels = tmp.observation_labels
                    if (num_observations is None) and (tmp.num_observations is not None):
                        num_observations = tmp.num_observations

        else:
            num_observations = len(observation_labels)
                    
                    
        # if no observation labels have been found at this point, set to default
        if (observation_labels is None) and (num_observations is not None):
            observation_labels = [f"X{i}" for i in range(num_observations)]

        if (observation_labels is not None) and (num_observations is not None):
            if len(observation_labels) != len(set(observation_labels)):
                raise ValueError("Observation labels must be unique")
            assert len(observation_labels) == num_observations, \
                "Inconsistent number of observation labels, length of observation_labels must equal num_observations"
            
        self._observation_labels = observation_labels
        # self._num_observations = None if self._observation_labels is None else len(self._observation_labels)
        self._num_observations = num_observations # len(self._observation_labels)

        self._data = DataKeeper(data=data, observation_labels=self._observation_labels)
        self._distances = DistanceKeeper(data=distances,
                                         observation_labels=self._observation_labels)
        self._similarities = DistanceKeeper(data=similarities,
                                            observation_labels=self._observation_labels)
        self._graphs = GraphKeeper(graphs=graphs)
        self._misc = misc if misc is not None else {}

        self._check_num_observations()
        self._check_observation_labels()

        if isinstance(outdir, str):
            outdir = Path(outdir)
        self.outdir = outdir
        if isinstance(self.outdir, Path) and not self.outdir.is_dir():
            self.outdir.mkdir()
            logger.msg(f"Created directory {self.outdir}.")


    def _check_num_observations(self):
        """ Check that num_observation are consistent (or None) across keepers. """
        num_cur = self._num_observations
        num_data = self._data.num_observations
        num_dist = self._distances.num_observations
        num_sim = self._similarities.num_observations

        record = []
        for dl in [num_cur, num_data, num_dist, num_sim]:
            if dl is not None:
                record.append(dl)
            else:
                record.append('None')

        while len(record) > 1:
            dl = record.pop()
            for dl2 in record:
                if dl != dl2:
                    raise ValueError("Missmatched number of observation detected between keepers.")
                

    def _check_observation_labels(self):
        """ Check that observation_labels are consistent (or None) across keepers. """
        cur_labels = self._observation_labels
        data_labels = self._data.observation_labels
        dist_labels = self._distances.observation_labels
        sim_labels = self._similarities.observation_labels

        labels = []
        for dl in [cur_labels, data_labels, dist_labels, sim_labels]:
            if dl is not None:
                labels.append(dl)
            else:
                labels.append('None')

        while len(labels) > 1:
            dl = labels.pop()
            for dl2 in labels:
                if dl != dl2:
                    raise ValueError("Missmatched observation labels detected between keepers.")

                
    # def _update_observation_labels(self):
    #     """ Update observation labels (and number of observations) if changed in keeper. """
    #     # first check that num_observations and observation labels match (or are None) across sub-keepers:
    #     self.check_num_observations()
    #     self._check_observation_labels()

    #     record = [None]
    #     for dl in [self._data, self._distances, self._similarities]:
    #         if dl.observation_labels is not None:
    #             record.append(dl.observation_labels)
    #             # num_labels.append(dl.num_observations)
    #             break  # already checked that labels are eqiovalent, so no need to keep looking

    #     labels = record[-1]
    #     # num_labels = num_labels[-1]

    #     if self._observation_labels is not None and labels is not None:
            

    #     if self._observation_labels is None:
    #         if labels is not None:
    #             if self._num_observations is None:
                    
    #             if (self._num_observations is not None) and (len(labels) != self_num_observations):
    #                 raise ValueError("Number of observations in sub-keepers does not match global keeper.")
    #             self._observation_labels = labels.copy()
    #             # check that all
    #     else:
    #         if (labels is not None) and (self._observation_labels != labels):
    #             raise ValueError("Inconsisstent observation labels detected between keepers.")
            
            
                  
    @property
    def observation_labels(self):
        """ Labels for each observation. """
        self._check_observation_labels()
        return self._observation_labels
    

    @property
    def data(self):
        """ The feature data sets. """
        return self._data


    @property
    def distances(self):
        """ The distances. """
        return self._distances


    @property
    def similarities(self):
        """ The similarities. """
        return self._similarities


    @property
    def graphs(self):
        """ The networks. """
        return self._graphs
    

    @property
    def misc(self):
        """ The misc data. """
        return self._misc

    
    @property
    def num_observations(self):
        """ The number of observations. """
        self._check_num_observations()
        
        # num_obs = self._data.num_observations
        # if num_obs is not None:
        #     return num_obs

        # num_obs = self._distances.num_observations
        # if num_obs is not None:
        #     return num_obs

        # num_obs = self._similarities.num_observations
        # if num_obs is not None:
        #     return num_obs

        # return None
        return self._num_observations
        
        
    def add_data(self, data, label):
        """ Add a feature data set to the data keeper.

        Parameters
        ----------
        data : {`numpy.ndarray`, `pandas.DataFrame`}
            The data set of size (num_features, num_observations).
        label : `str`
            Reference label describing the data set.        
        """
        self._data.add_data(data, label)
        logger.info(f"Added data input {label} to the keeper.")

        # If not yet initialized, update keeper observation_labels and num_observations
        if self._num_observations is None:
            self._num_observations = self._data[label].num_observations
            self._observation_labels = self._data[label].observation_labels            
            # if self._data[label]._observation_labels is None:
            if self._data._observation_labels is None:
                self._observation_labels = self._data._observation_labels = [f"X{i}" for i in range(self._num_observations)]

            # initialize distances and similarities:
            self._distances = DistanceKeeper(observation_labels=self._observation_labels)
            self._similarities = DistanceKeeper(observation_labels=self._observation_labels)

        self._check_observation_labels()
        self._check_num_observations()


    def add_distance(self, data, label):
        """ Add a distance array to the distance keeper.

        Parameters
        ----------
        data : {`numpy.ndarray`, `pandas.DataFrame`}
            The distance array of size (num_observations, num_observations).
        label : `str`
            Reference label describing the distance.
        """
        self._distances.add_data(data, label)
        logger.info(f"Added distance input {label} to the keeper.")

        # If not yet initialized, update keeper observation_labels and num_observations
        if self._num_observations is None:
            self._num_observations = self._distances[label].num_observations
            self._observation_labels = self._distances[label].observation_labels            
            if self._distances[label]._observation_labels is None:
                self._observation_labels = [f"X{i}" for i in range(self._num_observations)]

            # initialize data and similarities
            self._data = DataKeeper(observation_labels=self._observation_labels)
            self._similarities = DistanceKeeper(observation_labels=self._observation_labels)

        self._check_observation_labels()
        self._check_num_observations()


    def add_stacked_distance(self, data, label):
        """ Add a stacked distance array to the distance keeper.

        Parameters
        ----------
        data : `pandas.Series`
            The stacked distances of size (num_observations * (num_observations - 1) / 2,)
            with a 2-multi-index of the pairwise observation labels.
        label : `str`
            Reference label describing the distance.
        """
        self._distances.add_stacked_data(data, label)
        logger.info(f"Added distance input {label} to the keeper.")

        # If not yet initialized, update keeper observation_labels and num_observations
        if self._num_observations is None:
            self._num_observations = self._distances[label].num_observations
            self._observation_labels = self._distances[label].observation_labels            
            if self._distances[label]._observation_labels is None:
                self._observation_labels = [f"X{i}" for i in range(self._num_observations)]

            # initialize data and similarities
            self._data = DataKeeper(observation_labels=self._observation_labels)
            self._similarities = DistanceKeeper(observation_labels=self._observation_labels)

        self._check_observation_labels()
        self._check_num_observations()


    def add_similarity(self, data, label):
        """ Add a similarity array to the similarity keeper.

        Parameters
        ----------
        data : {`numpy.ndarray`, `pandas.DataFrame`}
            The similarity array of size (num_observations, num_observations).
        label : `str`
            Reference label describing the similarity.
        """
        self._similarities.add_data(data, label)
        logger.info(f"Added similarity input {label} to the keeper.")

        # If not yet initialized, update keeper observation_labels and num_observations
        if self._num_observations is None:
            self._num_observations = self._similarities[label].num_observations
            self._observation_labels = self._similarities[label].observation_labels            
            if self._similarities[label]._observation_labels is None:
                self._observation_labels = [f"X{i}" for i in range(self._num_observations)]

            # initialize data and distances
            self._data = DataKeeper(observation_labels=self._observation_labels)
            self._distances = DistanceKeeper(observation_labels=self._observation_labels)

        self._check_observation_labels()
        self._check_num_observations()
    
        
    def add_stacked_similarity(self, data, label, diag=1.):
        """ Add a stacked similarity array to the similarity keeper.

        Parameters
        ----------
        data : `pandas.Series`
            The stacked similarities of size (num_observations * (num_observations - 1) / 2,)
            with a 2-multi-index of the pairwise observation labels.
        label : `str`
            Reference label describing the similarity.
        diag : `float`
            Value used on diagonal. 
        """
        self._similarities.add_stacked_data(data, label, diag=diag)
        logger.info(f"Added similarity input {label} to the keeper.")

        # If not yet initialized, update keeper observation_labels and num_observations
        if self._num_observations is None:
            self._num_observations = self._similarities[label].num_observations
            self._observation_labels = self._similarities[label].observation_labels            
            if self._similarities[label]._observation_labels is None:
                self._observation_labels = [f"X{i}" for i in range(self._num_observations)]

            # initialize data and distances
            self._data = DataKeeper(observation_labels=self._observation_labels)
            self._distances = DistanceKeeper(observation_labels=self._observation_labels)

        self._check_observation_labels()
        self._check_num_observations()


    def add_graph(self, graph, label):
        """ Add a network to the graph keeper.

        Parameters
        ----------
        graph : `networkx.Graph`
            The network to add.
        label : `str`
            Reference label describing the network.
        """
        self._graphs.add_graph(graph, label)
        logger.info(f"Added graph input {label} to the keeper.")

        
    def add_misc(self, data, label):
        """ Add misc information to be stored.

        Parameters
        ----------
        data 
            The misc information, e.g., a graph.
        label : `str`
            Reference label describing the input.
        """
        if label in self._misc:
            raise KeyError(f"Dubplicate label detected, {label} already exists in the keeper.")

        self._misc[label] = data

        logger.info(f"Added misc input {label} to the keeper.")
    

    def load_data(self, file_name, label='data', file_path=None, file_format=None,                  
                  delimiter=',', dtype=None, **kwargs):
        """ Load data from file into the keeper.

        .. Note::

        Currently loads data using ``pandas.read_csv``.
        Additional formats will be added in the future.
                  
        Parameters
        ----------
        file_name: {`str`, `pathlib.Path`} 
            Input data file name.
        label : `str`, (default: 'data')
            Reference label describing the data set.        
        file_path: {`str` `pathlib.Path`}, optional (default: None)
            File path. Empty string by default
        file_format: `str`, optional (default: None)
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
            If `None`, ``file_format`` will be inferred from the file extension
            in ``file_name``.
            Currently, this is ignored.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        dtype
            If provided, ensure to convert data type after loaded. 
        **kwargs
            Additional key-word arguments passed to ``pandas.read_csv``.
        
        """
        data = load_from_file(file_name, file_path=file_path, file_format=file_format,
                              delimiter=delimiter, **kwargs)
        if dtype is not None:
            data = data.astype(dtype)
        self.add_data(data, label)


    def load_distance(self, file_name, label='distance', file_path=None, file_format=None,
                      delimiter=',', **kwargs):
        """ Load distance from file into the keeper. 

        .. Note::

        Assumed that the distance array is stored with the first row and first column as
        the index and header, respectively.
        
        Currently loads data using ``pandas.read_csv``.
        Additional formats will be added in the future.
                  
        Parameters
        ----------
        file_name: {`str`, `pathlib.Path`} 
            Input distance file name.
        label : `str`, (default: 'distance')
            Reference label describing the data set.        
        file_path: {`str` `pathlib.Path`}, optional (default: None)
            File path. Empty string by default
        file_format: `str`, optional (default: None)
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
            If `None`, ``file_format`` will be inferred from the file extension
            in ``file_name``.
            Currently, this is ignored.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        **kwargs
            Additional key-word arguments passed to ``pandas.read_csv``.
        """        
        data = load_from_file(file_name, file_path=file_path, file_format=file_format,
                              delimiter=delimiter, header=0, index_col=0, **kwargs)
        self.add_distance(data, label)

        
    def load_stacked_distance(self, file_name, label='distance', file_path=None, file_format=None,
                              delimiter=',', **kwargs):
        """ Load distance in stacked form from file, convert to unstacked form and store in the keeper. 

        .. Note::

        Assumed that the stacked distances are stored with a 2-multi-index of the pairwise-observattion
        (excluding self-pairs) and a single column with the pairwise distances.
        
        Currently loads data using ``pandas.read_csv``.
        Additional formats will be added in the future.
                  
        Parameters
        ----------
        file_name: {`str`, `pathlib.Path`} 
            Input distance file name.
        label : `str`, (default: 'distance')
            Reference label describing the data set.        
        file_path: {`str` `pathlib.Path`}, optional (default: None)
            File path. Empty string by default
        file_format: `str`, optional (default: None)
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
            If `None`, ``file_format`` will be inferred from the file extension
            in ``file_name``.
            Currently, this is ignored.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        **kwargs
            Additional key-word arguments passed to ``pandas.read_csv``.
        """        
        data = load_from_file(file_name, file_path=file_path, file_format=file_format,
                              delimiter=delimiter, header=0, index_col=(0, 1), **kwargs)
        self._distances.add_stacked_data(data, label)


    def load_similarity(self, file_name, label='similarity', file_path=None, file_format=None,
                        delimiter=',', **kwargs):
        """ Load similarity from file into the keeper. 

        .. Note::

        Assumed that the distance array is stored with the first row and first column as
        the index and header, respectively.
        
        Currently loads data using ``pandas.read_csv``.
        Additional formats will be added in the future.
                  
        Parameters
        ----------
        file_name: {`str`, `pathlib.Path`} 
            Input similarity file name.
        label : `str`, (default: 'similarity')
            Reference label describing the data set.        
        file_path: {`str` `pathlib.Path`}, optional (default: None)
            File path. Empty string by default
        file_format: `str`, optional (default: None)
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
            If `None`, ``file_format`` will be inferred from the file extension
            in ``file_name``.
            Currently, this is ignored.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        **kwargs
            Additional key-word arguments passed to ``pandas.read_csv``.
        """        
        data = load_from_file(file_name, file_path=file_path, file_format=file_format,
                              delimiter=delimiter, header=0, index_col=0, **kwargs)
        self.add_similarity(data, label)


    def load_stacked_similarity(self, file_name, label='similarity', diag=1.,
                                file_path=None, file_format=None,
                                delimiter=',', **kwargs):
        """ Load similarity in stacked form from file, convert to unstacked form and store in the keeper. 

        .. Note::

        Assumed that the stacked distances are stored with a 2-multi-index of the pairwise-observattion
        (excluding self-pairs) and a single column with the pairwise distances.
        
        Currently loads data using ``pandas.read_csv``.
        Additional formats will be added in the future.
                  
        Parameters
        ----------
        file_name: {`str`, `pathlib.Path`} 
            Input distance file name.
        label : `str`, (default: 'distance')
            Reference label describing the data set.
        diag : `float`
            Value used on diagonal. 
        file_path: {`str` `pathlib.Path`}, optional (default: None)
            File path. Empty string by default
        file_format: `str`, optional (default: None)
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
            If `None`, ``file_format`` will be inferred from the file extension
            in ``file_name``.
            Currently, this is ignored.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        **kwargs
            Additional key-word arguments passed to ``pandas.read_csv``.
        """        
        data = load_from_file(file_name, file_path=file_path, file_format=file_format,
                              delimiter=delimiter, header=0, index_col=(0, 1), **kwargs)
        self._similarities.add_stacked_data(data, label)


    def load_graph(self, file_name, label='graph', file_path=None,
                   file_format=None, delimiter=',',
                   source='source', target='target', **kwargs):
        """ Load network (edgelist) from file into graph and store in the keeper. 

        .. Note::

        Currently loads graph from edgelist. Future release will allow different
        graph types (e.g., adjacency, graphml).
        
        Assumed that the edge-list is stored as two columns, where the first row
        is labeled as source and target.
        
        Currently loads data using ``pandas.read_csv``.
        Additional formats will be added in the future.
                  
        Parameters
        ----------
        file_name: {`str`, `pathlib.Path`} 
            Input edge-list file name.
        label : `str`, (default: 'graph')
            Reference label describing the network.
        file_path: {`str` `pathlib.Path`}, optional (default: None)
            File path. Empty string by default
        file_format: `str`, optional (default: None)
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
            If `None`, ``file_format`` will be inferred from the file extension
            in ``file_name``.
            Currently, this is ignored.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        source : {`str`, `int`} (default: 'source')
            A valid column name (string or integer) for the source nodes
            passed to ``networkx.from_pandas_edgelist``.
        target : {`str`, `int`} (default: 'target')
            A valid column name (string or integer) for the target nodes
            passed to ``networkx.from_pandas_edgelist``.
        **kwargs
            Additional key-word arguments passed to ``pandas.read_csv``.
        """        
        E = load_from_file(file_name, file_path=file_path, file_format=file_format,
                           delimiter=delimiter, header=0, index_col=None, **kwargs)
        G = nx.from_pandas_edgelist(E, source=source, target=target)
        self.add_graph(G, label)


    def save_data(self, label, file_format='txt', delimiter=',', **kwargs):
        """ Save data to file. 

        .. Note::

        This currently only saves a pandas DataFrame to .txt, .csv, or .tsv.
        Future releases will allow for other formats.

        Data set is saved to the file named '{self.outdir}/data_{label}.{file_format}'.

        Parameters
        ----------
        label : `str`
            Reference label describing which data set to save.
        file_format : `str`
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        **kwargs
            Additional key-word arguments passed to ``pandas.to_csv``.
        """
        if self.outdir is None:
            raise ValueError("Cannot store data - outdir was not set.")

        _fp = self.outdir / f"data_{label}.{file_format}"
        if _fp.is_file():
            raise ValueError(f"File already exists, cannot save data to {_fp}.")

        df = self.data[label].to_frame()
        df.to_csv(_fp, header=True, index=True, sep=delimiter, **kwargs)
        logger.msg(f"Data set saved to {df}.")


    def save_distance(self, label, file_format='txt', delimiter=',', **kwargs):
        """ Save distance to file. 

        .. Note::

        This currently only saves a pandas DataFrame to .txt, .csv, or .tsv.
        Future releases will allow for other formats.

        Distance is saved to the file named '{self.outdir}/distance_{label}.{file_format}'.

        Parameters
        ----------
        label : `str`
            Reference label describing which distance to save.
        file_format : `str`
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        **kwargs
            Additional key-word arguments passed to ``pandas.to_csv``.
        """
        if self.outdir is None:
            raise ValueError("Cannot store distance - outdir was not set.")

        _fp = self.outdir / f"distance_{label}.{file_format}"
        if _fp.is_file():
            raise ValueError(f"File already exists, cannot save distance to {_fp}.")

        df = self.distances[label].to_frame()
        df.to_csv(_fp, header=True, index=True, sep=delimiter, **kwargs)
        logger.msg(f"Distance set saved to {_fp}.")


    def save_similarity(self, label, file_format='txt', delimiter=',', **kwargs):
        """ Save similarity to file. 

        .. Note::

        This currently only saves a pandas DataFrame to .txt, .csv, or .tsv.
        Future releases will allow for other formats.

        Similarity is saved to the file named '{self.outdir}/similarity_{label}.{file_format}'.

        Parameters
        ----------
        label : `str`
            Reference label describing which similarity to save.
        file_format : `str`
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        **kwargs
            Additional key-word arguments passed to ``pandas.to_csv``.
        """
        if self.outdir is None:
            raise ValueError("Cannot store similarity - outdir was not set.")

        _fp = self.outdir / f"similarity_{label}.{file_format}"
        if _fp.is_file():
            raise ValueError(f"File already exists, cannot save similarity to {_fp}.")

        df = self.similarities[label].to_frame()
        df.to_csv(_fp, header=True, index=True, sep=delimiter, **kwargs)
        logger.msg(f"Similarities set saved to {_fp}.")


    def save_misc(self, label, file_format='txt', delimiter=',', **kwargs):
        """ Save misc data to file. 

        .. Note::

        This currently only saves a pandas DataFrame to .txt, .csv, or .tsv.
        Future releases will allow for other formats.

        Misc data is saved to the file named '{self.outdir}/misc_{label}.{file_format}'.

        Parameters
        ----------
        label : `str`
            Reference label describing which misc data to save.
        file_format : `str`
            File format. Currently supported file formats: 'txt', 'csv', 'tsv'.
        delimiter: `str`, optional (default: ',')
            Delimiter to use.
        **kwargs
            Additional key-word arguments passed to ``pandas.to_csv``.
        """
        if self.outdir is None:
            raise ValueError("Cannot store misc data - outdir was not set.")

        _fp = self.outdir / f"misc_{label}.{file_format}"
        if _fp.is_file():
            raise ValueError(f"File already exists, cannot save misc data to {_fp}.")

        df = self.misc[label]
        df.to_csv(_fp, sep=delimiter, **kwargs)
        logger.msg(f"Misc data set saved to {_fp}.")


    def observation_index(self, observation_label):
        """ Return index of observation.

        Parameters
        ----------
        observation_label : `str`
            The observation label.

        Returns
        -------
        observation_index : `int`
            Index of observation in list of observation labels.
        """
        observation_index = self._observation_labels.index(observation_label)
        return observation_index


    def subset(self, observations, keep_misc=False, keep_graphs=False, outdir=None):
        """ Return a new instance of Keeper restricted to subset of observations.

        The default behavior is to not include misc or graphs in the Keeper subset.
        This is because there is no check for which observations the misc and graphs
        correspond to.

        Warning: The subset keeper and all data it contains is not a copy.

        Parameters
        ----------
        observations : `list`
            List of observations to include in the subset.
            This is treated differently depending on the type of observation labels :

            - If ``self._observation_labels`` is `List` [`str`], ``observations`` can be of the form:
        
              * `List` [`str`], to reference observations by their `str` label or;
              * `List` [`int`], to reference observations by their location index.

            - If ``self._observation_labels`` is `None` or `List` [`int`], ``observations`` must 
              be of the form `List` [`int`], to reference observations by their location index.
        keep_misc : `bool`
            If True, misc is added to the new Keeper. 
        keep_graphs : `bool`
            If True, the graphs are added to the new Keeper.
        outdir : {`None`, `str` `pathlib.Path`}
            Global path where any results will be saved. If `None`, no results will be saved.
        
        Returns
        -------
        keeper_subset : `Keeper`
            A Keeper object restricted to the selected observations.
        """
        keeper_subset = Keeper(observation_labels=observations, outdir=outdir)
        keeper_subset._data = self.data.subset(observations)
        keeper_subset._distances = self.distances.subset(observations)
        keeper_subset._similarities = self.similarities.subset(observations)

        if keep_misc:
            keeper_subset._misc = self.misc

        if keep_graphs:
            keeper_subset._graphs = self._graphs

        keeper_subset._check_num_observations()
        keeper_subset._check_observation_labels()

        return keeper_subset


    def distance_density(self, label):
        """ Compute each observation's density from a distance.

        The density of an observation is its net distance to all other observations.

        Parameters
        ----------
        label : `str`
            The reference label for the distance.

        Returns
        -------
        density : `pandas.Series`
            The densities indexed by the observation labels.
        """
        density = self.distances[label].density()
        return density


    def distance_density_argmin(self, label):
        """ Find the observation with the largest density from a distnace.

        The density of an observation is its net distance to all other observations.

        Parameters
        ----------
        label : `str`
            The reference label for the distance.

        Returns
        -------
        obs : `int`
            The index of the observation with the smallest density.
        """
        density = self.distances[label].density()
        obs = density.idxmin(axis=0)
        obs = self.observation_labels.index(obs)
        return obs


    def similarity_density(self, label):
        """ Compute each observation's density from a similarity.

        The density of an observation is its net similarity to all other observations.

        Parameters
        ----------
        label : `str`
            The reference label for the similarity.

        Returns
        -------
        density : `pandas.Series`
            The densities indexed by the observation labels.
        """
        density = self.similarities[label].density()
        return density


    def similarity_density_argmax(self, label):
        """ Find the observation with the largest density from a similarity.

        The density of an observation is its net similarity to all other observations.

        Parameters
        ----------
        label : `str`
            The reference label for the similarity.

        Returns
        -------
        obs : `str`
            The label of the observation with the largest density.
        """
        density = self.similarities[label].density()
        obs = density.idxmax(axis=0)
        obs = self.observation_labels.index(obs)
        return obs


    def wass_distance_pairwise_observation_profile(self, data_key, graph_key,
                                                   features=None, label=None,
                                                   graph_distances=None, edge_weight=None,
                                                   proc=mp.cpu_count(), chunksize=None,
                                                   measure_cutoff=1e-6, solvr=None):
        """ Compute Wasserstein distances between feature profiles of every two observations

        .. note::

          If ``object.outdir`` is not `None`, Wasserstein distances are saved to file every 10 iterations.
          Before starting the computation, check if the file exists. If so, load and remove already computed
          nodes from the iteration. Wasserstein distances are computed for the remaining nodes, combined with
          the previously computed and saved results before saving and returning the combined results.

          Only nodes with at least 2 neighbors are included, as leaf nodes will all have the same Wassserstein
          distance and do not provide any further information.

          The resulting observation-pairwise Wasserstein distances are saved to the DistanceKeeper (aka self.distances)
          and can be accessed by ``self.distances[f'{data_key}_{label}_wass_dist_observation_pairwise_profiles']``.

        Parameters
        ----------
        data_key : `str`
            The key to the data in the data keeper that should be used.
        graph_key : 'str'
            The key to the graph in the graph keeper that should be used.
            (Does not have to include all features in the data)
        features : {`None`, `list` [`str`])}
            List of features to compute profile distances on. If `None`, all features are used.
        label : str
            Label that resulting Wasserstein distances are saved in ``keeper.distances`` and
            name of file to store stacked results.
        graph_distances : `numpy.ndarray`, (n, n)
            A matrix of node-pairwise graph distances between the :math:`n` nodes (ordered from :math:`0, 1, ..., n-1`).
            If `None`, use hop distance.
        edge_weight : {`None`, `str`}
            The edge attribute used as the weight for computed the graph distances. This is
            ignored if ``graph_distances`` is provided, If `None`, no edge weight is used.
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
            Wasserstein distances between pairwise profiles where rows are observation-pairs and columns are node names,
            saved in ``keeper.distances`` with the key ``f'{data_key}_{label}_wass_dist_observation_pairwise_profiles'``.
        """
        inet = InfoNet(self, graph_key, layer=data_key)

        if graph_distances is None:
            graph_distances = inet.compute_graph_distances(weight=edge_weight)

        if label is None:
            label = f'{data_key}_wass_dist_observation_pairwise_profiles'
        else:
            label = f'{data_key}_{label}_wass_dist_observation_pairwise_profiles'
            
        inet.pairwise_observation_profile_wass_distance(features=features, graph_distances=graph_distances, 
                                                        label=label,
                                                        desc='Computing pairwise profile Wasserstein distances',
                                                        proc=proc, chunksize=chunksize,
                                                        measure_cutoff=measure_cutoff, solvr=solvr)

        
    def euc_distance_pairwise_observation_profile(self, data_key, features=None, label=None,
                                                  metric='euclidean', normalize=False, **kwargs):
        """ Compute Euclidean distances between feature profiles of every two observations

        .. note::

          If ``object.outdir`` is not `None`, Euclidean distances are saved to file.
          Before starting the computation, check if the file exists. If so, load and remove already computed
          nodes from the iteration. Euclidean distances are computed for the remaining nodes, combined with
          the previously computed and saved results before saving and returning the combined results.

          Only nodes with at least 2 neighbors are included, as leaf nodes will all have the same Wassserstein
          distance and do not provide any further information.

          The resulting observation-pairwise Wasserstein distances are saved to the DistanceKeeper (aka self.distances)
          and can be accessed by ``self.distances[f'{data_key}_{label}_euc_dist_observation_pairwise_profiles']``.

        Parameters
        ----------
        data_key : `str`
            The key to the data in the data keeper that should be used.
        features : {`None`, `list` [`str`])}
            List of features to compute profile distances on. If `None`, all features are used.
        label : str
            Label that resulting Wasserstein distances are saved in ``keeper.distances`` and
            name of file to store stacked results.
        metric : `str`
            The metric used to compute the distance, passed to scipy.spatial.distance.cdist.
        normalize : `bool`
            If `True`, normalize neighborhood profiles to sum to 1.
        **kwargs : `dict`
            Extra arguments to metric, passed to `scipy.spatial.distance.cdist`.

        Returns
        -------
        eds : `pandas.DataFrame`
            Euclidean distances between pairwise profiles where rows are observation-pairs and columns are node names,
            saved in ``keeper.distances`` with the key ``f'{data_key}_{label}_euc_dist_observation_pairwise_profiles'``.

        """
        inet = InfoNet(self, None, layer=data_key)

        if label is None:
            label = f'{data_key}_euc_dist_observation_pairwise_profiles'
        else:
            label = f'{data_key}_{label}_euc_dist_observation_pairwise_profiles'

        inet.pairwise_observation_profile_euc_distance(features=features,
                                                       label=label, 
                                                       desc='Computing pairwise profile Euclidean distances',
                                                       metric=metric, normalize=normalize, **kwargs)


    def wass_distance_pairwise_observation_feature_nbhd(self, data_key, graph_key,
                                                        features=None, include_self=False, label=None,
                                                        graph_distances=None, edge_weight=None,
                                                        proc=mp.cpu_count(), chunksize=None,
                                                        measure_cutoff=1e-6, solvr=None):
        """ Compute Wasserstein distances between feature neighborhoods of every two observations

        .. note::

          If ``object.outdir`` is not `None`, Wasserstein distances are saved to file every 10 iterations.
          Before starting the computation, check if the file exists. If so, load and remove already computed
          nodes from the iteration. Wasserstein distances are computed for the remaining nodes, combined with
          the previously computed and saved results before saving and returning the combined results.

          Only nodes with at least 2 neighbors are included, as leaf nodes will all have the same Wassserstein
          distance and do not provide any further information.

          The resulting observation-pairwise Wasserstein distances are saved to misc  (aka self.misc) and can be accessed by
          ``self.misc[f"{data_key}_{label}_wass_dist_observation_pairwise_nbhds_with{'' if include_self else 'out'}_self"]``.

        Parameters
        ----------
        data_key : `str`
            The key to the data in the data keeper that should be used.
        graph_key : 'str'
            The key to the graph in the graph keeper that should be used.
            (Does not have to include all features in the data)
        features : {`None`, `list` [`str`])}
            List of features (nodes) to compute neighborhood distances on.
            If `None`, all features are used.
        include_self : `bool`
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution
            over the neighborhood.
        label : str
            Label that resulting Wasserstein distances are saved in ``keeper.misc`` 
            and name of file to store stacked results.
        graph_distances : `numpy.ndarray`, (n, n)
            A matrix of node-pairwise graph distances between the
            :math:`n` nodes (ordered from :math:`0, 1, ..., n-1`).
            If `None`, use hop distance.
        edge_weight : {`None`, `str`}
            The edge attribute used as the weight for computed the graph distances. This is
            ignored if ``graph_distances`` is provided, If `None`, no edge weight is used.
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
            Wasserstein distances between pairwise observations where rows are observation-pairs and columns are
            feature (node) names.
            saved in ``keeper.misc`` with the key
            ``f"{data_key}_{label}_wass_dist_observation_pairwise_nbhds_with{'' if include_self else 'out'}_self"``.
        """
        inet = InfoNet(self, graph_key, layer=data_key)

        if graph_distances is None:
            graph_distances = inet.compute_graph_distances(weight=edge_weight)

        if label is None:
            label = f"{data_key}_wass_dist_observation_pairwise_nbhds_with{'' if include_self else 'out'}_self"
        else:
            label = f"{data_key}_{label}_wass_dist_observation_pairwise_nbhds_with{'' if include_self else 'out'}_self"
            
        inet.multiple_pairwise_observation_neighborhood_wass_distance(nodes=features, include_self=include_self,
                                                                      graph_distances=graph_distances,
                                                                      label=label,
                                                                      desc='Computing pairwise 1-hop nbhd Wasserstein distances',
                                                                      proc=proc, chunksize=chunksize,
                                                                      measure_cutoff=measure_cutoff, solvr=solvr)


    def euc_distance_pairwise_observation_feature_nbhd(self, data_key, graph_key,
                                                        features=None, include_self=False, label=None,
                                                        metric='euclidean', normalize=False, **kwargs):
        """ Compute Euclidean distances between feature neighborhoods of every two observations

        .. note::

          If ``object.outdir`` is not `None`, Euclidean distances are saved to file every 10 iterations.
          Before starting the computation, check if the file exists. If so, load and remove already computed
          nodes from the iteration. Euclidean distances are computed for the remaining nodes, combined with
          the previously computed and saved results before saving and returning the combined results.

          Only nodes with at least 2 neighbors are included, as leaf nodes will all have the same Euclidean
          distance and do not provide any further information.

          The resulting observation-pairwise Euclidean distances are saved to misc (aka self.misc) and can be accessed by
          ``self.misc[f"{data_key}_{label}_euc_dist_observation_pairwise_nbhds_with{'' if include_self else 'out'}_self"]``.

        Parameters
        ----------
        data_key : `str`
            The key to the data in the data keeper that should be used.
        graph_key : 'str'
            The key to the graph in the graph keeper that should be used.
            (Does not have to include all features in the data)
        features : {`None`, `list` [`str`])}
            List of features (nodes) to compute neighborhood distances on.
            If `None`, all features are used.
        include_self : `bool`
            If `True`, add node in neighborhood which will result in computing normalized profile over the neighborhood.
            If `False`, node is not included in neighborhood which results in computing the transition distribution
            over the neighborhood.
        label : str
            Label that resulting Wasserstein distances are saved in ``keeper.misc`` 
            and name of file to store stacked results.
        metric : `str`
            The metric used to compute the distance, passed to scipy.spatial.distance.cdist.
        normalize : `bool`
            If `True`, normalize neighborhood profiles to sum to 1.
        **kwargs : `dict`
            Extra arguments to metric, passed to `scipy.spatial.distance.cdist`.

        Returns
        -------
        eds : `pandas.DataFrame`
            Euclidean distances between pairwise observations where rows are observation-pairs and columns are
            feature (node) names.
            saved in ``keeper.misc`` with the key
            ``f"{data_key}_{label}_euc_dist_observation_pairwise_nbhds_with{'' if include_self else 'out'}_self"``.
        """
        inet = InfoNet(self, graph_key, layer=data_key)

        if label is None:
            label = f"{data_key}_euc_dist_observation_pairwise_nbhds_with{'' if include_self else 'out'}_self"
        else:
            label = f"{data_key}_{label}_euc_dist_observation_pairwise_nbhds_with{'' if include_self else 'out'}_self"
            
        if normalize :
            label = '_'.join([label, 'normalized'])
            
        inet.multiple_pairwise_observation_neighborhood_euc_distance(nodes=features, include_self=include_self,
                                                                     label=label,
                                                                     desc='Computing pairwise 1-hop nbhd Euclidean distances',
                                                                     metric=metric, normalize=normalize, **kwargs)


    def compute_sigmas(self, distance_key, label=None, n_neighbors=None,
                       method='max', return_nn=False):
        """ Set sigma for each obs as the distance to its k-th neighbor from keeper.
    
        Parameters
        ----------
        distance_key : `str`
            The label used to reference the distance matrix stored in ``keeper.distances``,
            of size (n_observations, n_observations).
        label : {`None`, `str`}
            If provided, this is appended to the context tag
            (``tag = f"{method}{n_neighbors}nn_{distance_key}"``). The key used to store the
            results defaults to the tag when ``label`` is not provided: ``key = tag``. Otherwise,
            the key is set to: ``key = tag + "-" + label``. The resulting sigmas
            are stored in ``keeper.misc['sigmas_' + key]``.
            
            If ``return_nn`` is `True`,
            nearest neighbor indices are stored in ``keeper.misc['nn_indices_' + key]`` and nearest
            neighbor distances are stored in ``keeper.misc['nn_distances_' + key]``.
        n_neighbors : {`int`, `None`}
            K-th nearest neighbor (or number of nearest neighbors) to use for computing ``sigmas``,
            ``n_neighbors > 0``. (Uses ``n_neighbors + 1``, since each obs is it's closest neighbor).
            If `None`, all neighbors are used.
        method : {'mean', 'median', 'max'}
            Indicate how to compute sigma.

            Options:

            - 'mean' : mean of distance to ``n_neighbors`` nearest neighbors
            - 'median' : median of distance to ``n_neighbors`` nearest neighbors
            - 'max' : distance to ``n_neighbors``-nearest neighbor
        return_nn : `bool`
            If `True`, also store indices and distances of ``n_neighbors`` nearest neighbors.
    
        Returns
        -------
        sigmas : `numpy.ndarray`, (n_observations, )
            The distance to the k-th nearest neighbor for all rows in ``d``.
            Sigmas represent the kernel width representing each data point's accessible neighbors.
            Written to ``keeper.misc['sigmas_' + key]``.
        indices : `numpy.ndarray`, (n_observations, )
            Indices of nearest neighbors where each row corresponds to an observation.
            Written, if ``return_nn`` is `True`, to ``keeper.misc['nn_indices_' + key]``.
        distances : `numpy.ndarray`, (n_observations, ``n_neighbors + 1``)
            Distances to nearest neighbors where each row corresponds to an obs.
            Written, if ``return_nn`` is `True`, to ``keeper.misc['nn_distances_' + key]``.
        """
        tag = f"{method}{n_neighbors}nn_{distance_key}"
        if label is not None:
            tag = "_".join([tag, label])
        
        nfs.sigma_knn(self, distance_key, label=tag, n_neighbors=n_neighbors,
                      method=method, return_nn=return_nn)


    def compute_similarity_from_distance(self, distance_key, n_neighbors, method, precomputed_method=None,
                                         label=None, knn=False):
        """
        Convert distance matrix to symmetric similarity measure.

        The resulting similarity is written to the similarity keeper.

        Parameters
        ----------
        distance_key : `str`
            The label used to reference the distance matrix stored in ``keeper.distances``,
            of size (n_observations, n_observations).    
        n_neighbors : {`int`, `None`}
            K-th nearest neighbor (or number of nearest neighbors) to use for computing ``sigmas``,
            ``n_neighbors > 0``. (Uses ``n_neighbors + 1``, since each obs is it's closest neighbor).
            If `None`, all neighbors are used.
        method : {`float`, 'mean', 'median', 'max', 'precomputed'}
            Indicate how to compute sigma.

            Options:

            - `float` : constant float to use as sigma
            - `int` : constant int to use as sigma
            - 'mean' : mean of distance to ``n_neighbors`` nearest neighbors
            - 'median' : median of distance to ``n_neighbors`` nearest neighbors
            - 'max' : distance to ``n_neighbors``-nearest neighbor
            - 'precomputed' : precomputed values extracted from ``keeper.misc[f"sigmas_{key}"]`` as a `numpy.ndarray` of size (n_observations, ).
        precomputed_method : {'mean', 'median', 'max'}
            This is ignored if `method` is not `'precomputed'`. When `method` is `'precomputed'`, specify the method that
            was previously used for computing sigmas. See `method` for description of options.
        label : {`None`, `str`}
            If provided, this is appended to the context tag (``tag = f"{method}{n_neighbors}nn_{distance_key}"``)
            The key used to store the resulting similarity matrix of size (n_observations, n_observations)
            in ``keeper.similarities[f"similarity_{key}]`` defaults to the tag when ``label`` is not provided: ``key = tag``. Otherwise,
            the key is set to: ``key = tag + "-" + label``.        
        knn : `bool`
            If `True`, restrict similarity measure to be non-zero only between ``n_neighbors`` nearest neighbors.

        Returns
        -------
        K : `numpy.ndarray`, (n_observations, n_observations)
            Symmetric similarity measure. 
            Written to ``keeper.similarities[key]``.
        """
        if method != 'precomputed':
            precomputed_method = method
            
        tag = f"{str(precomputed_method)}{n_neighbors}nn_{distance_key}"
        if label is not None:
            tag = "_".join([tag, label])
            
        nfs.distance_to_similarity(self, distance_key, n_neighbors, method,
                                   label=f"similarity_{tag}", sigmas=f"sigmas_{tag}",
                                   knn=knn, indices=None)



    def convert_similarity_to_distance(self, similarity_key):
        """ Convert a similarity to a distance.

        The distance, computed as 1-similarity, is added to the distance keeper
        with the key ``"distance_from_" + similarity_key``

        Parameters
        ----------
        similarity_key : `str`
            The similiratiy reference key.

        Returns
        -------
        The following are saved to the distance keeper:

            d : The new distance is saved to the keeper in
               ``keeper.distances[f"distance_from_{similarity_key}"].
        """
        sim = self.similarities[similarity_key]
        self.add_distance(1.-sim.data, f"distance_from_{similarity_key}")
        
        
    def fuse_similarities(self, similarity_keys, weights=None, fused_key=None):
        """ Fuse similarities in the keeper

        Parameters
        ----------
        similarity_keys : `list`
            Reference keys of similiraties to fuse.
        weights : `list`
            (Optional) Weight each similarity contributes to the fused similarity.
            Should be the same length as ``similarity_keys``.
            If not provided, default behavior is to apply uniform weights.
        fused_key : `str`
            (Optional) Specify key used to store the fused similarity in the keeper.
            Default behavior is to fuse the keys of the original similarities.

        Returns
        -------
        The following is added to the similarity keeper :

          - fused_sim : The fused similarity, where the reference key, if not provided,
            is fused from the original labels.
        """
        if (weights is not None) and (len(similarity_keys) != len(weights)):
            raise ValueError("``weights`` must have the same number of values as ``similarity_keys``.")

        if fused_key is None:            
            fused_key = fuse_labels(similarity_keys)

        s = self.similarities[similarity_keys[0]].data
        if weights is not None:
            s = weights[0] * s

        fused_sim = s
        for ix, key in enumerate(similarity_keys[1:], start=1):
            s = self.similarities[key].data
            if weights is not None:
                s = weights[ix] * s
            fused_sim = fused_sim + s

        if weights is None:
            n = len(similarity_keys)
            fused_sim = fused_sim / n

        self.add_similarity(fused_sim, fused_key)
        
        
    def compute_transitions_from_similarity(self, similarity_key, density_normalize: bool = True):
        """ Compute symmetric and asymmetric transition matrices and store in keeper.

        .. note:: Code primarily copied from `scanpy.neighbors`.

        Parameters
        ----------
        similarity_key : `str`
            Reference key to the `numpy.ndarray`, (n_observations, n_observations)
            symmetric similarity measure (with 1s on the diagonal) stored in the similarities
            in the keeper.
        density_normalize : `bool`
            The density rescaling of Coifman and Lafon (2006): Then only the
            geometry of the data matters, not the sampled density.

        Returns
        -------
        Adds the following to `keeper.misc` (with 0s on the diagonals):
            transitions_asym_{similarity_key} : `numpy.ndarray`, (n_observations, n_observations)
                Asymmetric Transition matrix.
            transitions_sym_{similarity_key} : `numpy.ndarray`, (n_observations, n_observations)
                Symmetric Transition matrix.
        
        """
        compute_transitions(self, similarity_key, density_normalize=density_normalize)


    def compute_dpt_from_augmented_sym_transitions(self, key):
        """ Compute the diffusion pseudotime metric between observations,
        computed from the symmetric transitions.

        .. Note::

            - :math:`T` is the symmetric transition matrix
            - :math:`M(x,z) = \sum_{i=1}^{n-1} (\lambda_i * (1 - \lambda_i))\psi_i(x)\psi_i^T(z)`
            - :math:`dpt(x,z) = ||M(x, .) - M(y, .)||^2

        Parameters
        ----------
        key : `str`
            Reference ID for the symmetric transitions `numpy.ndarray`, (n_observations, n_observations)
            stored in ``keeper.misc``.

        Returns
        -------
        dpt : `numpy.ndarray`, (n_observations, n_observations)
            Pairwise-observation Diffusion pseudotime distances are stored
            in keeper.distances["dpt_from_{key}"].
        """
        dpt_from_augmented_sym_transitions(self, key)


    def compute_dpt_from_similarity(self, similarity_key, density_normalize: bool = True):
        """ Compute the diffusion pseudotime metric between observations,
        computed from similarity 

        .. note::

            - This entails computing the augmented symmetric transitions.
            - :math:`T` is the symmetric transition matrix
            - :math:`M(x,z) = \sum_{i=1}^{n-1} (\lambda_i * (1 - \lambda_i))\psi_i(x)\psi_i^T(z)`
            - :math:`dpt(x,z) = ||M(x, .) - M(y, .)||^2

        Parameters
        ----------
        similarity_key : `str`
            Reference key to the `numpy.ndarray`, (n_observations, n_observations)
            symmetric similarity measure (with 1s on the diagonal) stored in the similarities
            in the keeper.
        density_normalize : `bool`
            The density rescaling of Coifman and Lafon (2006): Then only the
            geometry of the data matters, not the sampled density.

        
        Returns
        -------
        The following are stored in the keeper :
           transitions_asym : `numpy.ndarray`, (n_observations, n_observations)
                Asymmetric Transition matrix (with 0s on the diagonal) added to
                ``keeper.misc[f"transitions_asym_{similarity_key}"].
           transitions_sym : `numpy.ndarray`, (n_observations, n_observations)
                Symmetric Transition matrix (with 0s on the diagonal) added to
                ``keeper.misc[f"transitions_sym_{similarity_key}"].
        
           dpt : `numpy.ndarray`, (n_observations, n_observations)
               Pairwise-observation Diffusion pseudotime distances are stored
               in keeper.distances["dpt_from_transitions_asym_{similarity_key}"].
        """
        self.compute_transitions_from_similarity(similarity_key, density_normalize)
        T_sym_key = f"transitions_sym_{similarity_key}"
        self.compute_dpt_from_augmented_sym_transitions(T_sym_key)


    def construct_pose(self, key, root=None,
                       min_branch_size=5, choose_largest_segment=False,
                       flavor='haghverdi16', allow_kendall_tau_shift=False,
                       smooth_corr=True, brute=True, split=True, verbose=None,
                       n_branches=2, until_branched=False, annotate=True,
                       ):                
        """ Construct the POSE from specified distance.

        Parameters
        ----------
        key : `str`
            The label used to reference the distance matrix stored in ``keeper.distances``,
            of size (n_observations, n_observations).
        root : {`None`, `int`, 'density', 'ratio'}
            The root. If `None`, 'density' is used.

            options
            -------
            - `int` : index of observation
            - 'density' : select observation with minimal distance-density
            - 'ratio' : select observation which leads to maximal triangular ratio distance
        min_branch_size : {`int`, `float`}
            During recursive splitting of branches, only consider splitting a branch with at least
            ``min_branch_size > 2`` data points.
            If a `float`, ``min_branch_size`` refers to the fraction of the total number of data points
            (``0 < min_branch_size < 1``).
        choose_largest_segment : `bool`
            If `True`, select largest segment for branching.
        flavor : {'haghverdi16', 'wolf17_tri', 'wolf17_bi', 'wolf17_bi_un'}
            Branching algorithm (based on `scanpy` implementation).
        allow_kendall_tau_shift : `bool`
            If a very small branch is detected upon splitting, shift away from
            maximum correlation in Kendall tau criterion of [Haghverdi16]_ to
            stabilize the splitting.
        smooth_corr : `bool`, default = `False`
            If `True`, smooth correlations before identifying cut points for branch splitting.
        brute : `bool`
            If `True`, data points not associated with any branch upon split are combined with
            undecided (trunk) points. Otherwise, if `False`, they are treated as individual islands,
            not associated with any branch (and assigned branch index -1).
        split : `bool` (default = True)
                if `True`, split segment into multiple branches. Otherwise,
                determine a single branching off of the main segment.
                This is ignored if flavor is not 'haghverdi16'.
                If `True`, ``brute`` is ignored.
        n_branches : `int`
            Number of branches to look for (``n_branches > 0``).
        until_branched : `bool`
            If `True`, iteratively find segment to branch and perform branching
            until a segement is successfully branched or no branchable segments
            remain. Otherwise, if `False`, attempt to perform branching only once 
            on the next potentially branchable segment.

            ..note::

              This is only applicable when branching is being performed. If previous
              iterations of branching has already been performed, it is not possible to
              identify the number of iterations where no branching was performed.
        annotate : `bool`
            If `True`, annotate edges and nodes with POSE features.

        Returns
        -------
        poser : `netflow.pose.POSER`
            The object used to construct the POSE.
        G_poser_nn : `networkx.Graph`
            The updated graph with nearest neighbor edges.
            If ``annotate`` is `True`, edge attribute "edge_origin"
            is added with the possible values :

            - "POSE" : for edges in the original graph that are not nearest neighbor edges
            - "NN" : for nearest neighbor edges that were not in the original graph
            - "POSE + NN" : for edges in the original graph that are also nearest neighbor edges
        """                
        poser = POSER(self, key, root=root, min_branch_size=min_branch_size,
                      choose_largest_segment=choose_largest_segment,
                      flavor=flavor, allow_kendall_tau_shift=allow_kendall_tau_shift,
                      smooth_corr=smooth_corr, brute=brute, split=split, verbose=verbose)

        G_poser = poser.branchings_segments(n_branches, until_branched=until_branched, annotate=annotate)
        G_poser_nn = poser.construct_pose_nn_topology(G_poser, annotate=annotate)

        # label = [key]
        # if root is not None:
        #     label.append(f"root_{root}")
        # label.append(f"n_branches_{n_branches}")
        # label.append(f"min_branch_size_{min_branch_size}")
        # if choose_largest_segment:
        #     label.append("choose_largest_segment")
        # label.append(f"flavor_{flavor}")
        # if allow_kendall_tau_shift:
        #     label.append("allow_kendall_tau_shift")
        # if smooth_corr:
        #     label.append("smooth_corr")
        # if brute:
        #     label.append("brute")
        # if split:
        #     label.append("split")
        # if until_branched:
        #     label.append("until_branched")
        # label = "_".join(label)
        # keeper.add_graph(G_poser_nn, "POSE_{label}")

        return poser, G_poser_nn


    def log1p(self, key, base=None):
        """ Logarithmic data transformation.

        Computes :math:`data = \\log(data + 1)` with the natural logarithm as the default base.

        Parameters
        ----------
        key : `str`
            The reference key of the data in the data-keeper that will be
            logarithmically transformed.
        base : {`None`, `int`}
            Base used for the logarithmic transformation.

        Returns
        -------
        Logarithmically transformed data with label "{key}_log1p" is added to the data keeper.
        """
        data = self.data[key]
        features = data.feature_labels[:]
        obs = data.observation_labels[:]
        data = data.data.copy()
        if not (np.issubdtype(data.dtype, np.floating) or np.issubdtype(data.dtype, complex)):
            data = data.astype(float)
        data = np.log1p(data, out=data)
        if base is not None:
            np.divide(data, np.log(base), out=data)

        data = pd.DataFrame(data=data, index=features, columns=obs)
        self.add_data(data, f"{key}_log1p")
        

