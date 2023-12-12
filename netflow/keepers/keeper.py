"""
keeper
======

Classes used for data storage and manipulation. 
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .._utils import _docstring_parameter, _desc_distance, _desc_data_distance
from .._logging import logger


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
        self._observation_labels = observation_labels
        self._num_observations = None if observation_labels is None else len(observation_labels)
        self._features_labels = {}
        self._num_features = {}

        if isinstance(data, (pd.DataFrame, np.ndarray)):
            self.add_data(data, 'data')
        elif isinstance(data, dict):
            for label, cur_data in data.items():
                self.add_data(cur_data, label)


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
                self._observation_labels = data.columns.tolist()
                logger.debug("Set observation labels.")
            feature_labels = data.index.tolist()
            data = data[self._observation_labels].values
        else:  # np.ndarray
            feature_labels = None        

        self._data[label] = data
        self._features_labels[label] = feature_labels
        self._num_features[label] = data.shape[0]

        logger.debug(f"Data {label} with {data.shape[0]} features added to keeper.")
    
            

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
        data : `numpy.ndarray`, (``len(observations)``, ``len(features)``)
            The data subset where the :math:`ij^{th}` entry is the value of the
            :math:`i^{th}` feature in ``features`` for the  :math:`j^{th}`
            observation in ``observations``.
        """
        if (observations is None) and (features is None):
            raise ValueError("Either observations or features must be provided.")

        data = self._data
        if observations is not None:
            if isinstance(observations[0], str):  # convert to location index
                observations = [self._observation_labels.index(k) for k in observations]
            data = data[:, observations]

        if features is not None:
            if isinstance(features[0], str):  # convert to location index
                features = [self._feature_labels.index(k) for k in features]
            data = data[features, :]

        return data


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
        self._observation_labels = observation_labels
        self._num_observations = None if observation_labels is None else len(observation_labels)

        if isinstance(data, (pd.DataFrame, np.ndarray)):
            self.add_data(data, 'distance')
        elif isinstance(data, dict):
            for label, cur_distance in data.items():
                self.add_data(cur_distance, label)


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
                self._observation_labels = data.columns.tolist()
                logger.debug("Set observation labels.")
            data = data.loc[self._observation_labels, self._observation_labels].values

        self._data[label] = data

        logger.debug(f"Distance {label} between {data.shape[0]} observations added to keeper.")


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
        distance : `numpy.ndarray`, (``len(observations_a)``, ``len(observations_b)``)
            The sub-matrix of distances where the :math:`ij^{th}` entry is the distance
            between the :math:`i^{th}` observation in ``observations_a`` and the
            :math:`j^{th}` observation in ``observations_b``.
        """

        distance = self._data

        if isinstance(observations_a[0], str):  # convert to location index
                observations_a = [self._observation_labels.index(k) for k in observations_a]

        if observations_b is None:
            observations_b = observations_a
        elif isinstance(observations_b[0], str):  # convert to location index
                observations_b = [self._observation_labels.index(k) for k in observations_b]

        distance = distance[np.ix_(observations_a, observations_b)]
        return distance


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
        Similarit(y/ies) may be provided in multiple ways, analogous to ``distances``.:
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
                 misc=None, observation_labels=None, outdir=None):


        if (data is None) and (distances is None) and (similarities is None):
            raise ValueError("At least one of data, distances, or similarities must be provided.")
        
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
        if observation_labels is None:
            observation_labels = [f"X{i}" for i in range(num_observations)]
            
        self._observation_labels = observation_labels
        # self._num_observations = None if self._observation_labels is None else len(self._observation_labels)
        self._num_observations = len(self._observation_labels)

        self._data = DataKeeper(data=data, observation_labels=self._observation_labels)
        self._distances = DistanceKeeper(data=distances,
                                         observation_labels=self._observation_labels)
        self._similarities = DistanceKeeper(data=similarities,
                                            observation_labels=self._observation_labels)
        self._misc = misc if misc is not None else {}

        self._check_num_observations()
        self._check_observation_labels()

        if isinstance(outdir, str):
            outdir = Path(outdir)
        self.outdir = outdir
        if isinstance(self.outdir, Path) and not self.outdir.is_dir():
            self.outdir.mkdir()
            logger.msg(f"Creating directory {self.outdir}.")


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
        logger.msg(f"Added data input {label} to the keeper.")


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
        logger.msg(f"Added distance input {label} to the keeper.")


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
        logger.msg(f"Added similarity input {label} to the keeper.")    


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

        logger.msg(f"Added misc input {label} to the keeper.")


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

        
