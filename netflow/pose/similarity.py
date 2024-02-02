import numpy as np
import pandas as pd

from .._logging import logger



def get_knn_indices_distances(d, n_neighbors=None):
    """ Get indices of and distances to k-nearest neighbors

    Parameters
    ----------
    d : `numpy.ndarray`, (m, m)
        Symmetric distance matrix.
    n_neighbors : {`int`, `None`}
        K-th nearest neighbor (or number of nearest neighbors) to use for computing ``sigmas``,
        ``n_neighbors > 0``. (Uses ``n_neighbors + 1``, since each obs is it's closest neighbor).
        If `None`, all neighbors are used.
    Returns
    -------
    indices : `numpy.ndarray`, (m, n_neighbors)
        Matrix with indices of k-nearest neighbors in each row
        Note, this does not include itself in output)
    distances : `numpy.ndarray`, (m, n_neighbors)
        Matrix with distance to k-nearest neighbors
        Note, this does not include itself in output.
    """
    if isinstance(d, (pd.DataFrame, pd.Series)):
        d = d.values
    n_neighbors = d.shape[0] - 1 if n_neighbors is None else n_neighbors
    sample_range = np.arange(d.shape[0])[:, None]
    indices = np.argpartition(d, n_neighbors, axis=1)[:, :n_neighbors + 1]
    indices = indices[sample_range, np.argsort(d[sample_range, indices])] # each row has indices of k nearest neighbors sorted in order of nearest neighbor (including self)
    distances = d[sample_range, indices] # each row has sorted distances to k nearest neighbors (including self)

    # exclude the first point (self - 0th neighbor)
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    return indices, distances


def sigma_knn_(d, n_neighbors=None, method='mean', return_nn=False):
    """ Determine sigma for each obs as the distance to its k-th neighbor.    

    Parameters
    ----------
    d : `numpy.ndarray`, (n_observations, n_observations)
        Symmetric distance matrix.
    n_neighbors : {`int`, `None`}
        K-th nearest neighbor (or number of nearest neighbors) to use for computing ``sigmas``,
        ``n_neighbors > 0``. (Uses ``n_neighbors + 1``, since each obs is it's closest neighbor).
        If `None`, all neighbors are used.
    method : {'mean', 'median', 'max'}
        Indicate how to compute sigma.
    return_nn : `bool`
        If `True`, also return indices and distances of ``n_neighbors`` nearest neighbors.

        Options:

        - 'mean' : mean of distance to ``n_neighbors`` nearest neighbors
        - 'median' : median of distance to ``n_neighbors`` nearest neighbors
        - 'max' : distance to ``n_neighbors``-nearest neighbor

    Returns
    -------
    sigmas : `numpy.ndarray`, (n_observations, )
        The distance to the k-th nearest neighbor for all rows in ``d``.
        Sigmas represent the kernel width representing each data point's accessible neighbors.
    indices : `numpy.ndarray`, (n_observations, )
        Indices of nearest neighbors where each row corresponds to an observation.
        Returned if ``return_nn`` is `True`.
    distances : `numpy.ndarray`, (n_observations, ``n_neighbors + 1``)
        Distances to nearest neighbors where each row corresponds to an obs.
        Returned if ``return_nn`` is `True`.
    """
    indices, distances = get_knn_indices_distances(d, n_neighbors=n_neighbors)
    if method == 'mean':
        sigmas = np.mean(distances, axis=1)
    elif method == 'median':
        sigmas = np.median(distances, axis=1)
    elif method == 'max':
        sigmas = distances[:, -1] / 2
    else:
        msg = f"{method!r} not recognized for value of method, must be one of ['mean', 'median', 'max']."
        raise AssertionError(msg)

    if return_nn:
        return sigmas, indices, distances
    
    return sigmas


def sigma_knn(keeper, key, label=None, n_neighbors=None, method='mean', return_nn=False):
    """ Set sigma for each obs as the distance to its k-th neighbor from keeper.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object that stores the symmetric distance matrix of size (n_observations, n_observations).
    key : `str`
        The label used to reference the distance matrix stored in ``keeper.distances``,
        of size (n_observations, n_observations).
    label : `str`
        Label used to store resulting sigmas in ``keeper.misc['sigmas_'+label]``. If ``return_nn`` is `True`,
        nearest neighbor indices are stored in ``keeper.misc['nn_indices_'+label]`` and nearest
        neighbor distances are stored in ``keeper.misc['nn_distances_'+label]``.
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
        If `True`, also return/store indices and distances of ``n_neighbors`` nearest neighbors.
    Returns
    -------
    sigmas : `numpy.ndarray`, (n_observations, )
        The distance to the k-th nearest neighbor for all rows in ``d``.
        Sigmas represent the kernel width representing each data point's accessible neighbors.
    indices : `numpy.ndarray`, (n_observations, )
        Indices of nearest neighbors where each row corresponds to an observation.
        Returned if ``return_nn`` is `True`.
    distances : `numpy.ndarray`, (n_observations, ``n_neighbors + 1``)
        Distances to nearest neighbors where each row corresponds to an obs.
        Returned if ``return_nn`` is `True`.
    """
    d = keeper.distances[key]
    sigmas, indices, distances = sigma_knn_(d.data, n_neighbors=n_neighbors, method=method, return_nn=True)

    if label is None:
        if return_nn:
            return sigmas, indices, distances
        else:
            return sigmas
    else:
        label = label if label == '' else '_'+label
        keeper.add_misc(sigmas, 'sigmas'+label)
        if return_nn:
            keeper.add_misc(indices, 'nn_indices'+label)
            keeper.add_misc(distances, 'nn_distances'+label)
        return None

# this used to be def similarity_measure_()
def distance_to_similarity_(d, n_neighbors, method, sigmas=None, knn=False, indices=None):
    """
    Convert distance matrix to symmetric similarity measure.

    .. math::

       K = \sqrt{2\sigma_i\sigma_j / (\sigma_i^2 + \sigma_j^2)}\exp{-(x-y)^2 / (\sigma_x^2 + \sigma_y^2)}.

    Parameters
    ----------
    d : `numpy.ndarray`, (n_observations, n_observations)
        Symmetric distance matrix.
    n_neighbors : {`int`, `None`}
        K-th nearest neighbor (or number of nearest neighbors) to use for computing ``sigmas``,
        ``n_neighbors > 0``. (Uses ``n_neighbors + 1``, since each obs is it's closest neighbor).
        If `None`, all neighbors are used.
    method : {`float`, `int`, 'mean', 'median', 'max', 'precomputed'}
        Indicate how to compute sigma.

        Options:

        - `float` : constant float to use as sigma
        - `int` : constant int to use as sigma
        - 'mean' : mean of distance to ``n_neighbors`` nearest neighbors
        - 'median' : median of distance to ``n_neighbors`` nearest neighbors
        - 'max' : distance to ``n_neighbors``-nearest neighbor
        - 'precomputed' : precomputed values passed to ``sigmas``
    sigmas : `numpy.ndarray`, (n_observations, )
        Option to provide precomputed sigmas , ignored unless ``method='precomputed'``.
    knn : `bool`
        If `True`, restrict similarity measure to be non-zero only between ``n_neighbors`` nearest neighbors.
    indices : `numpy.ndarray`, (n_observations, n_neighbors)
        Option to provide precomputed indices of ``n_neighbors`` nearest neighbors for each obs when
        ``method`` = 'precomputed' and ``knn`` = `True`

    Returns
    -------
    K : `numpy.ndarray`, (n_observations, n_observations)
        Symmetric similarity measure.
    """
    if isinstance(method, float):
        sigmas = np.array([method]*d.shape[0])
    elif isinstance(method, int):
        sigmas = np.array([float(method)]*d.shape[0])
    elif isinstance(method, str):        
        if method == 'precomputed':
            if sigmas is None:
                msg = "When `method` is 'precomputed', `sigmas` must be provided."
                raise AssertionError(msg)
            if knn and (indices is None):
                msg = "When `method` is 'precomputed' and `knn` is `True`, `indices` must be provided."
                raise AssertionError(msg)
        else:
            if knn:
                sigmas, indices, distances = sigma_knn_(d, n_neighbors=n_neighbors, method=method, return_nn=True)
            else:
                sigmas = sigma_knn_(d, n_neighbors=n_neighbors, method=method, return_nn=False)
    else:
        raise TypeError("Uncrecognized type for `method`, must be `float`, `int`, or `str`.")        

    sigmas_sq = np.power(sigmas, 2)
    
    Num = 2 * np.multiply.outer(sigmas, sigmas)
    Den = np.add.outer(sigmas_sq, sigmas_sq)
    K = np.sqrt(Num / Den) * np.exp(-np.power(d, 2) / (2 * Den)) # TO DO: integral may not sum to 1
    # K = np.sqrt(Num / Den) * np.exp(-np.power(d, 2) / (Den)) # TO DO: integral may not sum to 1    

    if knn:  # restrict to nearest neighbors (symmetric)
        mask = np.zeros(d.shape, dtype=bool)
        for i, row in enumerate(indices):
            mask[i, row] = True
            for j in row:
                if i not in set(indices[j]):
                    K[j, i] = K[i, j]
                    mask[j, i] = True
        # set all entries that are not nearest neighbors to zero
        K[~mask] = 0

    # remove neglibible values:
    mask = K > 1e-11
    K[~mask] = 0

    return K


def distance_to_similarity(keeper, key, n_neighbors, method,
                           label=None, sigmas=None, knn=False, indices=None):
    """
    Convert distance matrix to symmetric similarity measure.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object that stores the symmetric distance matrix of size (n_observations, n_observations).
    key : `str`
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
        - 'precomputed' : precomputed values passed to ``sigmas``
    label : `str`
        Label used to store resulting similarity matrix of size (n_observations, n_observations)
        in ``keeper.similarities``.
    sigmas : `str` 
        Option to provide precomputed sigmas, ignored unless ``method='precomputed'``.
        If provided, the precomputed sigmas are extracted from ``keeper.misc[sigmas]`` 
        as a `numpy.ndarray` of size (n_observations, ).
    knn : `bool`
        If `True`, restrict similarity measure to be non-zero only between ``n_neighbors`` nearest neighbors.
    indices : {`None`, `str`}
        Option to provide precomputed indices of ``n_neighbors`` nearest neighbors for each obs when
        ``method`` = 'precomputed' and ``knn`` = `True`.
        If provided, the indices are extracted from ``keeper.misc[indices]`` as a `numpy.ndarray` of size
        (n_observations, n_neighbors).

    Returns
    -------
    K : `numpy.ndarray`, (n_observations, n_observations)
        Symmetric similarity measure. if ``label`` is not `None`, this is
        stored in ``keeper.similarities[label]`` instead of being returned.
    """
    d = keeper.distances[key]

    if isinstance(method, str) and method=='precomputed':
        if sigmas is not None:
            sigmas = keeper.misc[sigmas]
        else:
            raise ValueError("Must pass sigmas key when method='precomputed'.")
            
    if indices is not None:
        indices = keeper.misc[indices]
    K = distance_to_similarity_(d.data, n_neighbors, method, sigmas=sigmas,
                                knn=knn, indices=indices)
    if label is None:
        return K
    else:
        keeper.add_similarity(K, label)
        return None
