from typing import Tuple, Optional, Sequence, List

import itertools
import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp

import netflow.utils as utl
from ._logging import logger

def norm_features_(X, method='L1'):
    """ Norm of (multi-)feature data points.

    Intended to compute the norm of pairwise distances between obs. :math:`s_q` and :math:`s_r` for all features
    :math:`f_i \in F`: :math:`D_{qr}^{(F)} = [d_{qr}^{(f_1)}, ..., d_{qr}^{(f_m)}]

    Parameters
    ----------
    X : array-like
        The norm is computed on rows of `X`.
    method : {'L1', 'L2', 'inf', 'mean', 'median'}
        Indicate which norm to compute. For each row of the form :math:`x = [x_1, x_2, ..., x_n]`:

        Options
        -------
        'L1' : :math:`\sum_{i=1}^n abs(x_i)`
        'L2' : :math:`\sqrt{\sum_{i=1}^n (x_i)^2}`
        'inf' : :math:`max_i abs(x_i)`
        'mean' : Mean of :math:`x`
        'median' : Median of :math:`x`


    Returns
    -------
    n : float or array-like
        Norm of the row(s).
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



def get_knn_indices_distances(d, n_neighbors=None):
    """ Get indices of and distances to k-nearest neighbors

    Parameters
    ----------
    d : numpy array
        Symmetric distance matrix of size (m, m).
    n_neighbors : int, n_neighbors > 0
        K-th nearest neighbor. (uses `n_neighbors` + 1, since each obs is it's closest neighbor).
        If `None`, all neighbors are used.
    
    Returns
    -------
    indices : numpy array
        Matrix of size (m, n_neighbors) with indices of k-nearest neighbors in each row
        (does not include self in output).
    distances : numpy array
        Matrix of size (m, n_neighbors) with distance to k-nearest neighbors
        (does not include self in output).
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
    """ Set sigma for each obs as the distance to its k-th neighbor

    Parameters
    ----------
    d : numpy array
        Symmetric distance matrix of size (m, m).
    n_neighbors : {int, n_neighbors > 0, None}
        K-th nearest neighbor. (uses `n_neighbors` + 1, since each obs is it's closest neighbor).
        If `None`, all neighbors are used.
    method : {'mean', 'median', 'max'}
        Indicate how to compute sigma.
    return_nn : bool
        If `True`, return indices and distances of `n_neighbors` nearest neighbors.

        Options
        -------
        'mean' : mean of distance to `n_neighbors` nearest neighbors
        'median' : median of distance to `n_neighbors` nearest neighbors
        'max' : distance to `n_neighbors`-nearest neighbor
    

    Returns
    -------
    sigmas : numpy array of size (m, )
        The distance to the k-th nearest neighbor for all rows in `d`.
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
    
    
def similarity_measure_(d, n_neighbors, method, sigmas=None, knn=False, indices=None):
    """
    Convert distance matrix to symmetric similarity measure.

    Parameters
    ----------
    d : numpy array
        Symmetric distance matrix of size (m, m).
    n_neighbors : {int; n_neighbors > 0, None}
        K-th nearest neighbor. (uses `n_neighbors` + 1, since each obs is it's closest neighbor).
        If `None`, all neighbors are used.
        .. note: this is ignored if `method` is 'precomputed' and `sigmas` is provided.
    method : {'mean', 'median', 'max', 'precomputed'}
        Indicate how to compute sigma.

        Options
        -------
        'mean' : mean of distance to `n_neighbors` nearest neighbors
        'median' : median of distance to `n_neighbors` nearest neighbors
        'max' : distance to `n_neighbors`-nearest neighbor
        'precomputed' : precomputed values passed to `sigmas`
    sigmas : numpy array of size (m, )
        Option to provide precomputed sigmas , ignored unless method='precomputed'.
    knn : bool
        If `True`, restrict similarity measure to be non-zero only between `n_neighbors` nearest neighbors.
    indices : numpy array of size (m, `n_neighbors`)
        Option to provide precomputed indices of `n_neighbors` nearest neighbors for each obs when
        `method` = 'precomputed' and `knn` = `True`

    Returns
    -------
    K : numpy array
        Symmetric similarity measure of size (m, m).    
    """

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

    sigmas_sq = np.power(sigmas, 2)
    
    Num = 2 * np.multiply.outer(sigmas, sigmas)
    Den = np.add.outer(sigmas_sq, sigmas_sq)
    K = np.sqrt(Num / Den) * np.exp(-np.power(d, 2) / (2 * Den)) # TO DO: integral may not sum to 1

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


class PseudoOrdering:
    """ Pseudo-ordering of observations (e.g., samples). 

    Represent data matrix as a graph of associations (i.e., edges) among data points (i.e., observations or nodes).

    Parameters
    ----------
    all_feats_pairwise_obs_dists : pandas DataFrame
        Distance matrix of the form :math:`D^{(F)} = [d_{ij}]` of size :math:`(m * (m-1) * 0.5, n)`,
        where :math:`m` is the number of observations (obs), :math:`n` is the number of features (feats),
        :math:`F` is the set of features where :math:`|F| = n`, with a multi-index of size 2
        of the form :math:`(obs_i, obs_j)` and :math:`d_{ij}` is the distance between pairwise-obs
        :math:`i = (obs_p, obs_q)` with respect to feature :math:`j`.
    method_norm : {'L1', 'L2', 'inf', 'mean', 'median'}
        Indicate how to compute norm over multiple features, for each row of the form :math:`x = [x_1, x_2, ..., x_n]`:

        Options
        -------
        'L1' : :math:`\sum_{i=1}^n abs(x_i)`
        'L2' : :math:`\sqrt{\sum_{i=1}^n (x_i)^2}`
        'inf' : :math:`max_i abs(x_i)`
        'mean' : Mean of :math:`x`
        'median' : Median of :math:`x`
    n_neighbors : {int, n_neighbors > 0, None}
        K-th nearest neighbor (or number of nearest neighbors) to use for computing `sigmas`
        (uses `n_neighbors` + 1, since each obs is it's closest neighbor).
        If `None`, all neighbors are used.
    method_sigma : {'mean', 'median', 'max', 'precomputed'}
        Indicate how to compute sigma.

        Options
        -------
        'mean' : mean of distance to `n_neighbors` nearest neighbors
        'median' : median of distance to `n_neighbors` nearest neighbors
        'max' : distance to `n_neighbors`-nearest neighbor
        'precomputed' : precomputed values passed to `sigmas`
    sigmas : numpy array of size (m, )
        Option to provide precomputed sigmas, ignored unless method='precomputed'.
    knn : bool
        If `True`, restrict similarity measure to be non-zero only between `n_neighbors` nearest neighbors.
    root : int
        Index of root obs that pesudo-ordering is computed from (`root` > 0).
    """
    def __init__(self, all_feats_pairwise_obs_dists, method_norm='L1',
                 n_neighbors=5, method_sigma='max', sigmas=None, knn=False,
                 root=None):
        self.all_feats_pairwise_obs_dists = all_feats_pairwise_obs_dists
        self.index = sorted(set(itertools.chain(*self.all_feats_pairwise_obs_dists.index)))

        self._method_norm = method_norm
        self._n_neighbors = n_neighbors
        self._method_sigma = method_sigma
        self._sigmas = sigmas
        self._knn = knn
        self.root = root
        self.pseudotime = None
        
        # pairwise-obs distances of size :math:`(m, m)`, which gives the norm of distance over all features in `all_feats_pairwise_obs_dists`:
        self._distances = None
        self._nn_distances = None
        self._nn_indices = None

        self._similarities = None # convert `_distances` to `similarity` measure.


        self._transitions_asym = None
        self._transitions_sym = None

        self.pseudotime_distances = None


    @property
    def distances(self):
        """ Distances between data points. """        
        return self._distances


    @property
    def sigmas(self):
        """ Kernel width representing each data point's accessible neighbors. """
        return self._sigmas

    @property
    def nn_indices(self):
        """ Indices of nearest neighbors where each row corresponds to an obs. """
        return self._nn_indices

    @property
    def nn_distances(self):
        """ distances to nearest neighbors where each row corresponds to an obs. """
        return self._nn_distances
    

    @property
    def similarities(self):
        """ Similarities between data points. """        
        return self._similarities


    def compute_norm_features(self):
        """ Compute norm of obs-pairwise distances with respect to all features in `all_feats_pairwise_obs_dists` returned
        as symmetric obs-pairwise distance matrix of size (m, m). """
        self._distances = utl.unstack_triu_(norm_features_(self.all_feats_pairwise_obs_dists,
                                                          method=self._method_norm),
                                            index=self.index)


    def compute_sigma_knn(self):
        """ Compute sigmas from nearest neighbors. """
        if self._distances is None:
            msg = "Must compute distances before sigmas can be computed, running `compute_norm_features` now."
            logger.msg(msg)
            self.compute_norm_features()
            
        self._sigmas, self._nn_indices, self._nn_distances = sigma_knn_(self._distances, n_neighbors=self._n_neighbors, method=self._method_sigma, return_nn=True)

        
    def compute_similarity_measure(self):
        """ convert `_distances` to `similarity` measure. """
        if self._distances is None:
            msg = "Must compute distances before similarities can be computed, running `compute_norm_features` now."
            # raise AssertionError(msg)
            logger.msg(msg)
            self.compute_norm_features()        

        if self._method_sigma == 'precomputed':
            self._similarities = similarity_measure_(self._distances, self._n_neighbors, self._method_sigma, sigmas=self._sigmas, knn=self._knn, indices=self._nn_indices)
        else:
            # if not yet computed, compute sigmas so computation performed only once and save values
            if (self._sigmas is None) or (self._knn and (self._nn_indices is None)):
                self.compute_sigma_knn()
                
            self._similarities = similarity_measure_(self._distances, self._n_neighbors, 'precomputed', sigmas=self._sigmas, knn=self._knn, indices=self._nn_indices)

    def compute_transitions(self, similarities=None, density_normalize: bool = True):
        """ Compute transition matrix.

        Parameters
        ----------
        similarities : numpy array
            Symmetric similarity measure of size (m, m).    
        density_normalize : bool
            The density rescaling of Coifman and Lafon (2006): Then only the
            geometry of the data matters, not the sampled density.

        Returns
        -------
        Makes attributes `.transitions_sym` and `.transitions` available.

        Comments
        --------
        Code copied from scanpy.neighbors.
        """
        
        W = self._similarities if similarities is None else similarities
        # density normalization as of Coifman et al. (2005)
        # ensures that kernel matrix is independent of sampling density
        if density_normalize:
            # q[i] is an estimate for the sampling density at point i
            # it's also the degree of the underlying graph
            q = np.asarray(W.sum(axis=0))
            if not issparse(W):
                Q = np.diag(1.0 / q)
            else:
                Q = scipy.sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
            K = Q @ W @ Q
        else:
            K = W

        # asym transitions
        # z[i] is the row sum of K
        z = np.asarray(K.sum(axis=0))
        if not issparse(K):
            Z = np.diag(1.0 / z)
        else:
            Z = scipy.sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
        self._transitions_asym = Z @ K

        # sym transitions
        # z[i] is the square root of the row sum of K
        z = np.sqrt(np.asarray(K.sum(axis=0)))
        if not issparse(K):
            Z = np.diag(1.0 / z)
        else:
            Z = scipy.sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
        self._transitions_sym = Z @ K @ Z



    def _set_pseudotime(self, root=None, ordering='distance', data=None):
        """ Return pseudotime with respect to root point.

        Parameters
        ----------
        root : int; 0 <= root < m where m is the number of obs.
            Root obs for computing pseudo-ordering. If `None`, `self.root` is used.
            If provided, `self.root` is updated to `root`.
        ordering : {'distance', 'similarity', 'transitions_asym', 'transitions_sym', 'precomputed'}
            Metric by which pseudo-ordering should be computed.

            Options
            -------
            'distance' : `self._distances`
            'similarity' : 1 - `self._similarities`
            'transitions_asym' : 1 - `self._transitions_asym`
            'transitions_sym' : 1 - `self._transitions_sym`
            'precomputed' : 1 - precomputed
        data :
            Similarity matrix used when `ordering` is "precomputed".
            .. note: this is ignored if `ordering` is not "precomputed".
        
        """

        self.root = root if root is not None else root
        if self.root is None:
            msg = "'root' must be specified in order to compute pseudo-ordering."
            raise AssertionError(msg)

        if ordering == 'distance':
            self.pseudotime = self._distances[self.root].copy()

        elif ordering == 'similarity':
            pt = self._similarities[self.root].copy()
            pt[self.root] = np.max(pt) + 1e-3
            self.pseudotime = np.max(pt) - pt

        elif ordering == 'transitions_asym':
            # not setting self to 1 - transition in case root has transition probability of 1 to another obs, and want self to have smallest pseudotime
            pt = self._transitions_asym[self.root].copy()
            pt[self.root] = np.max(pt) + 1e-3
            self.pseudotime = np.max(pt) - pt

        elif ordering == 'transitions_sym':
            # not setting self to 1 - transition in case root has transition probability of 1 to another obs, and want self to have smallest pseudotime
            pt = self._transitions_sym[self.root].copy()
            pt[self.root] = np.max(pt) + 1e-3
            self.pseudotime = np.max(pt) - pt

        elif ordering == 'precomputed':
            if data is None:
                msg = "Must provide `data` when ordering is precomputed."
                raise AssertionError(msg)

            pt = self._transitions_sym[self.root].copy()
            pt[self.root] = np.max(pt) + 1e-3
            self.pseudotime = np.max(pt) - pt


        else:
            msg = f"Unrecognized value {ordering!r} for `ordering`, must be one of ['distance', 'similarity', 'transitions_asym', 'transitions_sym']."
            raise AssertionError(msg)            
            
        self.pseudotime /= np.max(self.pseudotime[self.pseudotime < np.inf])

    def pseudotime_distances(self, ordering='distance', data=None):
        """ Compute distance matrix used for pseudotime branching 

        Parameters
        ----------
        ordering : {'distance', 'similarity', 'transitions_asym', 'transitions_sym', 'precomputed'}
            Metric by which pseudo-ordering should be computed.

            Options
            -------
            'distance' : `self._distances`
            'similarity' : 1 - `self._similarities`
            'transitions_asym' : 1 - `self._transitions_asym`
            'transitions_sym' : 1 - `self._transitions_sym`
            'precomputed' : 1 - precomputed
        data :
            Similarity matrix used when `ordering` is "precomputed".
            .. note: this is ignored if `ordering` is not "precomputed".
        
        """

        if ordering == 'distance':
            self.pseudotime_distances = self._distances.copy()

        elif ordering == 'similarity':
            pt = self._similarities
            self.pseudotime_distances = np.max(pt) - pt

        elif ordering == 'transitions_asym':
            # not setting self to 1 - transition in case root has transition probability of 1 to another obs, and want self to have smallest pseudotime
            pt = self._transitions_asym
            self.pseudotime_distances = np.max(pt) - pt

        elif ordering == 'transitions_sym':
            # not setting self to 1 - transition in case root has transition probability of 1 to another obs, and want self to have smallest pseudotime
            pt = self._transitions_sym
            self.pseudotime_distances = np.max(pt) - pt

        elif ordering == 'precomputed':
            if data is None:
                msg = "Must provide `data` when ordering is precomputed."
                raise AssertionError(msg)

            pt = self._transitions_sym
            self.pseudotime_distances = np.max(pt) - pt
            

        else:
            msg = f"Unrecognized value {ordering!r} for `ordering`, must be one of ['distance', 'similarity', 'transitions_asym', 'transitions_sym']."
            raise AssertionError(msg)            




class TDA:
    """ Class to compute topological branching analysis

    Parameters
    ----------
    distances : {pandas DataFrame, numpy array}
        Symmetric distance matrix between data points.
        .. note: If `distances` is a pandas DataFrame, it's Stored and processed as a numpy array but the index is stored for later conversion.
    min_branch_size : int
        Minimal number of data points needed to be considered as a branch.
    choose_largest_segment : bool

    flavor : {'haghverdi16', 'wolf17_tri', 'wolf17_bi', 'wolf17_bi_un'}
    
    allow_kendall_tau_shift : bool
        If a very small branch is detected upon splitting, shift away from
        maximum correlation in Kendall tau criterion of [Haghverdi16]_ to
        stabilize the splitting.
    
    """
    def __init__(self, distances, min_branch_size=5, choose_largest_segment=False,
                 flavor='haghverdi16', allow_kendall_tau_shift=False):
        self.distances = distances if not isinstance(distances, pd.DataFrame) else distances.values
        self.index_labels = None if not isinstance(distances, pd.DataFrame) else distances.index
        self.min_branch_size = min_branch_size
        self.choose_largest_segment = choose_largest_segment
        self.flavor = flavor
        self.allow_kendall_tau_shift = allow_kendall_tau_shift

        
    def detect_branches(self, n_branches, root=None):
        """ Detect up to `n_branches` branches.

        Parameters
        ----------
        n_branches : int
            Number of branches to look for (`n_branches` > 0).
        root : int
            Index of root obs that pesudo-ordering is computed from (`root` > 0).
    
        Returns
        -------
        
        """
        # distances = distances if not isinstance(self.distances, pd.DataFrame) else distances.values
        
        indices_all = np.arange(self.distances.shape[0], dtype=int)
        # branch_hierarchy = [indices_all]  # keep record
        segs = [indices_all]

        # get first tip (farthest from root)
        if root is None:
            tip_0 = np.argmax(self.distances[0])
        else:
            tip_0 = np.argmax(self.distances[root])

        # get tip of other end (farthest from tip_0)
        tips_all = np.array([tip_0, np.argmax(self.distances[tip_0])])
        # branch_tips = [tips_all]  # keep record
        segs_tips = [tips_all]

        segs_connects = [[]]
        segs_undecided = [True]
        segs_adjacency = [[]]
        # segs_terminate_branching = [False]  # RE added - to indicate if segment has already been branched as much as possible 
        
        for ibranch in range(n_branches):
            logger.warning(f"*ibranch = {ibranch}")
            iseg, tips3 = self.select_segment(segs, segs_tips, segs_undecided)
            logger.warning(f"*iseg = {iseg}, tips3 = {tips3}, selected_seg = {segs[iseg]}")
            if iseg == -1:
                logger.debug('    partitioning converged')
                break
            logger.debug(
                f'    branching {ibranch + 1}: split group {iseg}',
            )  # [third start end]
            # detect branching and update segs and segs_tips
            self.detect_branching(
                segs,
                segs_tips,
                segs_connects,
                segs_undecided,
                segs_adjacency,
                iseg,
                tips3,
            )

        # store as class members
        self.segs = segs
        self.segs_tips = segs_tips
        self.segs_undecided = segs_undecided
        # the following is a bit too much, but this allows easy storage
        self.segs_adjacency = sp.sparse.lil_matrix((len(segs), len(segs)), dtype=float)
        self.segs_connects = sp.sparse.lil_matrix((len(segs), len(segs)), dtype=int)
        for i, seg_adjacency in enumerate(segs_adjacency):
            self.segs_connects[i, seg_adjacency] = segs_connects[i]
        for i in range(len(segs)):
            for j in range(len(segs)):
                self.segs_adjacency[i, j] = self.distances[
                    self.segs_connects[i, j], self.segs_connects[j, i]
                ]
        self.segs_adjacency = self.segs_adjacency.tocsr()
        self.segs_connects = self.segs_connects.tocsr()

        # RE: Add for points that weren't found in any of the resulting segments

            


    def select_segment(self, segs, segs_tips, segs_undecided):
        """ Select segment with most distant second data point

        Returns
        -------
        iseg
            Index identifying the position within the list of line segments.
        tips3
            Positions of tips within chosen segment.
        """
        scores_tips = np.zeros((len(segs), 4))
        allindices = np.arange(self.distances.shape[0], dtype=int)
        # logger.warning(f"{len(segs)} segs iterating over.")
        for iseg, seg in enumerate(segs):
            # logger.warning(f"iseg = {iseg} begin iteration")
            if segs_tips[iseg][0] == -1: # do not consider too small segments???
                logger.warning(f"iseg = {iseg} ending iterations short")
                continue

            # restrict distance matrix to points in segment
            Dseg = self.distances[np.ix_(seg, seg)]

            third_maximizer = None
            
            if segs_undecided[iseg]:
                # check that no tip connects with tip of another seg
                for jseg in range(len(segs)):
                    if jseg != iseg:
                        for itip in range(2):
                            if (
                                self.distances[
                                    segs_tips[jseg][1], segs_tips[iseg][itip]
                                ]
                                < 0.5
                                * self.distances[
                                    segs_tips[iseg][~itip], segs_tips[iseg][itip]
                                ]
                            ):
                                # logger.debug(
                                #     '    group', iseg, 'with tip', segs_tips[iseg][itip],
                                #     'connects with', jseg, 'with tip', segs_tips[jseg][1],
                                # )
                                # logger.debug('    do not use the tip for "triangulation"')
                                third_maximizer = itip
                                
            
            # map the global position to the position within the segment
            # logger.warning(f"iseg = {iseg}, tips = {segs_tips[iseg]}, seg = {seg}, ")
            tips = [np.where(allindices[seg] == tip)[0][0] for tip in segs_tips[iseg]] # local index of tips in the seg
            # logger.warning(f"iseg = {iseg} made it to line 658 and 660")
            # find the third point on the seg that has maximal added distance from the two tip points:
            dseg = Dseg[tips[0]] + Dseg[tips[1]]
            if not np.isfinite(dseg).any():
                continue
            third_tip = np.argmax(dseg)

            # logger.warning(f"iseg = {iseg} made it to line 658 and 666")
            if third_maximizer is not None:
                # find a fourth point that has maximal distance to all three
                dseg += Dseg[third_tip]
                fourth_tip = np.argmax(dseg)
                if fourth_tip != tips[0] and fourth_tip != third_tip:
                    tips[1] = fourth_tip
                    dseg -= Dseg[tips[1]]
                else:
                    dseg -= Dseg[third_tip]
            tips3 = np.append(tips, third_tip)

            # logger.warning(f"iseg = {iseg} made it to line 658 and 677")

            # compute the score as ratio of the added distance to the third tip,
            # to what it would be if it were on the straight line between the
            # two first tips, given by Dseg[tips[:2]]
            # if we did not normalize, there would be a danger of simply
            # assigning the highest score to the longest segment
            score = dseg[tips3[2]] / Dseg[tips3[0], tips3[1]]
            # logger.warning(f"iseg = {iseg}, SCORE = {score}")
            # score = (
            #     len(seg) if self.choose_largest_segment else score
            # )  # simply the number of points
            score = len(seg) if self.choose_largest_segment else score # simply the number of points

            # RE - following not needed because score automatically set to 0 if len(seg) < min_branch_size (RE TODO: set threshold to check min_branch_size > 2)
            # score = 0. if len(seg) <= 1 else score # TODO: RE added - remove segs with 1 point (maybe also remove segs with <= 2 points?)
            # score = 0. if np.isnan(score) else score # TODO: RE added - in case ratio of (d(0,x) + d(x, 1)) / d(0,1) is np.nan?
            
            # logger.warning(f"iseg = {iseg}, SCORE2 = {score}")
            logger.debug(
                f'    group {iseg} score {score} n_points {len(seg)} ' + '(too small)'
                if len(seg) < self.min_branch_size
                else '',
            )
            
            if len(seg) <= self.min_branch_size:
                score = 0
                # logger.warning(f"iseg = {iseg}, SCORE3 = {score}")
            # write result
            scores_tips[iseg, 0] = score
            scores_tips[iseg, 1:] = tips3
            # logger.warning(f"iseg = {iseg} end of iteration")
            
        iseg = np.argmax(scores_tips[:, 0])
        if scores_tips[iseg, 0] == 0:
            return -1, None
        tips3 = scores_tips[iseg, 1:].astype(int)
        return iseg, tips3            
        

    def detect_branching(self, segs, segs_tips, segs_connects, segs_undecided,
                         segs_adjacency, iseg, tips3):
        """ Detect branching on a given segment and update list parameters in place.

        Parameters
        ----------
        segs
            Dchosen distance matrix restricted to segment.
        segs_tips
            Stores all tip points for the segments in segs.
        iseg
            Position of segment under study in segs.
        tips3
            The three tip points. They form a 'triangle' that contains the data.
        """

        seg = segs[iseg]
        Dseg = self.distances[np.ix_(seg, seg)]
        # logger.warning(f"*seg = {seg}")

        # given the three tip points and the distance matrix detect the
        # branching on the segment, return the list ssegs of segments that
        # are defined by splitting this segment
        result = self._detect_branch(Dseg, tips3, seg)        
        if result is None: # RE ADDED THIS CONDITION
            logger.warning(f"No unique branch detected - removed from consideration.")
        else:
            ssegs, ssegs_tips, ssegs_adjacency, ssegs_connects, trunk = result

            # map back to global indices
            for iseg_new, seg_new in enumerate(ssegs):
                ssegs[iseg_new] = seg[seg_new]
                ssegs_tips[iseg_new] = seg[ssegs_tips[iseg_new]]
                ssegs_connects[iseg_new] = list(seg[ssegs_connects[iseg_new]])

            # remove previous segment
            segs.pop(iseg)
            segs_tips.pop(iseg)

            # insert trunk/undecided_cells at same position
            segs.insert(iseg, ssegs[trunk])
            segs_tips.insert(iseg, ssegs_tips[trunk])

            # append other segments
            segs += [seg for iseg, seg in enumerate(ssegs) if iseg != trunk]
            segs_tips += [
                seg_tips for iseg, seg_tips in enumerate(ssegs_tips) if iseg != trunk
            ]
            if len(ssegs) == 4:
                # insert undecided cells at same position
                segs_undecided.pop(iseg)
                segs_undecided.insert(iseg, True)

            # QUESTION FROM HERE
            # correct edges in adjacency matrix
            n_add = len(ssegs) - 1
            prev_connecting_segments = segs_adjacency[iseg].copy()
            if self.flavor == 'haghverdi16':
                segs_adjacency += [[iseg] for i in range(n_add)]
                segs_connects += [
                    seg_connects
                    for iiseg, seg_connects in enumerate(ssegs_connects)
                    if iiseg != trunk
                ]
                prev_connecting_points = segs_connects[  # noqa: F841  TODO Evaluate whether to assign the variable or not
                    iseg
                ]
                for jseg_cnt, jseg in enumerate(prev_connecting_segments):
                    iseg_cnt = 0
                    for iseg_new, seg_new in enumerate(ssegs):
                        if iseg_new != trunk:
                            pos = segs_adjacency[jseg].index(iseg)
                            connection_to_iseg = segs_connects[jseg][pos]
                            if connection_to_iseg in seg_new:
                                kseg = len(segs) - n_add + iseg_cnt
                                segs_adjacency[jseg][pos] = kseg
                                pos_2 = segs_adjacency[iseg].index(jseg)
                                segs_adjacency[iseg].pop(pos_2)
                                idx = segs_connects[iseg].pop(pos_2)
                                segs_adjacency[kseg].append(jseg)
                                segs_connects[kseg].append(idx)
                                break
                            iseg_cnt += 1
                segs_adjacency[iseg] += list(
                    range(len(segs_adjacency) - n_add, len(segs_adjacency))
                )
                segs_connects[iseg] += ssegs_connects[trunk]
            else:
                segs_adjacency += [[] for i in range(n_add)]
                segs_connects += [[] for i in range(n_add)]
                kseg_list = [iseg] + list(range(len(segs) - n_add, len(segs)))
                for jseg in prev_connecting_segments:
                    pos = segs_adjacency[jseg].index(iseg)
                    distances = []
                    closest_points_in_jseg = []
                    closest_points_in_kseg = []
                    for kseg in kseg_list:
                        reference_point_in_k = segs_tips[kseg][0]
                        closest_points_in_jseg.append(
                            segs[jseg][
                                np.argmin(
                                    self.distances[reference_point_in_k, segs[jseg]]
                                )
                            ]
                        )
                        # do not use the tip in the large segment j, instead, use the closest point
                        reference_point_in_j = closest_points_in_jseg[
                            -1
                        ]  # segs_tips[jseg][0]
                        closest_points_in_kseg.append(
                            segs[kseg][
                                np.argmin(
                                    self.distances[reference_point_in_j, segs[kseg]]
                                )
                            ]
                        )
                        distances.append(
                            self.distances[
                                closest_points_in_jseg[-1], closest_points_in_kseg[-1]
                            ]
                        )
                        # print(jseg, '(', segs_tips[jseg][0], closest_points_in_jseg[-1], ')',
                        #       kseg, '(', segs_tips[kseg][0], closest_points_in_kseg[-1], ') :', distances[-1])
                    idx = np.argmin(distances)
                    kseg_min = kseg_list[idx]
                    segs_adjacency[jseg][pos] = kseg_min
                    segs_connects[jseg][pos] = closest_points_in_kseg[idx]
                    pos_2 = segs_adjacency[iseg].index(jseg)
                    segs_adjacency[iseg].pop(pos_2)
                    segs_connects[iseg].pop(pos_2)
                    segs_adjacency[kseg_min].append(jseg)
                    segs_connects[kseg_min].append(closest_points_in_jseg[idx])

                # if we split two clusters, we need to check whether the new segments connect to any of the other
                # old segments
                # if not, we add a link between the new segments, if yes, we add two links to connect them at the
                # correct old segments
                do_not_attach_kseg = False
                for kseg in kseg_list:
                    distances = []
                    closest_points_in_jseg = []
                    closest_points_in_kseg = []
                    jseg_list = [
                        jseg
                        for jseg in range(len(segs))
                        if jseg != kseg and jseg not in prev_connecting_segments
                    ]
                    for jseg in jseg_list:
                        reference_point_in_k = segs_tips[kseg][0]
                        closest_points_in_jseg.append(
                            segs[jseg][
                                np.argmin(
                                    self.distances[reference_point_in_k, segs[jseg]]
                                )
                            ]
                        )
                        # do not use the tip in the large segment j, instead, use the closest point
                        reference_point_in_j = closest_points_in_jseg[
                            -1
                        ]  # segs_tips[jseg][0]
                        closest_points_in_kseg.append(
                            segs[kseg][
                                np.argmin(
                                    self.distances[reference_point_in_j, segs[kseg]]
                                )
                            ]
                        )
                        distances.append(
                            self.distances[
                                closest_points_in_jseg[-1], closest_points_in_kseg[-1]
                            ]
                        )
                    idx = np.argmin(distances)
                    jseg_min = jseg_list[idx]
                    if jseg_min not in kseg_list:
                        segs_adjacency_sparse = sp.sparse.lil_matrix(
                            (len(segs), len(segs)), dtype=float
                        )
                        for i, seg_adjacency in enumerate(segs_adjacency):
                            segs_adjacency_sparse[i, seg_adjacency] = 1
                            G = nx.Graph(segs_adjacency_sparse)
                        paths_all = nx.single_source_dijkstra_path(G, source=kseg)
                        if jseg_min not in paths_all:
                            segs_adjacency[jseg_min].append(kseg)
                            segs_connects[jseg_min].append(closest_points_in_kseg[idx])
                            segs_adjacency[kseg].append(jseg_min)
                            segs_connects[kseg].append(closest_points_in_jseg[idx])
                            logg.debug(f'    attaching new segment {kseg} at {jseg_min}')
                            # if we split the cluster, we should not attach kseg
                            do_not_attach_kseg = True
                        else:
                            logger.debug(
                                f'    cannot attach new segment {kseg} at {jseg_min} '
                                '(would produce cycle)'
                            )
                            if kseg != kseg_list[-1]:
                                logger.debug('        continue')
                                continue
                            else:
                                logger.debug('        do not add another link')
                                break

                    if jseg_min in kseg_list and not do_not_attach_kseg:
                        segs_adjacency[jseg_min].append(kseg)
                        segs_connects[jseg_min].append(closest_points_in_kseg[idx])
                        segs_adjacency[kseg].append(jseg_min)
                        segs_connects[kseg].append(closest_points_in_jseg[idx])
                        break

            segs_undecided += [False for i in range(n_add)]                
                
            
            


    def _detect_branch(self, Dseg: np.ndarray,
                       tips: np.ndarray,
                       seg_reference=None,
                       ) -> Tuple[
                           List[np.ndarray],
                           List[np.ndarray],
                           List[List[int]],
                           List[List[int]],
                           int,
                       ]:
        """ Detect branching on given segment.

        Call function __detect_branching three times for all three orderings of
        tips. Points that do not belong to the same segment in all three
        orderings are assigned to a fourth segment. The latter is, by Haghverdi
        et al. (2016) referred to as 'undecided cells'.

        Parameters
        ----------
        Dseg
            Dchosen distance matrix restricted to segment.
        tips
            The three tip points. They form a 'triangle' that contains the data.

        Returns
        -------
        ssegs
            List of segments obtained from splitting the single segment defined
            via the first two tip cells.
        ssegs_tips
            List of tips of segments in ssegs.
        ssegs_adjacency : list
            List of lists of the same length as ssegs,
            where the i-th entry is a list with the index of the trunk, if the i-th segment is not the trunk.
            Otherwise, the i-th entry is a list with the indices of all other segments beside the trunk.
        ssegs_connects : list
            List of lists of the same length as ssegs,
            where the i-th entry is a list of the form [index of data point in the trunk closest to the root of the i-th segment],
            if the i-th segment is not the trunk. Otherwise, the i-th entry is a list of indices of the closest cell in each other (non-trunk) segment
            to the trunk root.
        trunk : int 
            Index of segment in ssegs that is the trunk. When there are undecided points, the trunk is the seg of undecided points.
            If there are no undecided points and 3 segments in sseg (i.e. branching), then the trunk is the seg with the smallest distance to
            the other segments. If there are only two segments in sseg, the first segment is set as the trunk.
        """
        if self.flavor == 'haghverdi16':
            ssegs = self._detect_branching_single_haghverdi16(Dseg, tips)  # correlation
        elif self.flavor == 'wolf17_tri':
            ssegs = self._detect_branching_single_wolf17_tri(Dseg, tips)  # closer in distance to tip than other two tips
        elif self.flavor == 'wolf17_bi' or self.flavor == 'wolf17_bi_un': 
            ssegs = self._detect_branching_single_wolf17_bi(Dseg, tips)
        else:
            raise ValueError(
                '`flavor` needs to be in {"haghverdi16", "wolf17_tri", "wolf17_bi, "wolf17_bi_un""}.'
            )

        # make sure that each data point has a unique association with a segment
        masks = np.zeros((len(ssegs), Dseg.shape[0]), dtype=bool)
        for iseg, seg in enumerate(ssegs):
            masks[iseg][seg] = True
        nonunique = np.sum(masks, axis=0) > 1
        ssegs = []

        for iseg, mask in enumerate(masks):
            mask[nonunique] = False
            # RE START MODIFIED  - IN EVENT THAT THE SEG HAD NO UNIQUE MEMBERS AND IS NOW EMPTY:
            # ssegs.append(np.arange(Dseg.shape[0], dtype=int)[mask])

            newseg = np.arange(Dseg.shape[0], dtype=int)[mask]
            if newseg.shape[0] == 0:
                logger.warning("Unique segment is empty, removing from consideration and no branching performed.")
                # continue
                return None
            
            ssegs.append(newseg)
            # RE END MODIFIED

        # compute new tips within new segments
        ssegs_tips = []
        logger.warning(f"*ssegs = {ssegs}")
        for inewseg, newseg in enumerate(ssegs):
            # RE MODIFIED START
            # if len(np.flatnonzero(newseg)) <= 1:
            #     logger.warning(f'detected group with only {len(np.flatnonzero(newseg))} data points')
            # secondtip = newseg[np.argmax(Dseg[tips[inewseg]][newseg])]
            # ssegs_tips.append([tips[inewseg], secondtip]) # RE: SHOULD BE CHANGED

            logger.warning(f"*inewseg = {inewseg}, newseg = {newseg}")
            secondtip = newseg[np.argmax(Dseg[tips[inewseg]][newseg])]
            firsttip = tips[inewseg]
            
            if len(np.flatnonzero(newseg)) <= 1:
                logger.warning(f'detected group with only {len(np.flatnonzero(newseg))} data points')
            if firsttip not in set(newseg):
                new_firsttip = newseg[np.argmin(Dseg[tips[inewseg]][newseg])]
                logger.warning(f'tip is no longer in the unique branched sub-segment, update tip to its nearest point in the new segment: {firsttip} -> {new_firsttip}')
                firsttip = new_firsttip
            
            ssegs_tips.append([firsttip, secondtip]) # RE: SHOULD BE CHANGED
            # RE MODIFIED END
        undecided_cells = np.arange(Dseg.shape[0], dtype=int)[nonunique]

        if len(undecided_cells) > 0:
            ssegs.append(undecided_cells)
            # establish the connecting points with the other segments
            ssegs_connects = [[], [], [], []]
            for inewseg, newseg_tips in enumerate(ssegs_tips):
                reference_point = newseg_tips[0]
                # closest cell to the new segment within undecided cells
                closest_cell = undecided_cells[
                    np.argmin(Dseg[reference_point][undecided_cells])
                ]
                ssegs_connects[inewseg].append(closest_cell)
                # closest cell to the undecided cells within new segment
                closest_cell = ssegs[inewseg][
                    np.argmin(Dseg[closest_cell][ssegs[inewseg]])
                ]
                ssegs_connects[-1].append(closest_cell)

            # also compute tips for the undecided cells
            tip_0 = undecided_cells[
                np.argmax(Dseg[undecided_cells[0]][undecided_cells])
            ]

            tip_1 = undecided_cells[np.argmax(Dseg[tip_0][undecided_cells])]
            ssegs_tips.append([tip_0, tip_1])
            ssegs_adjacency = [[3], [3], [3], [0, 1, 2]]
            # RE START MODIFIED
            # trunk = 3
            trunk = len(ssegs) - 1
            logger.warning(f"trunk = {trunk}")
            # RE END MODIFIED
        elif len(ssegs) == 3:
            reference_point = np.zeros(3, dtype=int)
            reference_point[0] = ssegs_tips[0][0]
            reference_point[1] = ssegs_tips[1][0]
            reference_point[2] = ssegs_tips[2][0]
            closest_points = np.zeros((3, 3), dtype=int)

            # this is another strategy than for the undecided_cells
            # here it's possible to use the more symmetric procedure
            # shouldn't make much of a difference
            closest_points[0, 1] = ssegs[1][
                np.argmin(Dseg[reference_point[0]][ssegs[1]])
            ]
            closest_points[1, 0] = ssegs[0][
                np.argmin(Dseg[reference_point[1]][ssegs[0]])
            ]
            closest_points[0, 2] = ssegs[2][
                np.argmin(Dseg[reference_point[0]][ssegs[2]])
            ]
            closest_points[2, 0] = ssegs[0][
                np.argmin(Dseg[reference_point[2]][ssegs[0]])
            ]
            closest_points[1, 2] = ssegs[2][
                np.argmin(Dseg[reference_point[1]][ssegs[2]])
            ]
            closest_points[2, 1] = ssegs[1][
                np.argmin(Dseg[reference_point[2]][ssegs[1]])
            ]

            added_dist = np.zeros(3)
            added_dist[0] = (
                Dseg[closest_points[1, 0], closest_points[0, 1]]
                + Dseg[closest_points[2, 0], closest_points[0, 2]]
            )

            added_dist[1] = (
                Dseg[closest_points[0, 1], closest_points[1, 0]]
                + Dseg[closest_points[2, 1], closest_points[1, 2]]
            )

            added_dist[2] = (
                Dseg[closest_points[1, 2], closest_points[2, 1]]
                + Dseg[closest_points[0, 2], closest_points[2, 0]]
            )

            trunk = np.argmin(added_dist)
            ssegs_adjacency = [
                [trunk] if i != trunk else [j for j in range(3) if j != trunk]
                for i in range(3)
            ]

            ssegs_connects = [
                [closest_points[i, trunk]]
                if i != trunk
                else [closest_points[trunk, j] for j in range(3) if j != trunk]
                for i in range(3)
            ]

        else:
            trunk = 0
            ssegs_adjacency = [[1], [0]]
            reference_point_in_0 = ssegs_tips[0][0]
            closest_point_in_1 = ssegs[1][
                np.argmin(Dseg[reference_point_in_0][ssegs[1]])
            ]
            reference_point_in_1 = closest_point_in_1  # ssegs_tips[1][0]
            closest_point_in_0 = ssegs[0][
                np.argmin(Dseg[reference_point_in_1][ssegs[0]])
            ]
            ssegs_connects = [[closest_point_in_1], [closest_point_in_0]]

        return ssegs, ssegs_tips, ssegs_adjacency, ssegs_connects, trunk            


    
        
    def _detect_branching_single_haghverdi16(self, Dseg, tips):
        """Detect branching on given segment."""
        # compute branchings using different starting points the first index of
        # tips is the starting point for the other two, the order does not
        # matter
        ssegs = []
        # permutations of tip cells
        ps = [
            [0, 1, 2],  # start by computing distances from the first tip
            [1, 2, 0],  #             -"-                       second tip
            [2, 0, 1],  #             -"-                       third tip
        ]
        for i, p in enumerate(ps):
            # logger.warning(f"*iterating p: i = {i}")
            ssegs.append(self.__detect_branching_haghverdi16(Dseg, tips[p]))
        return ssegs


    def _detect_branching_single_wolf17_tri(self, Dseg, tips):
        # all pairwise distances
        dist_from_0 = Dseg[tips[0]]
        dist_from_1 = Dseg[tips[1]]
        dist_from_2 = Dseg[tips[2]]
        closer_to_0_than_to_1 = dist_from_0 < dist_from_1
        closer_to_0_than_to_2 = dist_from_0 < dist_from_2
        closer_to_1_than_to_2 = dist_from_1 < dist_from_2
        masks = np.zeros((2, Dseg.shape[0]), dtype=bool)
        masks[0] = closer_to_0_than_to_1
        masks[1] = closer_to_0_than_to_2
        segment_0 = np.sum(masks, axis=0) == 2
        masks = np.zeros((2, Dseg.shape[0]), dtype=bool)
        masks[0] = ~closer_to_0_than_to_1
        masks[1] = closer_to_1_than_to_2
        segment_1 = np.sum(masks, axis=0) == 2
        masks = np.zeros((2, Dseg.shape[0]), dtype=bool)
        masks[0] = ~closer_to_0_than_to_2
        masks[1] = ~closer_to_1_than_to_2
        segment_2 = np.sum(masks, axis=0) == 2
        ssegs = [segment_0, segment_1, segment_2]
        return ssegs


    def _detect_branching_single_wolf17_bi(self, Dseg, tips):
        dist_from_0 = Dseg[tips[0]]
        dist_from_1 = Dseg[tips[1]]
        closer_to_0_than_to_1 = dist_from_0 < dist_from_1
        ssegs = [closer_to_0_than_to_1, ~closer_to_0_than_to_1]
        return ssegs
    

    def __detect_branching_haghverdi16(self, Dseg: np.ndarray, tips: np.ndarray) -> np.ndarray:
        """
        Detect branching on given segment.

        Compute point that maximizes kendall tau correlation of the sequences of
        distances to the second and the third tip, respectively, when 'moving
        away' from the first tip: tips[0]. 'Moving away' means moving in the
        direction of increasing distance from the first tip.

        Parameters
        ----------
        Dseg
            Dchosen distance matrix restricted to segment.
        tips
            The three tip points. They form a 'triangle' that contains the data.

        Returns
        -------
        Segments obtained from "splitting away the first tip cell".
        """
        # sort distance from first tip point
        # then the sequence of distances Dseg[tips[0]][idcs] increases
        # logger.warning(f"*inside: Dseg = {Dseg}, tips = {tips}")
        idcs = np.argsort(Dseg[tips[0]])
        # logger.warning(f"*inside now")
        # logger.warning(f"*inside: idcs = {idcs}")
        # consider now the sequence of distances from the other
        # two tip points, which only increase when being close to `tips[0]`
        # where they become correlated
        # at the point where this happens, we define a branching point
        # logger.warning(f"*tips = {tips}")
        # logger.warning(f"*a = {Dseg[tips[1]][idcs]}")
        # logger.warning(f"*b = {Dseg[tips[2]][idcs]}")
        

        if True:
            imax = self.kendall_tau_split(
                Dseg[tips[1]][idcs],
                Dseg[tips[2]][idcs],
            )
        if False:
            # if we were in euclidian space, the following should work
            # as well, but here, it doesn't because the scales in Dseg are
            # highly different, one would need to write the following equation
            # in terms of an ordering, such as exploited by the kendall
            # correlation method above
            imax = np.argmin(
                Dseg[tips[0]][idcs] + Dseg[tips[1]][idcs] + Dseg[tips[2]][idcs]
            )

        # init list to store new segments
        # NOTE: ssegs not used...
        ssegs = []  # noqa: F841  # TODO Look into this
        # first new segment: all points until, but excluding the branching point
        # increasing the following slightly from imax is a more conservative choice
        # as the criterion based on normalized distances, which follows below,
        # is less stable
        if imax > 0.95 * len(idcs) and self.allow_kendall_tau_shift:            
            # if "everything" is correlated (very large value of imax), a more
            # conservative choice amounts to reducing this
            logger.warning(
                'shifting branching point away from maximal kendall-tau '
                'correlation (suppress this with `allow_kendall_tau_shift=False`)'
            )
            ibranch = int(0.95 * imax)            
        else:
            # otherwise, a more conservative choice is the following
            ibranch = imax + 1
        return idcs[:ibranch]


    def kendall_tau_split(self, a, b,  min_length=5) -> int:
        """Return splitting index that maximizes correlation in the sequences.

        Compute difference in Kendall tau for all splitted sequences.

        For each splitting index i, compute the difference of the two
        correlation measures kendalltau(a[:i], b[:i]) and
        kendalltau(a[i:], b[i:]).

        Returns the splitting index that maximizes
            kendalltau(a[:i], b[:i]) - kendalltau(a[i:], b[i:])

        Parameters
        ----------
        a, b : np.ndarray
            One dimensional sequences.
        min_length : int (`min_length` > 0)
            Minimum number of data points automatically included in branch.

        Returns
        -------
        Splitting index according to above description.
        """
        if a.size != b.size:
            raise ValueError('a and b need to have the same size')
        if a.ndim != b.ndim != 1:
            raise ValueError('a and b need to be one-dimensional arrays')

        # logger.warning(f"*a = {a}")
        # logger.warning(f"*b = {b}")
        # logger.warning(f"*a[:min_length] = {a[:min_length]}")
        # logger.warning(f"*a[min_length:] = {a[min_length:]}")
        # logger.warning(f"*b[:min_length] = {b[:min_length]}")
        # logger.warning(f"*b[min_length:] = {b[min_length:]}")
        
        n = a.size
        idx_range = np.arange(min_length, a.size - min_length - 1, dtype=int)
        corr_coeff = np.zeros(idx_range.size)
        pos_old = sp.stats.kendalltau(a[:min_length], b[:min_length])[0]
        neg_old = sp.stats.kendalltau(a[min_length:], b[min_length:])[0]
        # logger.warning(f"*pos_old = {pos_old}")
        # logger.warning(f"*neg_old = {neg_old}")
        # logger.warning(f"*idx_range = {idx_range}")
        # logger.warning(f"*corr_coeff = {corr_coeff}")
        for ii, i in enumerate(idx_range):
            # logger.warning(f"*ii, i = {ii}, {i}")
            if True:
                # compute differences in concordance when adding a[i] and b[i]
                # to the first subsequence, and removing these elements from
                # the second subsequence
                diff_pos, diff_neg = self._kendall_tau_diff(a, b, i)
                pos = pos_old + self._kendall_tau_add(i, diff_pos, pos_old)
                neg = neg_old + self._kendall_tau_subtract(n - i, diff_neg, neg_old)
                pos_old = pos
                neg_old = neg
            if False:
                # computation using sp.stats.kendalltau, takes much longer!
                # just for debugging purposes
                pos = sp.stats.kendalltau(a[: i + 1], b[: i + 1])[0]
                neg = sp.stats.kendalltau(a[i + 1 :], b[i + 1 :])[0]
            if False:
                # the following is much slower than using sp.stats.kendalltau,
                # it is only good for debugging because it allows to compute the
                # tau-a version, which does not account for ties, whereas
                # sp.stats.kendalltau computes tau-b version, which accounts for
                # ties
                pos = sp.stats.mstats.kendalltau(a[:i], b[:i], use_ties=False)[0]
                neg = sp.stats.mstats.kendalltau(a[i:], b[i:], use_ties=False)[0]

            corr_coeff[ii] = pos - neg
            # logger.warning(f"*corr_coeff = {corr_coeff}, corr_coeff[{ii}] = {pos - neg}")
        # RE START MODIFIED
        # iimax = np.argmax(corr_coeff)
        # imax = min_length + iimax
        # corr_coeff_max = corr_coeff[iimax]
        iimax = 0 if corr_coeff.size == 0 else np.argmax(corr_coeff)
        imax = min_length + iimax
        corr_coeff_max = 0. if corr_coeff.size == 0 else corr_coeff[iimax]
        
        # RE END MODIFIED
        if corr_coeff_max < 0.3:
            logger.debug('    is root itself, never obtain significant correlation')
        return imax


    def _kendall_tau_add(self, len_old: int, diff_pos: int, tau_old: float):
        """Compute Kendall tau delta.

        The new sequence has length len_old + 1.

        Parameters
        ----------
        len_old
            The length of the old sequence, used to compute tau_old.
        diff_pos
            Difference between concordant and non-concordant pairs.
        tau_old
            Kendall rank correlation of the old sequence.
        """
        return 2.0 / (len_old + 1) * (float(diff_pos) / len_old - tau_old)


    def _kendall_tau_subtract(self, len_old: int, diff_neg: int, tau_old: float):
        """Compute Kendall tau delta.

        The new sequence has length len_old - 1.

        Parameters
        ----------
        len_old
            The length of the old sequence, used to compute tau_old.
        diff_neg
            Difference between concordant and non-concordant pairs.
        tau_old
            Kendall rank correlation of the old sequence.
        """
        return 2.0 / (len_old - 2) * (-float(diff_neg) / (len_old - 1) + tau_old)


    def _kendall_tau_diff(self, a: np.ndarray, b: np.ndarray, i) -> Tuple[int, int]:
        """Compute difference in concordance of pairs in split sequences.

        Consider splitting a and b at index i.

        Parameters
        ----------
        a, b : np.ndarray
            One dimensional sequences.
        i : int
            Index for splitting `a` and `b`.

        Returns
        -------
        diff_pos
            Difference between concordant pairs for both subsequences.
        diff_neg
            Difference between non-concordant pairs for both subsequences.
        """
        # compute ordering relation of the single points a[i] and b[i]
        # with all previous points of the sequences a and b, respectively
        a_pos = np.zeros(a[:i].size, dtype=int)
        a_pos[a[:i] > a[i]] = 1
        a_pos[a[:i] < a[i]] = -1
        b_pos = np.zeros(b[:i].size, dtype=int)
        b_pos[b[:i] > b[i]] = 1
        b_pos[b[:i] < b[i]] = -1
        diff_pos = np.dot(a_pos, b_pos).astype(float)

        # compute ordering relation of the single points a[i] and b[i]
        # with all later points of the sequences
        a_neg = np.zeros(a[i:].size, dtype=int)
        a_neg[a[i:] > a[i]] = 1
        a_neg[a[i:] < a[i]] = -1
        b_neg = np.zeros(b[i:].size, dtype=int)
        b_neg[b[i:] > b[i]] = 1
        b_neg[b[i:] < b[i]] = -1
        diff_neg = np.dot(a_neg, b_neg)


        return diff_pos, diff_neg

        

            
        
            


    

                



