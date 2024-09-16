"""
organization
============

**Description**

The purpose of This module is to construct the organization of the
schema from a distance matrix and a single (or multiple)
data point(s) designated as the source.

This is done by using the branch detection algorithm
from the diffusion pseudo-time (DPT) algorithm for reconstructing
developmental progression and differentiation of cells proposed
in [Haghverdi16]_ as implemented in scanpy.

**Acknowledgement**

A large portion of the code was taken from
scanpy.tools._dpt.py and code related to
the method :mod:`scanpy.tools._dpt.dpt`.

Some noted differences made in scanpy implementation :

  - Add smoothing when computing maximal correlation cutoff
  - Include points not identified with any branch after split in the trunk (nonunique).

To do:

  - Set branchable aspect of TreeNode.
"""

from typing import Tuple, Optional, Sequence, List

import itertools
import networkx as nx
import numpy as np
from operator import itemgetter
import pandas as pd
import pptree
import scipy as sp
from scipy.sparse import issparse
from collections import defaultdict as ddict

# from importlib import reload
# pptree = reload(pptree)

# import netflow.utils as utl
import netflow.utils as utl
# from importlib import reload
# reload(utl)
# from .._logging import logger, set_verbose ###
from .._logging import _gen_logger, set_verbose

logger = _gen_logger(__name__)

# RE: TODO: check how branches are connected to each other if flavor != 'haghverdi16'
# RE: TODO: ADD OPTION FOR MULTIPLE ROOTS
### RE: TODO: CHECK HOW UNIQUE BRANCHES ARE DETERMINED
# RE: TODO: CHECK WHICH TRANSITION MATRIX IS USED AND HOW IT'S DEFINED
### RE: TODO: CHECK HOW BRANCHING CONNECTIONS IS DEFINED WHEN THERE IS A TRUNK
# RE: TODO: IF MAX CORR < THRESH, MAYBE DON"T INCLUDE BRANCH?
# RE: TODO: ADD OPTION TO CHANGE ROOT AND UPDATE PSEUDOTIME, SEGS, AND ORDERING? -- this should be for earlier step in pipeline
# RE: TODO: SHOULD -1 be included in segs_names_unique?

def get_pose(keeper, key, label, n_branches, until_branched=False, 
             root=None, min_branch_size=5, choose_largest_segment=False,
             flavor='haghverdi16', allow_kendall_tau_shift=False,
             smooth_corr=False, brute=True, split=True, verbose=None):
    """ Compute the pose and saved to keeper.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object that stores the distance matrix of size (n_observations, n_observations).
    key : `str`
        The label used to reference the distance matrix stored in ``keeper.distances``,
        of size (n_observations, n_observations).
    label : `str`
        Label used to store resulting schema in ``keeper.misc[label]`` and POSE topology in
        ``keeper.graphs[label]``..
    n_branches : `int`
            Number of branch splits to perform (``n_branches > 0``).
    until_branched : `bool`
            If `True`, iteratively find segment to branch and perform branching
            until a segement is successfully branched or no branchable segments
            remain. Otherwise, if `False`, attempt to perform branching only once 
            on the next potentially branchable segment.
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

    Returns
    -------
    Writes the following to the keeper :

      - poser : `POSER`

        * The poser object with the pseudo-organizational branching structure
          is stored in ``keeper.misc['poser_{label}']``.
      - G_pose : `networkx.Graph`    

        * The resulting pose topology is stored in ``keeper.graphs['pose_{label}']``.
      - G_pose_nn : `networkx.Graph`

        * The resulting pose + nearest-neighbor (nn) topology is stored in
          ``keeper.graphs['pose_nn_{label}]``.
    """
    poser = POSER(keeper, key, root=root,
                  min_branch_size=min_branch_size, choose_largest_segment=choose_largest_segment,
                  flavor=flavor, allow_kendall_tau_shift=allow_kendall_tau_shift,
                  smooth_corr=smooth_corr, brute=brute, split=split, verbose=verbose)
    G_pose = poser.branchings_segments(n_branches, until_branched=until_branched)
    G_pose.name = 'pose_' + label
    keeper.add_misc(poser, 'poser_' + label)
    keeper.add_graph(G_pose, 'pose_' + label)

    G_pose_nn = poser.construct_pose_nn_topology(G_pose)
    G_pose_nn.name = 'pose_nn_' + label
    keeper.add_graph(G_pose_nn, 'pose_nn_' + label)


class TreeNode:
    """ Node of a general tree data structure.

    Each node is intended to refer to a branch.

    Parameters
    ----------
    name
        Reference name of node (branch).
    data 
        Data associated with the node.
        Intended to be a `list` of indices corresponding to the branch members.
    children : `list` [`TreeNode`]
        List of children `TreeNode` objects.
    parent : `TreeNode`
        Parent `TreeNode` object.
    nonunique : `bool`
        Indicate if node (branch) is the trunk.
    unidentified : `bool`
        Indicate if node (branch) is a set of points that were not identified
        with a particular branch after splitting.
    branchable : `bool`
        Indicate if node can potentially be further branched.
    is_trunk : `bool`
        Indicate if node referes to undecided trunk branch.
    """
    def __init__(self, name='root', data=None, children=None, parent=None,
                 nonunique=None, unidentified=None, branchable=True, is_trunk=None):
        self.name = name
        self.name_string = str(name)
        if nonunique:
            self.name_string = " - ".join([self.name_string, 'trunk'])
        if data is not None:
            self.name_string = self.name_string + f" (n = {len(data)})"
        self.data = data
        self.parent = parent        
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)

        self.nonunique = nonunique
        self.unidentified = unidentified        
        self.branchable = branchable # True # indicate if branch can be further split
        self.is_trunk = is_trunk # undecided

        self.tips = None
        self.score = None
        self.connections = None

        # used to indicate the order in which nodes are inserted and as unique identifier
        self._counter = None         


    def __repr__(self):
        return self.name_string

    
    def is_root(self):
        if self.parent is None:
            return True
        else:
            return False

        
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False

        
    def depth(self):
        """ Depth of current node. """
        if self.is_root():
            return 0
        else:
            return 1 + self.parent.depth()

        
    def add_child(self, node):
        """ Add child to node.

        Parameters
        ----------
        node : `TreeNode`
            The child node.
        """
        node.parent = self
        if isinstance(node, TreeNode):
            self.children.append(node)
        else:
            raise TypeError("Unrecognized type, node must be a TreeNode.")


    def contains(self, value):
        """ Check if value is in data. """
        found = value in self.data
        return found


    def disp(self):
        cur_name = self.name
        self.name = str(self.name)
        pptree.print_tree(self, 'children', 'name_string')
        self.name = cur_name


class Tree:
    """
    Tree implemenation as a collection of TreeNode objects.

    Intended to represent the hierarchical branching.
    """
    def __init__(self):
        self.root = None
        # self.height = 0
        self.nodes = []
        self._counter = 0
        # self.node_adjacency = ddict(list)
        self.node_connection = [] # ddict(list)


    def disp(self):
        print(self.tree.root.disp())
        

    def insert(self, node, index=None, parent=None):
        """ Insert a node into the Tree.

        Parameters
        ----------
        node : `TreeNode`
            Node to insert.
        index : {`None`, `int`}
            Index in list of nodes where the node should be inserted.
            (Intended to match current structure for updating segments
            until tree structure is fully leveraged (e.g., using tree
            leaf nodes when searching for which segment to select).

            If `None`, the node is appended to the end of the list.
        parent : {`None`, `TreeNode`}
            Parent node. If `None`, node is set as the root node.
        """
        if parent is not None:
            if self.root is None:
                raise AssertionError("The first node inserted into the tree must be the root node with no parent.")
            parent.add_child(node)
        else:
            if self.root is None:
                self.root = node

        if index is None:
            self.nodes.append(node)
        else:
            self.nodes.insert(index, node)

        node._counter = self._counter
        self._counter += 1


    def get_node_from_name(self, name, bottom_up=True):
        """ Search and return node in Tree by its name.

        Assumes no nodes at the same depth have the same name.
        If more than one node has the same name, return the node
        of the deepest node (farthest from root), when
        ``bottom_up = True``, otherwise, return the index of the 
        shallowest (closest to root) node.

        If no such node is found with the specified name, `None`
        is returned.
        
        Parameters
        ----------
        name
            Name of node to search for.
        bottom_up : `bool`
            Indicate if the index of the shallowest or deepest node
            should be returned when more than one node has the same
            name. It is assumed that no two nodes at the same depth
            have the same name.

        Returns
        -------
        node : `TreeNode`
            Node in the tree. If node is not found, returns `None`.
        """
        index = self.search(name, bottom_up=bottom_up)
        node = self.nodes[index]
        return node
            

    def search(self, name, bottom_up=True):
        """ Search and return index of node in Tree by its name.

        Assumes no nodes at the same depth have the same name.
        If more than one node has the same name, return the index
        of the deepest node (farthest from root), when
        ``bottom_up = True``, otherwise, return the index of the 
        shallowest (closest to root) node.

        If no such node is found with the specified name, the
        value -1 is returned.
        
        Parameters
        ----------
        name
            Name of node to search for.
        bottom_up : `bool`
            Indicate if the index of the shallowest or deepest node
            should be returned when more than one node has the same
            name. It is assumed that no two nodes at the same depth
            have the same name.

        Returns
        -------
        index : `int`
            Index of node in the tree. If node is not found, returns -1.
        """
        nodes = [(ix, node.depth()) for ix, node in enumerate(self.nodes) if node.name == name]
        if len(nodes) == 0:
            index = -1
        else:
            nodes = sorted(nodes, key=itemgetter(1))
            if bottom_up:
                index = nodes[-1][0]
            else:
                index = nodes[0][0]
        
        # found = False
        # for ix, node in enumerate(self.nodes):
        #     if node.name == name:
        #         found = True
        #         break
        # index = ix if found else -1
        
        return index


    def search_data(self, value, bottom_up=True):
        """ Search and return index of node in Tree with value in node data.

        If the value is in the data of more than one node, return the index
        of the deepest node (farthest from root), when
        ``bottom_up = True``, otherwise, return the index of the shallowest 
        (closest to root) node.

        If no such node is found with the specified value in its data, the
        value -1 is returned.
        
        Parameters
        ----------
        value
            Value in node data to search for.
        bottom_up : `bool`
            Indicate if the index of the shallowest or deepest node
            should be returned when more than one node has the same
            name. It is assumed that no two nodes at the same depth
            have the same name.

        Returns
        -------
        index : `int`
            Index of node in the tree. If node is not found, returns -1.
        """
        nodes = [(ix, node.depth()) for ix, node in enumerate(self.nodes) if node.contains(value)]
        if len(nodes) == 0:
            index = -1
        else:
            nodes = sorted(nodes, key=itemgetter(1))
            if bottom_up:
                index = nodes[-1][0]
            else:
                index = nodes[0][0]
        
        # found = False
        # for ix, node in enumerate(self.nodes):
        #     if node.name == name:
        #         found = True
        #         break
        # index = ix if found else -1
        
        return index

    
    def get_node(self, index):
        """ Return node by its index. """
        node = self.nodes[index]
        return node


    def _get_node_from_counter(self, counter):
        """ Search and return node in Tree by its counter ID.

        Assumes no nodes have the same counter ID.

        If no such node is found with the specified counter, `None`
        is returned.
        
        Parameters
        ----------
        counter : `int`
            Counter ID of node to search for.

        Returns
        -------
        node : `TreeNode`
            Node in the tree. If node is not found, returns `None`.
        """
        index = self._search_counter(counter)
        node = self.nodes[index]
        return node


    def _search_counter(self, counter):
        """ Search and return index of node in Tree by its counter ID.

        Assumes no nodes have the same couner.

        If no such node is found with the specified counter, the
        value -1 is returned.
        
        Parameters
        ----------
        counter : `int`
            Counter ID of node to search for.

        Returns
        -------
        index : `int`
            Index of node in the tree. If node is not found, returns -1.
        """
        nodes = [ix for ix, node in enumerate(self.nodes) if node._counter == counter]
        if len(nodes) == 0:
            index = -1
        elif len(nodes) > 1:
            raise ValueError("Unexpected number of nodes found with the same counter, expected to be unique.")
        else:
            index = nodes[0]
        
        return index
    
    
    def root(self):
        return self.root


    def max_depth(self):
        """ Return max depth of the tree. """
        return max([node.depth() for node in self.nodes])


    def all_data(self):
        """ Return sorted set of all data points in all nodes in the tree. """
        data = set()
        for node in self.nodes:
            data = data | set(node.data)

        data = sorted(data)
        return data


    def get_leaves_indices(self):
        """ Return indices of leaf nodes in the tree. """
        indices = [ix for ix, node in enumerate(self.nodes) if node.is_leaf()]
        return indices
    

    def get_leaves(self):
        """ Return leaf nodes in the tree. """
        leaves = [node for node in self.nodes if node.is_leaf()]
        return leaves


    def co_branch_indicator(self):
        """ Return binary symmetric `pandas.DataFrame` of size (num_data_points, num_data_points)
        where the (i,j)-th entry is 1 if the i-th and j-th data points are found in the same node
        (i.e., branch) and i is not the same data point as j. Otherwise, if i = j, or if the i-th
        and j-th data points are not found in the same node, the (i.j)-th entry is 0.
        """
        data_points = self.all_data()
        co_tracker = {k: {} for k in data_points}
        for ix, obs_a in enumerate(data_points):
            co_tracker[obs_a][obs_a] = 0
            node = self.get_node(self.search_data(obs_a))
            assert node.is_leaf(), "Expected leaf node."

            for obs_b in data_points[ix+1:]:
                if node.contains(obs_b):
                    co_tracker[obs_a][obs_b] = 1
                    co_tracker[obs_b][obs_a] = 1
                else:
                    co_tracker[obs_a][obs_b] = 0
                    co_tracker[obs_b][obs_a] = 0

        co_tracker = pd.DataFrame(co_tracker)
        # ensure symmetric and sorted:
        co_tracker = co_tracker.loc[data_points, data_points] 
        return co_tracker

        
def _compute_transitions(similarity=None, density_normalize: bool = True):
    """ Compute transition matrix.

    Parameters
    ----------
    similarity : `numpy.ndarray`, (n_observations, n_observations)
        Symmetric similarity measure (with 1s on the diagonal).
    density_normalize : `bool`
        The density rescaling of Coifman and Lafon (2006): Then only the
        geometry of the data matters, not the sampled density.

    Returns
    -------
    transitions_asym : `numpy.ndarray`, (n_observations, n_observations)
        Asymmetric Transition matrix.
    transitions_sym : `numpy.ndarray`, (n_observations, n_observations)
        Symmetric Transition matrix.
    
    Notes
    -----
    Code copied from `scanpy.neighbors`.
    """

    W = similarity.copy()
    
    # set diagonal to zero:
    np.fill_diagonal(W, 0.)
    
    # density normalization as of Coifman et al. (2005)
    # ensures that kernel matrix is independent of sampling density
    if density_normalize:
        # q[i] is an estimate for the sampling density at point i
        # it's also the degree of the underlying graph
        q = np.asarray(W.sum(axis=0))
        if not issparse(W):
            Q = np.diag(1.0 / q)
        else:
            Q = sp.sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
        K = Q @ W @ Q
    else:
        K = W

    # asym transitions
    # z[i] is the row sum of K
    z = np.asarray(K.sum(axis=0))
    if not issparse(K):
        Z = np.diag(1.0 / z)
    else:
        Z = sp.sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
    transitions_asym = Z @ K

    # sym transitions
    # z[i] is the square root of the row sum of K
    z = np.sqrt(np.asarray(K.sum(axis=0)))
    if not issparse(K):
        Z = np.diag(1.0 / z)
    else:
        Z = sp.sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
    transitions_sym = Z @ K @ Z

    # to compute first eigenvector phi0 if did density normalization (from matlab code):
    # D1_ = np.asarray(K.sum(axis=0))
    # phi0 = D1_ / np.sqrt(np.power(D1_, 2).sum())  # TODO: check if ever need this to be sparse

    return transitions_asym, transitions_sym


def compute_transitions(keeper, similarity_key, density_normalize: bool = True):
    """ Compute symmetric and asymmetric transition matrices and store in keeper.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object.
    similarity_key : `str`
        Reference key to the `numpy.ndarray`, (n_observations, n_observations)
        symmetric similarity measure (with 1s on the diagonal) stored in the similarities
        in the keeper.
    density_normalize : `bool`
        The density rescaling of Coifman and Lafon (2006): Then only the
        geometry of the data matters, not the sampled density.

    Returns
    -------
    Adds the following to the keeper.misc (with 0s on the diagonals):
        transitions_asym_{similarity_key} : `numpy.ndarray`, (n_observations, n_observations)
            Asymmetric Transition matrix.
        transitions_sym_{similarity_key} : `numpy.ndarray`, (n_observations, n_observations)
            Symmetric Transition matrix.
    
    Notes
    -----
    Code primarily copied from `scanpy.neighbors`.
    """
    similarity = keeper.similarities[similarity_key].data
    
    transitions_asym, transitions_sym = _compute_transitions(similarity=similarity,
                                                             density_normalize=density_normalize)

    asym_label = f"transitions_asym_{similarity_key}"
    sym_label = f"transitions_sym_{similarity_key}"
    # if label is not None:
    #     asym_label = "_".join([asym_label, label])
    #     sym_label = "_".join([sym_label, label])

    keeper.add_misc(transitions_asym, asym_label)
    keeper.add_misc(transitions_sym, sym_label)


def _dpt_from_augmented_sym_transitions(T):
    """ Return the diffusion pseudotime metric between observations,
    computed from the symmetric transitions.

    .. Note::

        - :math:`T` is the symmetric transition matrix
        - :math:`M(x,z) = \sum_{i=1}^{n-1} (\lambda_i * (1 - \lambda_i))\psi_i(x)\psi_i^T(z)`
        - :math:`dpt(x,z) = ||M(x, .) - M(y, .)||^2
    
    Parameters
    ----------
    T : `numpy.ndarray`, (n_observations, n_observations)
        Symmetric transitions.

    Returns
    -------
    dpt : `numpy.ndarray`, (n_observations, n_observations)
        Pairwise-observation Diffusion pseudotime distances.
    """
    evals, evecs = utl.compute_eigen(T, n_comps=0, sort="decrease")
    if np.abs(evals[0] - 1.) > 1e-3:
        raise ValueError(f"Largest eigenvalue is expected to be close to 1, found to be {np.round(evals[0], 4)}")
    if evals[1] > 1.0:
        raise ValueError(f"Expected second largest eigenvalue to be less than 1, found to be {np.round(evals[1], 4)}")
    EVALS = np.diag(evals[1:] / (1. - evals[1:]))
    M = evecs[:, 1:] @ EVALS @ evecs[:, 1:].T
    dpt = sp.spatial.distance.cdist(M, M)
    return dpt


def dpt_from_augmented_sym_transitions(keeper, key):
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
    T = keeper.misc[key]
    dpt = _dpt_from_augmented_sym_transitions(T)
    keeper.add_distance(dpt, f"dpt_from_{key}")
    

def root_max_ratio(keeper, key):
    """ Returns root index of observation that leads to the largest triangle..

    Parameters
    ----------
    keeper : `netflow.Keeper`
       The keeper object.
    key : `str`
        Reference key of distance in keeper used to determine the root.

    Returns
    -------
    root : `int`
        The root index.
    """
    d = keeper.distances[key].to_frame()
    max_d_pairs = d.idxmax()
    sum_max_d_pairs = max_d_pairs.to_frame('a').T.apply(lambda x: d.loc[x.name] + d.loc[x.loc['a']])
    tip3 = sum_max_d_pairs.idxmax()

    max_d_pairs.name = 'tip2'
    tip3.name = 'tip3'
    tips = pd.concat([max_d_pairs, tip3], axis=1)
    tri_ratio = tips.T.apply(lambda x: (d.loc[x.name, x.loc['tip3']] + d.loc[x.loc['tip2'], x.loc['tip3']]) / (d.loc[x.name, x.loc['tip2']]))

    # uncomment to get index of observation furthest from observation leading to largest triangular ratio:
    # root_x = tri_ratio.idxmax()
    # root_lbl = d.loc[root_x].idxmax()
    root_lbl = tri_ratio.idxmax() 
    root = keeper.observation_index(root_lbl)
    return root

    
class POSER:
    """
    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object that stores the distance matrix of size (n_observations, n_observations).
    key : `str`
        The label used to reference the distance matrix stored in ``keeper.distances``,
        of size (n_observations, n_observations).
    root : {`None`, `int`, 'density', 'density_inv', 'ratio'}
        The root. If `None`, 'density' is used.

        options
        -------
        - `int` : index of observation
        - 'density' : select observation with minimal distance-density
        - 'density_inv' : select observation with maximal distance-density
        - 'ratio' : select observation which leads to maximal triangular ratio distance
    root_as_tip : `bool`
        If `True`, force first tip as the root.
        Defaults to `False` following scanpy implementation.
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
    """
    def __init__(self, keeper, key, root=None, root_as_tip=False,
                 min_branch_size=5, choose_largest_segment=False,
                 flavor='haghverdi16', allow_kendall_tau_shift=False,
                 smooth_corr=True, brute=True, split=True, verbose=None):

        if verbose is not None:
            set_verbose(logger, verbose)

        self.distances = keeper.distances[key].data
        # self.num_observations = keeper.num_observations
        self.observation_labels = keeper.observation_labels

        if isinstance(min_branch_size, int):
            assert min_branch_size > 2, "As an integer, `min_branch_size` must be greater than 2."
            self.check_min_branch_size = lambda x: len(x)>self.min_branch_size
        elif isinstance(min_branch_size, float):
            assert 0. < min_branch_size < 1., "As a float, `min_branch_size` must satisfy 0 < `min_branch_size` < 1."
            self.check_min_branch_size = lambda x: (len(x) / self.distances.shape[0]) > self.min_branch_size
        else:
            raise TypeError("Unrecognized type for `min_branch_size`, must be an int or float.")
        self.min_branch_size = min_branch_size
        self.choose_largest_segment = choose_largest_segment
        self.flavor = flavor
        self.allow_kendall_tau_shift = allow_kendall_tau_shift
        self.smooth_corr = smooth_corr
        self.brute = brute
        self.split = split

        if root is None:
            root = "density"
        if isinstance(root, str):
            if root == 'density':
                root = keeper.distance_density_argmin(key)
            elif root == 'density_inv':
                root = keeper.distance_density_argmax(key)
            elif root == 'ratio':
                root = root_max_ratio(keeper, key)
            else:
                raise ValueError("Unrecognized method for determining root, expected to be one of ['density', 'ratio'].")
        elif isinstance(root, int):
            if (root < 0) or (keeper.num_observations - 1 < root):
                raise ValueError("Unrecognized value for root, expected to be the index of an observation in the keeper.")
        else:
            raise ValueError("Unrecognized value for root.")
        self.root = root

        self.pseudo_dist = None # was pseudotime
        self._set_pseudo_dist()

        self.tree = Tree()        
        # initialize the tree
        seg = np.arange(self.distances.shape[0], dtype=int)
        node = TreeNode(name=0, data=seg,
                        nonunique=False, unidentified=False,
                        branchable=True if self.check_min_branch_size(seg) else False,
                        is_trunk=False)
        # get tips:
        # tip_0 = root
        # get first tip (farthest from root)
        if root_as_tip:
            tip_0 = self.root
        else:
            tip_0 = np.argmax(self.distances[self.root])
        tip_1 = np.argmax(self.distances[tip_0])
        node.tips = np.array([tip_0, tip_1])
        self.tree.insert(node)
        self.branched_ordering = []  # [node]
        
        self.unidentified_points = set()

        
    def _set_pseudo_dist(self):
        """ Return pseudo-distance with respect to root point. """
        self.pseudo_dist = self.distances[self.root].copy()

        self.pseudo_dist /= np.max(self.pseudo_dist[self.pseudo_dist < np.inf])

    def select_segment(self):
        """ Select segment with most distant triangulated data point.

        Returns
        -------
        node : `TreeNode`
            The node corresponding to the selected segment.
            If no nodes are branchable, returns None.
        """        
        selected_segs = []
        segs = self.tree.get_leaves()
        
        for iseg, seg in enumerate(segs):
            
            if not seg.branchable:
                continue

            if seg.score is not None:
                selected_segs.append(seg)
                continue

            # restrict distance matrix to points in segment
            Dseg = self.distances[np.ix_(seg.data, seg.data)]

            if len(seg.tips) == 2:

                third_maximizer = None
                if seg.is_trunk:
                    # check that no tip connects with tip of another seg
                    for jseg in range(len(segs)):
                        if jseg != iseg:
                            for itip in range(2):
                                if (
                                        self.distances[
                                            segs[jseg].tips[1], seg.tips[itip]
                                        ]
                                        < 0.5
                                        * self.distances[
                                            seg.tips[~itip], seg.tips[itip]
                                        ]
                                ):
                                    third_maximizer = itip

                # map the global position to the position within the segment
                allindices = np.arange(self.distances.shape[0], dtype=int)
                # find the third point on the seg that has maximal added distance from the two tip points:
                tips = [np.where(allindices[seg.data] == tip)[0][0] for tip in seg.tips] # local index of tips in the seg
                # find the third point on the seg that has maximal added distance from the two tip points:
                dseg = Dseg[tips[0]] + Dseg[tips[1]]
                
                if not np.isfinite(dseg).any():
                    seg.branchable = False
                    continue
                third_tip = np.argmax(dseg)

                if third_maximizer is not None:
                    logger.warning(f"TODO: THIRD MAXIMIZER IS NOT NONE... IS THIS CORRECT???")
                    # find a fourth point that has maximal distance to all three
                    dseg += Dseg[third_tip]
                    fourth_tip = np.argmax(dseg)
                    # should it be >>> if fourth_tip != tips[third_maximizer] and fourth_tip != third_tip: ... and >>> tips[third_maximizer] = fourth_tip ???
                    if fourth_tip != tips[0] and fourth_tip != third_tip: 
                        # dseg -= Dseg[tips[1]] # RE CHANGED TO COMPUTE BEFORE UPDATING TIP
                        # logger.msg(f"TODO: tip1 changed from {tips[1]} to {fourth_tip}...")
                        tips[1] = fourth_tip
                        dseg -= Dseg[tips[1]] # OLD WAY COMPUTED AFTER UPDATING TIP --- should it be dseg += Dseg[tips[1]]?
                        # dseg = Dseg[tips[0]] + Dseg[tips[1]] # RE ADDED SECOND NEW WAY OPTION:
                    else:
                        dseg -= Dseg[third_tip]
                tips3 = np.append(tips, third_tip)

                # update third tip in global coordinates NOTE: double check that this is correct
                # seg.tips.append(seg.data[third_tip])
                # seg.tips = np.append(seg.tips, seg.data[third_tip])
                seg.tips = seg.data[tips3]
                
            elif len(seg.tips) == 3:
                # map the global position to the position within the segment
                allindices = np.arange(self.distances.shape[0], dtype=int)
                # find the third point on the seg that has maximal added distance from the two tip points:
                tips3 = np.array([np.where(allindices[seg.data] == tip)[0][0] for tip in seg.tips]) # local index of tips in the seg
            else:
                raise AssertionError("Unexpected number of tips.")
            
            # compute the score as ratio of the added distance to the third tip,
            # to what it would be if it were on the straight line between the
            # two first tips, given by Dseg[tips[:2]]
            # if we did not normalize, there would be a danger of simply
            # assigning the highest score to the longest segment
            score = dseg[tips3[2]] / Dseg[tips3[0], tips3[1]]
            score = len(seg.data) if self.choose_largest_segment else score # simply the number of points
            seg.score = score
            if not self.check_min_branch_size(seg.data):
                seg.branchable = False
            else:
                selected_segs.append(seg)


        if len(selected_segs) > 0:
            selected_seg = max(selected_segs, key=lambda x: x.score)
        else:
            selected_seg = None
        return selected_seg


    def _kendall_tau_add(self, len_old: int, diff_pos: int, tau_old: float):
        """ Compute Kendall tau delta.

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
        a, b : `numpy.ndarray`
            One dimensional sequences.
        i : `int`
            Index for splitting ``a`` and ``b``.

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
        a, b : `numpy.ndarray`
            One dimensional sequences.
        min_length : `int`, (``min_length > 0``)
            Minimum number of data points automatically included in branch.

        Returns
        -------
        imax : `int`
            Splitting index according to above description.
        """
        if a.size != b.size:
            raise ValueError('a and b need to have the same size')
        if a.ndim != b.ndim != 1:
            raise ValueError('a and b need to be one-dimensional arrays')

        n = a.size
        idx_range = np.arange(min_length, a.size - min_length - 1, dtype=int)
        corr_coeff = np.zeros(idx_range.size)
        pos_old = sp.stats.kendalltau(a[:min_length], b[:min_length])[0]
        neg_old = sp.stats.kendalltau(a[min_length:], b[min_length:])[0]

        for ii, i in enumerate(idx_range):
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

        # TODO: add smoothing to corr_coeff before selecting max index
        if corr_coeff.size == 0:
            iimax = 0
            corr_coeff_max = 0.
            
        else:
            if self.smooth_corr:
                # logger.msg(f"corr before smoothing: {corr_coeff}")
                # logger.msg(f"imax before smoothing: {np.argmax(corr_coeff)}")
                corr_coeff = utl.gauss_conv(corr_coeff, window_size=5, smoothness=2.5)
                # logger.msg(f"corr after smoothing: {corr_coeff}")
                # logger.msg(f"imax after smoothing: {np.argmax(corr_coeff)}")

            iimax = np.argmax(corr_coeff)
            corr_coeff_max = corr_coeff[iimax]                
            
        # iimax = 0 if corr_coeff.size == 0 else np.argmax(corr_coeff)
        imax = min_length + iimax
        # corr_coeff_max = 0. if corr_coeff.size == 0 else corr_coeff[iimax]
        
        # RE END MODIFIED
        if corr_coeff_max < 0.3:
            logger.info('    is root itself, never obtain significant correlation')
        return imax


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
            The distance matrix restricted to segment.
        tips
            The three tip points in local coordinates to the segment.
            They form a 'triangle' that contains the data.

        Returns
        -------
        branch : `numpy.ndarray` (k,)
            Segment obtained from "splitting away the first tip data point",
            where k is the number of data points in the branch.
        """
        # sort distance from first tip point
        # then the sequence of distances Dseg[tips[0]][idcs] increases
        idcs = np.argsort(Dseg[tips[0]])

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

        # first new segment: all points until, but excluding the branching point
        # increasing the following slightly from imax is a more conservative choice
        # as the criterion based on normalized distances, which follows below,
        # is less stable
        if imax > 0.95 * len(idcs):
            logger.info('segment is more than 95\% correlated.')
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

            
        branch = idcs[:ibranch]
        return branch
        
        
    def _detect_branching_single_haghverdi16(self, Dseg: np.ndarray, tips: np.ndarray):
        """Detect branching on given segment.

        Parameters
        ----------
        Dseg
        tips

        Returns
        -------
        ssegs : `list[numpy.ndarray]`
            The branched segments.
        """
        # compute branchings using different starting points the first index of
        # tips is the starting point for the other two, the order does not
        # matter
        ssegs = [] 
        # permutations of tip cells
        if self.split:
            ps = [
                [0, 1, 2],  # start by computing distances from the first tip
                [1, 2, 0],  #             -"-                       second tip
                [2, 0, 1],  #             -"-                       third tip
            ]
        else:
            # ps = [[2, 0, 1]]
            ps = [[1, 2, 0]]

        for i, p in enumerate(ps):
            # logger.warning(f"*iterating p: i = {i}")
            ssegs.append(self.__detect_branching_haghverdi16(Dseg, tips[p]))

        
        # if not self.split:
        #     # add main portion of branch
        #     ssegs.append(list(set(range(Dseg.shape[0])) - set(ssegs[0])))
            
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
    

    def _detect_branch(self, Dseg: np.ndarray, tips: np.ndarray):
        """ Detect branching on given segment.

        If ``self.split``, Call function __detect_branching three times for all three orderings 
        of tips. Points that do not belong to the same segment in all three
        orderings are assigned to a fourth segment. The latter is, by Haghverdi
        et al. (2016) referred to as 'undecided points' (which make up the so-called 'trunk').
        Otherwise, only the branch off the main segment is detected from the third tip.


        If ``split`` and ``flavor == 'haghverdi16'`` : If any of the branches from the three
        consist of zero unique observations, resulting in an empty branch, the process is
        terminated and no branching is performed on the current segment.

        ..note::
        
          In practice, this has only occurred in small segments. If finer resolution partitioning
          is desired, this may be changed in a future release to account for an offshoot resulting
          in two branches (and possibly a trunk with undecided points). 


        Parameters
        ----------
        Dseg
            The distance matrix restricted to segment.
        tips : `numpy.ndarray`
            Tips in local coordinates relative to the segment.

        Returns
        -------
        ssegs : list[list]
            Stores branched segments in local coordinates.
        ssegs_tips : list[list]
            Stores all tip points in local coordinates for the segments in ``ssegs``.
        ssegs_connects list[list]
            A list of k lists, where k is the number of inter-segment connections between
            the segments in ``ssegs``. Each entry is a 2-list of the form
            [[index of first seg in ``ssegs``, index of second seg in ``ssegs``], 
            [source observation, target observation]].
        trunk : `int`
            Index of segment in `ssegs` that all other segments in ``ssegs`` stem from.
        trunk_undecided : `bool`
            If True, the trunk are made up of undecided points.
        unidentified_points : `set`            
            Points in local coordinates relative to the segment before branching
            that are not associated with any branch after splitting.        
        """
        if self.flavor == 'haghverdi16':
            ssegs = self._detect_branching_single_haghverdi16(Dseg, tips)  # correlation
        elif self.flavor == 'wolf17_tri':
            ssegs = self._detect_branching_single_wolf17_tri(Dseg, tips)  # closer in distance to tip than other two tips
        elif self.flavor == 'wolf17_bi' or self.flavor == 'wolf17_bi_un': 
            ssegs = self._detect_branching_single_wolf17_bi(seg, Dseg)
        else:
            raise ValueError(
                '`flavor` needs to be in {"haghverdi16", "wolf17_tri", "wolf17_bi, "wolf17_bi_un""}.'
            )
        
        trunk_undecided = False
        if self.split or (self.flavor != 'haghverdi16'):
            # make sure that each data point has a unique association with a segment
            masks = np.zeros((len(ssegs), Dseg.shape[0]), dtype=bool)
            for iseg, seg in enumerate(ssegs):
                masks[iseg][seg] = True                
            nonunique = np.sum(masks, axis=0) > 1
            logger.info(f"* {nonunique.sum()} nonunique points.")

            # RE START MODIFIED - to account for points not associated with any branch
            unidentified = np.sum(masks, axis=0) == 0
            logger.info(f"* {unidentified.sum()} unidentified points.")        
            # RE END MODIFIED

            # RE START MODIFIED - uncomment to match how original paper defines unique
            # if len(ssegs) == 3:
            #     allbranches = np.sum(masks, axis=0) == len(ssegs)
            #     twobranches = np.sum(masks[1:, :], axis=0) == len(ssegs) - 1
            #     nonunique = allbranches | twobranches
            # else:
            #     nonunique = np.sum(masks, axis=0) > 1
            # # RE END MODIFIED
            ssegs = []

            for iseg, mask in enumerate(masks):
                mask[nonunique] = False
                # RE START MODIFIED  - IN EVENT THAT THE SEG HAD NO UNIQUE MEMBERS AND IS NOW EMPTY:
                # ssegs.append(np.arange(Dseg.shape[0], dtype=int)[mask])

                newseg = np.arange(Dseg.shape[0], dtype=int)[mask]
                if newseg.shape[0] == 0:
                    # logger.debug("Unique segment is empty, continuing branching without it.")
                    # continue # TODO - future release switch return to continue for finer resolution partitioning
                    logger.debug("Unique segment is empty, removing from consideration and no branching performed.")                    
                    return None

                ssegs.append(newseg)
                # RE END MODIFIED
            if len(ssegs) <= 1:  # RE MODIFIED HERE
                return None

            # compute new tips within new segments (relative to the full segment)
            ssegs_tips = []
            for newseg, tip in zip(ssegs, tips):
                # logger.warning(f"*** tip = {tip}")
                newseg_tips = self.identify_local_tips(Dseg, newseg, tip)                
                ssegs_tips.append(newseg_tips)

            # RE MODIFIED END

            # RE START MODIFIED - to account for points not associated with any branch
            if self.brute:
                nonunique = nonunique | unidentified
            # RE END MODIFIED
            undecided_cells = np.arange(Dseg.shape[0], dtype=int)[nonunique]

            # RE START MODIFIED - to account for points not associated with any branch
            unidentified_points = np.arange(Dseg.shape[0], dtype=int)[unidentified]
            # RE END MODIFIED

            
            if len(undecided_cells) > 0:
                ssegs.append(undecided_cells)
                trunk = len(ssegs) - 1
                trunk_undecided = True
                # establish the connecting points with the other segments
                ssegs_connects = [] # [[]]*len(ssegs) # [[], [], [], []]
                for inewseg, newseg_tips in enumerate(ssegs_tips):
                    reference_point = newseg_tips[0]
                    # closest (undecided) cell to the new segment tip within undecided cells
                    closest_cell_a = undecided_cells[
                        np.argmin(Dseg[reference_point][undecided_cells])
                    ]
                    # RE START MODIFIED
                    # # ssegs_connects[inewseg].append(closest_cell)
                    # ssegs_connects[-1].append(closest_cell_a)
                    # RE END MODIFIED
                    # closest cell to the undecided cells within new segment
                    closest_cell_b = ssegs[inewseg][
                        np.argmin(Dseg[closest_cell_a][ssegs[inewseg]])
                    ]
                    # RE START MODIFIED
                    # # ssegs_connects[-1].append(closest_cell)
                    # ssegs_connects[inewseg].append(closest_cell)
                    # RE END MODIFIED

                    ssegs_connects.append([[trunk, inewseg], [closest_cell_a, closest_cell_b]])

                # also compute tips for the undecided cells
                tip_0 = undecided_cells[
                    np.argmax(Dseg[undecided_cells[0]][undecided_cells])
                ]

                tip_1 = undecided_cells[np.argmax(Dseg[tip_0][undecided_cells])]
                ssegs_tips.append([tip_0, tip_1])
                # RE START MODIFIED
                # trunk = 3
                # trunk = len(ssegs) - 1
                # logger.warning(f"trunk = {trunk}")
                # RE END MODIFIED
                # ssegs_adjacency = [[trunk]]*trunk + [list(range(trunk))] # [[3], [3], [3], [0, 1, 2]]
                
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
                # ssegs_adjacency = [
                #     [trunk] if i != trunk else [j for j in range(3) if j != trunk]
                #     for i in range(3)
                # ]

                # RE START MODIFIED
                # # ssegs_connects = [
                # #     [closest_points[i, trunk]]
                # #     if i != trunk
                # #     else [closest_points[trunk, j] for j in range(3) if j != trunk]
                # #     for i in range(3)
                # # ]
                # ssegs_connects = [
                #     [closest_points[trunk, i]]
                #     if i != trunk
                #     else [closest_points[j, trunk] for j in range(3) if j != trunk]
                #     for i in range(3)
                # ]
                ssegs_connects = [[[trunk, i],
                                   [closest_points[trunk, i],
                                    closest_points[i, trunk]]] for i in range(3) if i != trunk]

                # RE END MODIFIED
            else:
                trunk = 0
                # ssegs_adjacency = [[1], [0]]
                reference_point_in_0 = ssegs_tips[0][0]
                closest_point_in_1 = ssegs[1][
                    np.argmin(Dseg[reference_point_in_0][ssegs[1]])
                ]
                reference_point_in_1 = closest_point_in_1  # ssegs_tips[1][0]
                closest_point_in_0 = ssegs[0][
                    np.argmin(Dseg[reference_point_in_1][ssegs[0]])
                ]
                # RE START MODIFIED
                # # ssegs_connects = [[closest_point_in_1], [closest_point_in_0]]
                # ssegs_connects = [[closest_point_in_0], [closest_point_in_1]]
                ssegs_connects = [[[1, 0], [closest_point_in_0, closest_point_in_1]]]
                # RE END MODIFIED

        else:
            if len(ssegs) < 1:
                return None
                
            branch_seg = ssegs[0]
            main_seg = [k for k in range(Dseg.shape[0]) if k not in branch_seg]
            if len(main_seg) == 0:
                return None
            ssegs = [main_seg, branch_seg]

            # compute new tips within new segments
            # ssegs_tips = [list(tips[:2])]
            # branch_tips = self.identify_local_tips(Dseg, branch_seg, tips[2])
            # ssegs_tips.append(branch_tips)
            ssegs_tips = [self.identify_local_tips(Dseg,
                                                   ss,
                                                   tt) for ss, tt in zip([main_seg,
                                                                          branch_seg],
                                                                         [tips[0],
                                                                          tips[1], # tips[2],
                                                                          ])]
            

            # ssegs_connects = [[]]*len(ssegs) # [[], []]
            # point in branch closest to the main segment
            reference_point = tips[0]
            closest_cell_a = branch_seg[
                np.argmin(Dseg[reference_point][branch_seg])
            ]
            # RE START MODIFIED
            # # ssegs_connects[0].append(closest_cell)
            # ssegs_connects[-1].append(closest_cell_a)
            # RE END MODIFIED
            # point in main segment closest to the identified branch point
            closest_cell_b = main_seg[
                np.argmin(Dseg[closest_cell_a][main_seg])
            ]
            # RE START MODIFIED
            # # ssegs_connects[-1].append(closest_cell)
            # ssegs_connects[0].append(closest_cell_b)
            # RE END MODIFIED

            trunk = 0
            # ssegs_adjacency = [[1], [0]]
            ssegs_connects = [[[0, 1], [closest_cell_a, closest_cell_b]]]

            unidentified_points = []
        
        # return ssegs, ssegs_tips, ssegs_adjacency, ssegs_connects, \
        return ssegs, ssegs_tips, ssegs_connects, \
            trunk, trunk_undecided, unidentified_points
            

    def identify_local_tips(self, Dseg, newseg, tip):
        """ Identify new tips within the new segments

        Parameters
        ----------
        newseg : `list`
            New segment.
        tip : `int`
            Local index of the first tip, with respect to the original segment that determinned
            ``Dseg`` before the split.

        Returns
        -------
        tips : `np.ndarray` (2,)
            First and second tip indices in local coordinates relative to the original segment,
            before it was branched.
        """
        # RE MODIFIED START
        # if len(np.flatnonzero(newseg)) <= 1:
        #     logger.warning(f'detected group with only {len(np.flatnonzero(newseg))} data points')
        # secondtip = newseg[np.argmax(Dseg[tip][newseg])]
        # ssegs_tips.append([tip, secondtip]) # RE: SHOULD BE CHANGED

        secondtip = newseg[np.argmax(Dseg[tip][newseg])]
        firsttip = tip

        if len(np.flatnonzero(newseg)) <= 1:
            logger.info(f'detected group with only {len(np.flatnonzero(newseg))} data points')
        if firsttip not in set(newseg):
            new_firsttip = newseg[np.argmin(Dseg[tip][newseg])]
            logger.info(f'tip is no longer in the unique branched sub-segment, update tip to its nearest point in the new segment: {firsttip} -> {new_firsttip}')
            firsttip = new_firsttip

        tips = np.array([firsttip, secondtip])

        return tips

    
    def detect_branching(self, node):
        """ Detect branching on a given segment and update TreeNode parameters in place.

        Parameters
        ----------
        node : `TreeNode`
            The node of the segment to be branched.

        Returns
        -------
        updated : `bool`
            `True` if segment is successfully branched, `False` otherwise.
        """
        # seg_node = self.tree.get_node(self.tree.search(iseg, bottom_up=True))
        allindices = np.arange(self.distances.shape[0], dtype=int)
        Dseg = self.distances[np.ix_(node.data, node.data)]        
        tips3 = [np.where(allindices[node.data] == tip)[0][0] for tip in node.tips] # local index of tips in the seg
        tips3 = np.array(tips3).astype(int)
        
        # given the three tip points and the distance matrix detect the
        # branching on the segment, return the list ssegs of segments that
        # are defined by splitting this segment

        result = self._detect_branch(Dseg, tips3)
        if result is None: # RE ADDED THIS CONDITION
            logger.info(f"No unique branch detected - removed from consideration.")
            node.branchable = False
            updated = False
        else:
            # ssegs, ssegs_tips, ssegs_adjacency, ssegs_connects, \
            ssegs, ssegs_tips, ssegs_connects, \
                trunk, trunk_undecided, unidentified = result

            # map back to global indices
            unidentified = node.data[unidentified] # record data points not associated with any branch
            logger.info(f"* {len(unidentified)} unclaimed points.")

            for iseg_new, seg_new in enumerate(ssegs):
                ssegs[iseg_new] = node.data[seg_new]
                ssegs_tips[iseg_new] = node.data[ssegs_tips[iseg_new]] 
                # logger.warning(f"*** {ssegs_connects[iseg_new][-1]}")
                # ssegs_connects[iseg_new] = list(node.data[ssegs_connects[iseg_new]])

            for sc in ssegs_connects:
                sc[-1] = list(node.data[sc[-1]])

            # RE ADDED - update unidentified points 
            self.unidentified_points = self.unidentified_points | set(unidentified)

            # insert trunk/undecided_cells with same name
            cur_trunk_node = TreeNode(name=node.name, data=ssegs[trunk],
                                      parent=node, nonunique=True, unidentified=False,
                                      branchable=True if self.check_min_branch_size(ssegs[trunk]) else False,
                                      is_trunk=trunk_undecided,
                                      )
            cur_trunk_node.tips = ssegs_tips[trunk]
            self.tree.insert(cur_trunk_node,
                             parent=node)

            
            # append other segments
            cur_nodes = []
            # num_segs = len(self.tree.get_leaves())
            num_segs = max([k.name for k in self.tree.get_leaves()])
            for ix, (ixseg, ixseg_tips) in enumerate(zip(ssegs, ssegs_tips)):
                if ix != trunk:
                    cur_node = TreeNode(name=num_segs+1, # ix+num_segs,
                                        data=ixseg, parent=node,
                                        nonunique=False, unidentified=False,
                                        branchable=True if self.check_min_branch_size(ixseg) else False,
                                        is_trunk=False,
                                        )
                    num_segs += 1
                    cur_node.tips = ixseg_tips
                    self.tree.insert(cur_node, parent=node)
                    cur_nodes.append(cur_node._counter)
                else:
                    cur_nodes.append(cur_trunk_node._counter)
            
            # update edges between nodes
            # for i, adj in enumerate(ssegs_adjacency):
            #     self.tree.node_adjacency[cur_nodes[i]] += [cur_nodes[j] for j in adj]
            for ix, sc in enumerate(ssegs_connects):
                sc[0] = list([cur_nodes[i] for i in sc[0]])
            self.tree.node_connection += ssegs_connects
            # for i, con in enumerate(ssegs_connects):
            #     self.tree.node_connection[cur_nodes[i]] += con

            # NOTE: USED TO HAVE CONDITION FOR NOT HAGHVERDI16

            updated = True
            self.branched_ordering.append(node)

        return updated


    def single_branch(self, until_branched=False):
        """ Perform single branching in place.

        Parameters
        ----------
        until_branched : `bool`
            If `True`, iteratively find segment to branch and perform branching
            until a segement is successfully branched or no branchable segments
            remain. Otherwise, if `False`, attempt to perform branching only once 
            on the next potentially branchable segment.

        Returns
        -------
        branched_flag : `bool`
            Indicates if branching was successfully completed.
        """
        branched_flag = False # not until_branched # stop when true

        while not branched_flag:
            
                
            node = self.select_segment()
            logger.info(f"* selected node = {node}")

            if node is None:
                logger.warning("No branchable segments remain -- partitioning converged.")
                # branched_flag = True
                break                
            else:
                branched_flag = self.detect_branching(node)

            if not until_branched:
                # branched_flag = True
                break

        return branched_flag
                

    def detect_branches(self, n_branches, until_branched=False):
        """ Detect up to ``n_branches`` branchings and update tree in place.

        Parameters
        ----------
        n_branches : `int`
            Number of branch splits to perform (``n_branches > 0``).
        until_branched : `bool`
            If `True`, iteratively find segment to branch and perform branching
            until a segement is successfully branched or no branchable segments
            remain. Otherwise, if `False`, attempt to perform branching only once 
            on the next potentially branchable segment.

            ..note::

              This is only applicable when branching is being performed. If previous
              iterations of branching has already been performed, it is not possible to
              identify the number of iterations where no branching was performed.
        """
        if n_branches > len(self.branched_ordering):
            for ibranch in range(len(self.branched_ordering), n_branches):
                logger.info(f"*ibranch = {ibranch}")

                branched_flag = self.single_branch(until_branched=until_branched)
                # logger.warning(f"* was branched = {branched_flag}")
                if not branched_flag:
                    logger.info("No further branching occured at this iteration.")
        else:
            logger.warning(f"`n_branches` branchings have already occurred. No further branching is performed.")
                
            
    def extract_branchings(self, n_branches):
        """ Extract POSE from up to `n_branches` branchings

        Parameters
        ----------
        n_branches : `int`
            Number of branches to look for (``n_branches > 0``).

        Returns
        -------
        tree : `Tree`
            The tree with up to `n_branches` branchings.
            If ``n_branches`` is more than or equal to the number of
            branchings in the tree, the original tree is returned.
            Otherwise, a reduced tree is returned.
        """
        if n_branches >= len(self.branched_ordering):
            tree = self.tree
        else:
            n_branches = min(n_branches, len(self.branched_ordering))
            tree = Tree()

            node = self.branched_ordering[0]
            new_node = TreeNode(name=node.name, data=node.data,
                                nonunique=node.nonunique, unidentified=node.unidentified,
                                branchable=node.branchable, is_trunk=node.is_trunk)
            tree.insert(new_node)
            for ix, node in enumerate(self.branched_orderings[:n_branches]):
                parent = tree.get_node_from_name(node.name, bottom_up=True)
                for child in sorted(node.children, key=lambda x: x._counter):
                    new_node = TreeNode(name=child.name, data=child.data, parent=parent,
                                        nonunique=child.nonunique, unidentified=child.unidentified,
                                        branchable=child.branchable, is_trunk=child.is_trunk)
                    tree.insert(new_node, parent=parent)

            # mx = max([k._counter for k in tree.nodes])
            # sub_adj = lambda x: [a for a in self.tree.node_adjacency[x] if a <= mx]            
            # tree.node_adjacency = {counter: sub_adj[counter] for counter in range(mx+1)}
            # tree.node_adjacency = self.tree.node_adjacency
            # tree.node_connection = self.tree.node_connection

        return tree
            
            
            
    def branchings_segments(self, n_branches, until_branched=False, annotate=True):
        """ Detect up to `n_branches` branches and partition the data into corresponding segments.
        
        Parameters
        ----------
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
            If `True`, annotate nodes with root and tips.

        Returns
        -------
        G : `nx.Graph`
            The graph of the resulting POSE.
        """
        self.detect_branches(n_branches, until_branched=until_branched)

        tree = self.extract_branchings(n_branches)

        leaves = tree.get_leaves()
        segs = {}

        for node in leaves:
            tips = np.array(node.tips)
            seg = np.array(node.data)
            segs[node._counter] = {'name': node.name,
                                   'tips': tips[np.argsort(self.pseudo_dist[node.tips])],
                                   'seg': seg[np.argsort(self.pseudo_dist[node.data])],
                                   'undecided' : node.is_trunk,
                                   }

        G = self._construct_topology(segs, annotate=annotate)
        if annotate:
            nx.set_node_attributes(G, {k: 'Yes' if k==self.root else 'No' for k in G},
                                   name='is_root')
            nx.set_node_attributes(G, {k: vl for k, vl in enumerate(self.pseudo_dist)},
                           name='pseudo-distance from root')
        
        return G


    def _construct_topology(self, segs, annotate=True):
        """ Construct POSE connections between data points.

        Parameters
        ----------
        segs : `dict`
            The banched segments indexed by the node's unique identifier
        annotate : `bool`
            If `True`, annotate edges with edge origin and distance.

        Returns
        -------
        G : `networkx.Graph`
            Graph where each node is a data point and edges reflect connections between them.
            If ``annotate`` is `True`, the following annotations are added:
        
            Edges have attributes {'connection' : (str) 'intra-branch' or 'inter-branch'}
            Nodes have attributes

            .. code-block:: py

               {'branch' : (`int`) -1, 0, 1, ... where -1 indicates the data point was not identified with a branch,
                                   'undecided' : (bool) True if the data point is part of a trunk and False otherwise,
                                   'name' : (str) Original label if given data was a dataframe, otherwise the same as the node id,
                'unidentified' : (0 or 1) 1 if data point was ever not associated with any branch upon split,
                                   0 otherwise.,
               }        
        """
        G = nx.Graph()

        # add missing nodes as island nodes
        seg_nodes = set(itertools.chain(*[n['seg'] for n in segs.values()]))
        missing_nodes = set(range(self.distances.shape[0])) - seg_nodes
        G.add_nodes_from(list(missing_nodes), branch=-1)
        if annotate:
            nx.set_node_attributes(G, {k: False for k in missing_nodes}, name='undecided')

        # add segments and intra-segemnt edges
        for ix, seg in segs.items():
            if len(seg['seg']) == 1:
                G.add_node(seg['seg'][0])
            else:
                G.add_edges_from(zip(seg['seg'], seg['seg'][1:]), connection='intra-branch')
            if annotate:
                nx.set_node_attributes(G, {v: {'branch': seg['name'],
                                               'undecided': seg['undecided'],
                                               } for v in seg['seg']})

        # add inter-segment edges
        mx = max(segs.keys())
        # segs_adjacency = sp.sparse.lil_matrix((mx, mx), dtype=float)
        # segs_connection = sp.sparse.lil_matrix((mx, mx), dtype=float)
        inter_edges = [k[1] for k in self.tree.node_connection if all([ix <= mx for ix in k[0]])]
        G.add_edges_from(inter_edges, connection='inter-branch')

        # add node names
        nx.set_node_attributes(G, dict(zip(range(self.distances.shape[0]),
                                           self.observation_labels)), name='name')


        # add if node was ever an unidentified point:
        if annotate:
            nx.set_node_attributes(G, {v: 1 if v in self.unidentified_points else 0 for v in G},
                                   name='unidentified')

            nx.set_edge_attributes(G, {ee: self.distances[ee[0], ee[1]] for ee in G.edges()},
                                   name="distance")
            nx.set_edge_attributes(G, {ee: np.max(self.distances) + 1e-6 - self.distances[ee[0],
                                                                                          ee[1]] for ee in G.edges()},
                                   name="inverted_distance")

        return G


    def construct_pose_nn_topology(self, G, annotate=True):
        """ Add nearest neighbor (nn) edges to POSE topology.

        Parameters
        ----------
        G : `networkx.Graph`
            Nearest-neighbor edges are added to a copy of the POSE graph.
        annotate : `bool`
            If `True`, annotate edges.

        Returns
        -------
        Gnn : `networkx.Graph`
            The updated graph with nearest neighbor edges.
            If ``annotate`` is `True`, edge attribute "edge_origin"
            is added with the possible values :

            - "POSE" : for edges in the original graph that are not nearest neighbor edges
            - "NN" : for nearest neighbor edges that were not in the original graph
            - "POSE + NN" : for edges in the original graph that are also nearest neighbor edges
        """
        Gnn = G.copy()
        d = self.distances
        # d = d + (np.max(d)+1e-3)*np.eye(*d.shape)
        # nn = np.argmin(d, axis=0)
        nn = np.argpartition(d, 1, axis=1)[:, 1]
        if annotate:
            nn_edges = [tuple(sorted([i,j])) for i, j in zip(range(d.shape[0]), nn)]
            nn_edges = list(set(nn_edges))

            pose_edges = list(set([tuple(sorted(ee)) for ee in Gnn.edges()]))

            nn_unique_edges = list(set(nn_edges) - set(pose_edges))
            pose_unique_edges = list(set(pose_edges) - set(nn_edges))
            nn_pose_edges = list(set(nn_edges) & set(pose_edges))

            if len(nn_unique_edges) + len(nn_pose_edges) != len(nn_edges):
                raise AssertionError("Unexpected number of nearest-neighbor edges.")

            Gnn.add_edges_from(nn_unique_edges)

            nx.set_edge_attributes(Gnn, {**{ee: "POSE + NN" for ee in nn_pose_edges},
                                         **{ee: "NN" for ee in nn_unique_edges},
                                         **{ee: "POSE" for ee in pose_unique_edges}},
                                   name="edge_origin")

            nx.set_edge_attributes(Gnn, {ee: "intra-branch" if Gnn.nodes[ee[0]]['branch'] == Gnn.nodes[ee[1]]['branch'] else "inter-branch" for ee in nn_unique_edges},
                                   name="connection")

            nx.set_edge_attributes(Gnn, {ee: self.distances[ee[0], ee[1]] for ee in Gnn.edges()}, name="distance")
            nx.set_edge_attributes(Gnn, {ee: np.max(self.distances) + 1e-6 - self.distances[ee[0], ee[1]] for ee in Gnn.edges()}, name="inverted_distance")

        else:
            nn_edges = [tuple([i,j]) for i, j in zip(range(d.shape[0]), nn)]
            Gnn.add_edges_from(nn_edges)

        return Gnn


    def construct_pose_mst_topology(self, G):
        """ Construct pose topology with minimum spanning tree (MST) edges. 

        Parameters
        ----------
        G : `networkx.Graph`
            The POSE graph. MST edges are added to a copy of the graph.

        Returns
        -------
        Gmst : `networkx.Graph`
            The updated graph with MST edges and edge attribute "edge_origin"
            with the possible values :

            - "POSE" : for edges in the original graph that are not MST edges
            - "MST" : for MST edges that were not in the original graph
            - "POSE + MST" : for edges in the original graph that are also MST edges
        """
        G = G.copy()
        
        d = pd.DataFrame(self.distances)
        edgelist = utl.stack_triu_(d)
        edgelist.name = 'weight'
        edgelist = edgelist.reset_index().rename(columns={'level_0': 'source', 'level_1': 'target'})
        Gfull = nx.from_pandas_edgelist(edgelist, edge_attr='weight')
        Gmst = nx.minimum_spanning_tree(Gfull)

        mst_edges = [tuple(sorted([i,j])) for i,j in Gmst.edges()]
        mst_edges = list(set(mst_edges))

        pose_edges = list(set([tuple(sorted(ee)) for ee in G.edges()]))

        mst_unique_edges = list(set(mst_edges) - set(pose_edges))
        pose_unique_edges = list(set(pose_edges) - set(mst_edges))
        mst_pose_edges = list(set(mst_edges) & set(pose_edges))

        if len(mst_unique_edges) + len(mst_pose_edges) != len(mst_edges):
            raise AssertionError("Unexpected number of MST edges.")

        G.add_edges_from(mst_unique_edges)

        nx.set_edge_attributes(G, {**{ee: "POSE + MST" for ee in mst_pose_edges},
                                   **{ee: "MST" for ee in mst_unique_edges},
                                   **{ee: "POSE" for ee in pose_unique_edges}},
                               name="edge_origin")

        nx.set_edge_attributes(G, {ee: self.distances[ee[0], ee[1]] for ee in G.edges()}, name="distance")
        nx.set_edge_attributes(G, {ee: np.max(self.distances) + 1e-6 - self.distances[ee[0], ee[1]] for ee in G.edges()}, name="inverted_distance")

        return G


                
        
        

        
        
        
        
            
            

            

            

        
            
                

            

    
            
            
        
        
# to do: change to schema
class TDA:    
    """ Class to compute topological branching analysis

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object that stores the distance matrix of size (n_observations, n_observations).
    key : `str`
        The label used to reference the distance matrix stored in ``keeper.distances``,
        of size (n_observations, n_observations).
    label : `str`
        Label used to store resulting schema in ``keeper.misc[label]``.
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
    root : {`int`, `str`, `None`}
        Root observation from which pesudo-ordering is computed.

        Options :

        - `None` : Default row index of smallest distance (off-diagonal) is used as the root.
        - `int` : Index of root observation (``root > 0``).
        - `str` : Observation label.
    smooth_corr : `bool`, default = `False`
        If `True`, smooth correlations before identifying cut points for branch splitting.
    brute : `bool`
        If `True`, data points not associated with any branch upon split are combined with
        undecided (trunk) points. Otherwise, if `False`, they are treated as individual islands,
        not associated with any branch (and assigned branch index -1).
    """
    
    def __init__(self, keeper, key, label=None, # distances,
                 min_branch_size=5, choose_largest_segment=False,
                 flavor='haghverdi16', allow_kendall_tau_shift=False, root=None,
                 smooth_corr=False, brute=True, verbose=None):

        # TODO: set root upfront and call _set_pseudotime after distance is computed
        # self.keeper = keeper

        if verbose is not None:
            set_verbose(logger, verbose)
        
        self.distances = keeper.distances[key].data
        self.num_observations = keeper.num_observations
        self.observation_labels = keeper.observation_labels

        if isinstance(min_branch_size, int):
            assert min_branch_size > 2, "As an integer, `min_branch_size` must be greater than 2."
            self.check_min_branch_size = lambda x: len(x)>self.min_branch_size
        elif isinstance(min_branch_size, float):
            assert 0. < min_branch_size < 1., "As a float, `min_branch_size` must satisfy 0 < `min_branch_size` < 1."
            self.check_min_branch_size = lambda x: (len(x) / self.distances.shape[0]) > self.min_branch_size
        self.min_branch_size = min_branch_size
        self.choose_largest_segment = choose_largest_segment
        self.flavor = flavor
        self.allow_kendall_tau_shift = allow_kendall_tau_shift

        if root is None:
            # set root as index of row with largest distance
            # logger.msg(f"Suggesting root node -- to be implemented....")
            # self.root = np.unravel_index(np.argmin(self.distances, axis=None), self.distances.shape)[0] - includes diagonal of zeros
            d_tmp = self.distances.copy()
            d_tmp[np.eye(*d_tmp.shape).astype(bool)] = d_tmp.max()
            self.root = np.unravel_index(np.argmin(d_tmp, axis=None), d_tmp.shape)[0]
            logger.info(f"Suggested root set as index {self.root}")
        elif isinstance(root, str):
            # self.root = self.distances.observation_index(root)
            self.root = self.observation_labels.index(root)
            # self.root = keeper.observation_labels.index(self.root)
        elif isinstance(root, int):
            if 0 <= root < self.num_observations:
                self.root = root
            else:
                raise ValueError("Unexpected value for root of type int, must be 0 <= root < num_observations.")
        else:
            raise TypeError("Unexpected type for root, must be `None`, int or str.")
        
        # self.root = root
        self.pseudotime = None
        self._set_pseudotime()

        self.smooth_corr = smooth_corr
        self.brute = brute

        self.tree = Tree()
        

        
    def detect_branches(self, n_branches):
        """ Detect up to `n_branches` branches.

        Parameters
        ----------
        n_branches : `int`
            Number of branch splits to perform (``n_branches > 0``).
    
        Notes
        -----
        Writes : 

        - segs : `list`
        
          * list of arrays of length (number of segments). Each entry stores
            the indices of the members of a segment.
        - segs_tips : `list`
        
          * List of arrays of length (number of segments) where Each entry stores the
            indices of the two tip points of each segment.
        - segs_undecided : 
        
          * ?
        - segs_adjacency : `list`
        
          * List of lists of the same length as  ``segs``, , where the i-th entry is a list
            with the index of the trunk, if the i-th segment is not the trunk. Otherwise,
            the i-th entry is a list with the indices of all other segments beside the trunk. 
        - segs_connects : `list`
        
          * List of lists of the same length as ``segs``, where the i-th entry is a list of
            the form [index of data point in the trunk closest to the root of the i-th segment], 
            if the i-th segment is not the trunk. Otherwise, the i-th entry is a list of indices
            of the closest cell in each other (non-trunk) segment to the trunk root. 
        """

        # distances = distances if not isinstance(self.distances, pd.DataFrame) else distances.values
        
        indices_all = np.arange(self.distances.shape[0], dtype=int)
        # branch_hierarchy = [indices_all]  # keep record
        segs = [indices_all]

        self.tree.insert(TreeNode(name=0, data=indices_all, nonunique=False, unidentified=False,
                                  branchable=True if self.check_min_branch_size(indices_all) else False))

        # segs_by_index = [[0]*indices_all.shape[0]] # HERE

        # get first tip (farthest from root)
        # if self.root is None:
        #     tip_0 = np.argmax(self.distances[0])
        # else:
        #     tip_0 = np.argmax(self.distances[self.root])
        tip_0 = np.argmax(self.distances[self.root])
        # tip_0 = self.root
        # logger.msg(f"*** UPDATED TIP_0 ***")
        
        # get tip of other end (farthest from tip_0)
        tips_all = np.array([tip_0, np.argmax(self.distances[tip_0])])

        # net_distance = self.distances[self.root] + self.distances[tip_0]
        # tip_1 = np.argmax(net_distance)
        # tips_all = np.array([tip_0, tip_1]) ### CHANGED HERE
        # tips_all = np.array([self.root, tip_0]) ### CHANGED HERE
        
        # branch_tips = [tips_all]  # keep record
        segs_tips = [tips_all]
        
        segs_connects = [[]]
        segs_undecided = [True]
        segs_adjacency = [[]]
        segs_terminate_branching = [False]  # RE added - to indicate if segment has already been branched as much as possible
        # RE added - to account for points that are not associated with any branch after split
        # Note: if brute = True, unidentified points are included in trunk
        unidentified_points = set()          
        
        for ibranch in range(n_branches):
            logger.info(f"*ibranch = {ibranch}")
            # logger.warning(f"*{len(segs)} segs = {segs}")
            iseg, tips3 = self.select_segment(segs, segs_tips, segs_undecided, segs_terminate_branching)
            # logger.warning(f"*iseg = {iseg}, tips3 = {tips3}, selected_seg = {segs[iseg]}")
            if iseg == -1:
                logger.warning(f'    partitioning converged: ibranch = {ibranch}')
                break
            logger.debug(
                f'    branching {ibranch + 1}: split group {iseg}',
            )  # [third start end]
            # detect branching and update segs and segs_tips
            n_segs = len(segs)
            self.detect_branching(
                segs,
                segs_tips,
                segs_connects,
                segs_undecided,
                segs_adjacency,
                iseg,
                tips3,
                segs_terminate_branching,
                unidentified_points,
            )
            
            # RE START MODIFIED - added to indicate when segment should not be further branched
            if len(segs) == n_segs:
                logger.warning("No further branching occured at this iteration.")
                segs_terminate_branching[iseg] = True
                

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

        self.unidentified_points = unidentified_points


    def select_segment(self, segs, segs_tips, segs_undecided, segs_terminate_branching):
        """ Select segment with most distant second data point.

        Returns
        -------
        iseg
            Index identifying the position within the list of line segments.
        tips3
            Positions of tips within chosen segment (local indices of tips relative to the segment).
        """
        scores_tips = np.zeros((len(segs), 4))
        allindices = np.arange(self.distances.shape[0], dtype=int)
        # logger.warning(f"{len(segs)} segs iterating over.")
        for iseg, seg in enumerate(segs):
            # logger.warning(f"iseg = {iseg} begin iteration")
            if (segs_tips[iseg][0] == -1) or (segs_terminate_branching[iseg]): # do not consider too small segments???
                logger.debug(f"Ending iterations short")
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


            if third_maximizer is not None:
                logger.warning(f"TODO: THIRD MAXIMIZER IS NOT NONE... IS THIS CORRECT???")
                # find a fourth point that has maximal distance to all three
                dseg += Dseg[third_tip]
                fourth_tip = np.argmax(dseg)
                # should it be >>> if fourth_tip != tips[third_maximizer] and fourth_tip != third_tip: ... and >>> tips[third_maximizer] = fourth_tip ???
                if fourth_tip != tips[0] and fourth_tip != third_tip: 
                    # dseg -= Dseg[tips[1]] # RE CHANGED TO COMPUTE BEFORE UPDATING TIP
                    logger.warning(f"TODO: tip1 changed from {tips[1]} to {fourth_tip}...")
                    tips[1] = fourth_tip
                    dseg -= Dseg[tips[1]] # OLD WAY COMPUTED AFTER UPDATING TIP --- should it be dseg += Dseg[tips[1]]?
                    # dseg = Dseg[tips[0]] + Dseg[tips[1]] # RE ADDED SECOND NEW WAY OPTION:
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
                if not self.check_min_branch_size(seg) # len(seg) < self.min_branch_size
                else '',
            )
            
            if not self.check_min_branch_size(seg):  # len(seg) <= self.min_branch_size:
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
                         segs_adjacency, iseg, tips3, segs_terminate_branching,
                         unidentified_points):
        """ Detect branching on a given segment and update list parameters in place.

        Parameters
        ----------
        segs
            Stores all segments.
        segs_tips
            Stores all tip points for the segments in segs.
        iseg
            Position of segment under study in segs.
        tips3
            The three tip points local index relative to the seg.
            They form a 'triangle' that contains the data.
        unidentified_points : `set`
            Points that are not associated with any branch after splitting.
        """        
        seg = segs[iseg]

        seg_node = self.tree.get_node(self.tree.search(iseg, bottom_up=True))
        
        Dseg = self.distances[np.ix_(seg, seg)]
        # logger.warning(f"*seg = {seg}")

        # given the three tip points and the distance matrix detect the
        # branching on the segment, return the list ssegs of segments that
        # are defined by splitting this segment
        result = self._detect_branch(Dseg, tips3, seg)        
        if result is None: # RE ADDED THIS CONDITION
            logger.info(f"No unique branch detected - removed from consideration.")
            seg_node.branchable = False
        else:
            ssegs, ssegs_tips, ssegs_adjacency, ssegs_connects, trunk, unidentified = result

            # map back to global indices
            unidentified = seg[unidentified] # record data points not associated with any branch
            logger.info(f"* {len(unidentified)} unclaimed points.")
            
            for iseg_new, seg_new in enumerate(ssegs):
                ssegs[iseg_new] = seg[seg_new]
                ssegs_tips[iseg_new] = seg[ssegs_tips[iseg_new]]
                ssegs_connects[iseg_new] = list(seg[ssegs_connects[iseg_new]])

            # RE ADDED - update unidentified points
            unidentified_points = unidentified_points | set(unidentified)
            
            # remove previous segment
            segs.pop(iseg)
            segs_tips.pop(iseg)

            # insert trunk/undecided_cells at same position
            cur_tree_node = TreeNode(name=iseg, data=ssegs[trunk],
                                     parent=seg_node, nonunique=True, unidentified=False,
                                     branchable=True if self.check_min_branch_size(ssegs[trunk]) else False)
            # logger.warning(f"** {type(cur_tree_node)}")
            self.tree.insert(cur_tree_node,
                             parent=seg_node)
            segs.insert(iseg, ssegs[trunk])
            segs_tips.insert(iseg, ssegs_tips[trunk])

            # append other segments
            num_segs = len(segs)
            for ix, ixseg in enumerate(ssegs):
                if ix != trunk:
                    self.tree.insert(TreeNode(name=ix+num_segs, data=ixseg, parent=seg_node,
                                              nonunique=False, unidentified=False,
                                              branchable=True if self.check_min_branch_size(ixseg) else False),
                                     parent=seg_node)
                    
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

            # RE START MODIFIED
            # append seg reference for branching
            segs_terminate_branching += [False for i in range(len(ssegs) - 1)]
            # RE END MODIFIED
            
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
                    pos = segs_adjacency[jseg].index(iseg)
                    connection_to_iseg = segs_connects[jseg][pos]
                    for iseg_new, seg_new in enumerate(ssegs):
                        if iseg_new != trunk:
                            # pos = segs_adjacency[jseg].index(iseg)
                            # connection_to_iseg = segs_connects[jseg][pos]
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
        ssegs_adjacency : `list`
            List of lists of the same length as ssegs,
            where the i-th entry is a list with the index of the trunk, if the i-th segment is not the trunk.
            Otherwise, the i-th entry is a list with the indices of all other segments beside the trunk.
        ssegs_connects : `list`
            List of lists of the same length as ssegs,
            where the i-th entry is a list of the form [index relative to the original segment of data point in the trunk closest to the root of the i-th segment],
            if the i-th segment is not the trunk. Otherwise, the i-th entry is a list of indices of the closest cell in each other (non-trunk) segment
            to the trunk root.
        trunk : `int` 
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
        logger.info(f"* {nonunique.sum()} nonunique points.")

        # RE START MODIFIED - to account for points not associated with any branch
        unidentified = np.sum(masks, axis=0) == 0
        logger.info(f"* {unidentified.sum()} unidentified points.")        
        # RE END MODIFIED
        
        # RE START MODIFIED - uncomment to match how original paper defines unique
        # if len(ssegs) == 3:
        #     allbranches = np.sum(masks, axis=0) == len(ssegs)
        #     twobranches = np.sum(masks[1:, :], axis=0) == len(ssegs) - 1
        #     nonunique = allbranches | twobranches
        # else:
        #     nonunique = np.sum(masks, axis=0) > 1
        # # RE END MODIFIED
        ssegs = []

        for iseg, mask in enumerate(masks):
            mask[nonunique] = False
            # RE START MODIFIED  - IN EVENT THAT THE SEG HAD NO UNIQUE MEMBERS AND IS NOW EMPTY:
            # ssegs.append(np.arange(Dseg.shape[0], dtype=int)[mask])

            newseg = np.arange(Dseg.shape[0], dtype=int)[mask]
            if newseg.shape[0] == 0:
                logger.info("Unique segment is empty, removing from consideration and no branching performed.")
                # continue
                return None
            
            ssegs.append(newseg)
            # RE END MODIFIED

        # compute new tips within new segments
        ssegs_tips = []
        # logger.warning(f"*ssegs = {ssegs}")
        for inewseg, newseg in enumerate(ssegs):
            # RE MODIFIED START
            # if len(np.flatnonzero(newseg)) <= 1:
            #     logger.warning(f'detected group with only {len(np.flatnonzero(newseg))} data points')
            # secondtip = newseg[np.argmax(Dseg[tips[inewseg]][newseg])]
            # ssegs_tips.append([tips[inewseg], secondtip]) # RE: SHOULD BE CHANGED

            # logger.warning(f"*inewseg = {inewseg}, newseg = {newseg}")
            secondtip = newseg[np.argmax(Dseg[tips[inewseg]][newseg])]
            firsttip = tips[inewseg]
            
            if len(np.flatnonzero(newseg)) <= 1:
                logger.info(f'detected group with only {len(np.flatnonzero(newseg))} data points')
            if firsttip not in set(newseg):
                new_firsttip = newseg[np.argmin(Dseg[tips[inewseg]][newseg])]
                logger.warning(f'tip is no longer in the unique branched sub-segment, update tip to its nearest point in the new segment: {firsttip} -> {new_firsttip}')
                firsttip = new_firsttip
            
            ssegs_tips.append([firsttip, secondtip]) # RE: SHOULD BE CHANGED
            # RE MODIFIED END

        # RE START MODIFIED - to account for points not associated with any branch
        if self.brute:
            nonunique = nonunique | unidentified
        # RE END MODIFIED
        undecided_cells = np.arange(Dseg.shape[0], dtype=int)[nonunique]

        # RE START MODIFIED - to account for points not associated with any branch
        unidentified_points = np.arange(Dseg.shape[0], dtype=int)[unidentified]
        # RE END MODIFIED
        

        if len(undecided_cells) > 0:
            ssegs.append(undecided_cells)
            # establish the connecting points with the other segments
            ssegs_connects = [[], [], [], []]
            for inewseg, newseg_tips in enumerate(ssegs_tips):
                reference_point = newseg_tips[0]
                # closest (undecided) cell to the new segment tip within undecided cells
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
            # logger.warning(f"trunk = {trunk}")
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

        return ssegs, ssegs_tips, ssegs_adjacency, ssegs_connects, trunk, unidentified_points            

        
    def _detect_branching_single_haghverdi16(self, Dseg, tips):
        """Detect branching on given segment. """
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
        if imax > 0.95 * len(idcs):
            logger.warning('segment is more than 95\% correlated.')
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
        a, b : `numpy.ndarray`
            One dimensional sequences.
        min_length : `int`, (``min_length > 0``)
            Minimum number of data points automatically included in branch.

        Returns
        -------
        imax : `int`
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

        # TODO: add smoothing to corr_coeff before selecting max index
        if corr_coeff.size == 0:
            iimax = 0
            corr_coeff_max = 0.
            
        else:
            if self.smooth_corr:
                # logger.msg(f"corr before smoothing: {corr_coeff}")
                # logger.msg(f"imax before smoothing: {np.argmax(corr_coeff)}")
                corr_coeff = utl.gauss_conv(corr_coeff, window_size=5, smoothness=2.5)
                # logger.msg(f"corr after smoothing: {corr_coeff}")
                # logger.msg(f"imax after smoothing: {np.argmax(corr_coeff)}")

            iimax = np.argmax(corr_coeff)
            corr_coeff_max = corr_coeff[iimax]                
            
        # iimax = 0 if corr_coeff.size == 0 else np.argmax(corr_coeff)
        imax = min_length + iimax
        # corr_coeff_max = 0. if corr_coeff.size == 0 else corr_coeff[iimax]
        
        # RE END MODIFIED
        if corr_coeff_max < 0.3:
            logger.warning('    is root itself, never obtain significant correlation')
        return imax


    def _kendall_tau_add(self, len_old: int, diff_pos: int, tau_old: float):
        """ Compute Kendall tau delta.

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
        a, b : `numpy.ndarray`
            One dimensional sequences.
        i : `int`
            Index for splitting ``a`` and ``b``.

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


    def postprocess_segments(self):
        """ Convert the format of the segment class members. """
        # make segs an array, where the array is a list of (boolean) mask arrays, each of the same length as the number of data points,
        # it's easier to store as there is a hdf5 equivalent
        for iseg, seg in enumerate(self.segs):
            mask = np.zeros(self.distances.shape[0], dtype=bool)
            mask[seg] = True
            self.segs[iseg] = mask
        # convert to arrays
        self.segs = np.array(self.segs)
        self.segs_tips = np.array(self.segs_tips)
        # now each seg in segs is a boolean vector of length (self.distances.shape[0]), where it's True if that point is in the segment


    def set_segs_names(self):
        """ Return a single array that stores integer segment labels.

        -1 indicates observations that are not in any segment. """
        # RE START MODIFIED - otherwise points that aren't in any segment are not differentiated from first segment.
        # segs_names = np.zeros(self.distances.shape[0], dtype=np.int8)
        segs_names = -np.ones(self.distances.shape[0], dtype=np.int8)
        # RE END MODIFIED

        # RE START MODIFIED - include -1 in unique names
        # self.segs_names_unique = []
        # for iseg, seg in enumerate(self.segs):
        #     segs_names[seg] = iseg
        #     self.segs_names_unique.append(iseg)
        
        for iseg, seg in enumerate(self.segs):
            segs_names[seg] = iseg
        self.segs_names_unique = sorted(set(segs_names))
        # RE END MODIFIED
        self.segs_names = segs_names


    def order_pseudotime(self):
        """ Define indices that reflect segment and pseudotime order.

        Notes
        -----
        Writes : 

        - indices : np.ndarray (num_observations,)
              Index array, which stores an ordering of the data points
              with respect to increasing segment index and increasing pseudotime.
        - changepoints : np.ndarray, (num_segs - 1,)
              Index array of shape len(ssegs)-1, which stores the indices of
              points where the segment index changes, with respect to the ordering
              of indices.
        """
        # within segs_tips, order tips according to pseudotime
        if self.root is not None:
            for itips, tips in enumerate(self.segs_tips):
                if tips[0] != -1:
                    indices = np.argsort(self.pseudotime[tips])
                    self.segs_tips[itips] = self.segs_tips[itips][indices]
                else:
                    logger.warning(f"    group {itips} is very small")

        # sort indices according to segments
        indices = np.argsort(self.segs_names)
        segs_names = self.segs_names[indices]

        # find changepoints of segments
        changepoints = np.arange(indices.size - 1)[np.diff(segs_names) == 1] + 1
        if self.root is not None:
            pseudotime = self.pseudotime[indices]
            for iseg, seg in enumerate(self.segs):
                # only consider one segment, it's already ordered by segment
                seg_sorted = seg[indices]
                # consider the pseudotime on this segment and sort them
                seg_indices = np.argsort(pseudotime[seg_sorted])
                # within the segment, order indices according to increasing pseudotime
                indices[seg_sorted] = indices[seg_sorted][seg_indices]

        # define class members
        self.indices = indices  # indices of data points in original position, sorted by seg and then pseudotime
        self.changepoints = changepoints  # indices of first tip of all branches

        ordering_id = np.zeros(self.distances.shape[0], dtype=int)
        for count, idx in enumerate(self.indices):
            ordering_id[idx] = count
        self.ordering = ordering_id # ordered location of each data point


    def ordered_segs(self):
        """ returns List[array] of segments where the ``i-th`` entry has the sorted indices
        corresponding to the ``i-th`` branch.
        """
        # determine branches        
        if self.segs_names_unique[0] == -1:
            x0 = self.changepoints[0]
            changepoints = self.changepoints[1:]
        else: 
            x0 = 0
            changepoints = self.changepoints

        segs = []
        for x1 in changepoints:
            segs.append(self.indices[x0:x1])
            x0 = x1
        segs.append(self.indices[x0:])
        return segs


    def construct_topology(self):
        """ construct connections between data points.

        Returns
        -------
        G : `networkx.Graph`
            Graph where each node is a data point and edges reflect connections between them.
            Edges have attributes {'connection' : (str) 'intra-branch' or 'inter-branch'}
            Nodes have attributes

            .. code-block:: py

               {'branch' : (`int`) -1, 0, 1, ... where -1 indicates the data point was not identified with a branch,
                                   'undecided' : (bool) True if the data point is part of a trunk and False otherwise,
                                   'name' : (str) Original label if given data was a dataframe, otherwise the same as the node id,
                'unidentified' : (0 or 1) 1 if data point was ever not associated with any branch upon split,
                                   0 otherwise.,
               }        
        """
        G = nx.Graph()

        segs = self.ordered_segs()

        # # get points where each branch starts:
        # if self.segs_names_unique[0] == -1:
        #     x0 = self.changepoints[0]
        #     changepoints = self.changepoints[1:]
        # else: 
        #     x0 = 0
        #     changepoints = self.changepoints

        # # extract individual branches:
        # segs = []
        # for x1 in changepoints:
        #     segs.append(self.indices[x0:x1])
        #     x0 = x1
        # segs.append(self.indices[x0:])

        # add missing data points:
        missing_nodes = set((range(self.distances.shape[0]))) - set(itertools.chain(*segs))
        G.add_nodes_from(list(missing_nodes), branch=-1)
        nx.set_node_attributes(G, {k: False for k in missing_nodes}, name='undecided')

        # add edges within branch:
        for ix, seg in enumerate(segs):
            
            if len(seg) == 1:
                G.add_node(seg[0])
            else:
                G.add_edges_from(zip(seg, seg[1:]), connection='intra-branch')
            nx.set_node_attributes(G, {v: {'branch': ix, 'undecided': self.segs_undecided[ix]} for v in seg})

            # update tree label
            # nd = self.tree.nodes[self.tree.search_data(seg[0])]
            # nd.name_string = nd.name_string + f" |-> branch {ix}"

        # add edges connecting branches:
        segs_connects_triu = sp.sparse.triu(self.segs_connects).tocsr()
        rows, cols = segs_connects_triu.nonzero()
        inter_branch_edges = [(self.segs_connects[r, c], self.segs_connects[c, r]) for r, c in zip(rows, cols)]
        G.add_edges_from(inter_branch_edges, connection='inter-branch')

        # add node names:
        nx.set_node_attributes(G, dict(zip(range(self.num_observations), self.observation_labels)), name='name')

        # add if node was ever an unidentified point:
        nx.set_node_attributes(G, {v: 1 if v in self.unidentified_points else 0 for v in G}, name='unidentified')

        nx.set_edge_attributes(G, {ee: self.distances[ee[0], ee[1]] for ee in G.edges()}, name="distance")
        nx.set_edge_attributes(G, {ee: np.max(self.distances) + 1e-6 - self.distances[ee[0], ee[1]] for ee in G.edges()}, name="inverted_distance")

        return G


    
    def construct_pose_nn_topology(self, G=None, annotate=True):
        """ Construct pose topology with edges between nearest neighbors (nn).

        Parameters
        ----------
        G : `networkx.Graph`
            (Optional) If provided, nearest-neighbor edges are added to a copy of the graph.
            If not provided, the graph returned from `self.construct_topology()` is used.
        annotate : `bool`
            If `True`, annotate edges with edge origin and distance.

        Returns
        -------
        Gnn : `networkx.Graph`
            The updated graph with nearest neighbor edges.
            If ``annotate`` is `True`, edge attribute "edge_origin"
            is added with the possible values :

            - "POSE" : for edges in the original graph that are not nearest neighbor edges
            - "NN" : for nearest neighbor edges that were not in the original graph
            - "POSE + NN" : for edges in the original graph that are also nearest neighbor edges
        """
        if G is None:
            Gnn = self.construct_topology()
        else:
            Gnn = G.copy()
        d = self.distances
        # d = d + (np.max(d)+1e-3)*np.eye(*d.shape)
        # nn = np.argmin(d, axis=0)
        nn = np.argpartition(d, 1, axis=1)[:, 1]
        if annotate:
            nn_edges = [tuple(sorted([i,j])) for i, j in zip(range(d.shape[0]), nn)]
            nn_edges = list(set(nn_edges))

            pose_edges = list(set([tuple(sorted(ee)) for ee in Gnn.edges()]))

            nn_unique_edges = list(set(nn_edges) - set(pose_edges))
            pose_unique_edges = list(set(pose_edges) - set(nn_edges))
            nn_pose_edges = list(set(nn_edges) & set(pose_edges))

            if len(nn_unique_edges) + len(nn_pose_edges) != len(nn_edges):
                raise AssertionError("Unexpected number of nearest-neighbor edges.")

            Gnn.add_edges_from(nn_unique_edges)

            nx.set_edge_attributes(Gnn, {**{ee: "POSE + NN" for ee in nn_pose_edges},
                                         **{ee: "NN" for ee in nn_unique_edges},
                                         **{ee: "POSE" for ee in pose_unique_edges}},
                                   name="edge_origin")

            nx.set_edge_attributes(Gnn, {ee: self.distances[ee[0], ee[1]] for ee in Gnn.edges()}, name="distance")
            nx.set_edge_attributes(Gnn, {ee: np.max(self.distances) + 1e-6 - self.distances[ee[0], ee[1]] for ee in Gnn.edges()}, name="inverted_distance")
        else:
            nn_edges = [tuple([i,j]) for i, j in zip(range(d.shape[0]), nn)]
            Gnn.add_edges_from(nn_edges)

        return Gnn


    def construct_pose_mst_topology(self, G=None):
        """ Construct pose topology with minimum spanning tree (MST) edges. 

        Parameters
        ----------
        G : `networkx.Graph`
            (Optional) If provided, MST edges are added to a copy of the graph.
            If not provided, the graph returned from `self.construct_topology()` is used.

        Returns
        -------
        Gmst : `networkx.Graph`
            The updated graph with MST edges and edge attribute "edge_origin"
            with the possible values :

            - "POSE" : for edges in the original graph that are not MST edges
            - "MST" : for MST edges that were not in the original graph
            - "POSE + MST" : for edges in the original graph that are also MST edges
        """
        if G is None:
            G = self.construct_topology()
        else:
            G = G.copy()
        
        d = pd.DataFrame(self.distances)
        edgelist = utl.stack_triu_(d)
        edgelist.name = 'weight'
        edgelist = edgelist.reset_index().rename(columns={'level_0': 'source', 'level_1': 'target'})
        Gfull = nx.from_pandas_edgelist(edgelist, edge_attr='weight')
        Gmst = nx.minimum_spanning_tree(Gfull)

        mst_edges = [tuple(sorted([i,j])) for i,j in Gmst.edges()]
        mst_edges = list(set(mst_edges))

        pose_edges = list(set([tuple(sorted(ee)) for ee in G.edges()]))

        mst_unique_edges = list(set(mst_edges) - set(pose_edges))
        pose_unique_edges = list(set(pose_edges) - set(mst_edges))
        mst_pose_edges = list(set(mst_edges) & set(pose_edges))

        if len(mst_unique_edges) + len(mst_pose_edges) != len(mst_edges):
            raise AssertionError("Unexpected number of MST edges.")

        G.add_edges_from(mst_unique_edges)

        nx.set_edge_attributes(G, {**{ee: "POSE + MST" for ee in mst_pose_edges},
                                   **{ee: "MST" for ee in mst_unique_edges},
                                   **{ee: "POSE" for ee in pose_unique_edges}},
                               name="edge_origin")

        nx.set_edge_attributes(G, {ee: self.distances[ee[0], ee[1]] for ee in G.edges()}, name="distance")
        nx.set_edge_attributes(G, {ee: np.max(self.distances) + 1e-6 - self.distances[ee[0], ee[1]] for ee in G.edges()}, name="inverted_distance")

        return G


        
    def branchings_segments(self, n_branches):
        """ Detect up to `n_branches` branches and partition the data into corresponding segments.

        Parameters
        ----------
        n_branches : `int`
            Number of branches to look for (``n_branches > 0``).
         
        Notes
        -----
        Writes :
        
        - segs : `numpy.ndarray`, (n_segments, n_data_points)
              Array where each row stores a mask array that defines a segment.
        - segs_tips : `numpy.ndarray`, (n_segments, 2)
              Array where each row stores the
              indices of the two tip points of each segment.
        - segs_names : `numpy.ndarray`, (n_data_points, )
              Array that stores an integer label
              for each segment.
        """
        self.detect_branches(n_branches)
        self.postprocess_segments()
        self.set_segs_names()
        self.order_pseudotime()


    def _set_pseudotime(self):
        """ Return pseudotime with respect to root point. """
        # root = self.root if root is None else root
        # if root is None:
        #     msg = "'root' must be specified in order to compute pseudo-ordering."
        #     raise AssertionError(msg)

        self.pseudotime = self.distances[self.root].copy()

        self.pseudotime /= np.max(self.pseudotime[self.pseudotime < np.inf])





        


"""
scanpy.tools._dpt.py
line 157: dpt.branchings_segments()
    line 237: self.detect_branchings()  # Detect all branchings up to `n_branchings`. (here, called self.detect_branches())
    line 238: self.postprocess_segments() # make segs an array, where the array is a list of (boolean) mask arrays, each of the same length as the number of data points
    line 239: self.set_segs_names()
    line 240: self.order_pseudotime() # Define indices that reflect segment and pseudotime order.

"""
            
        
            


    

                



