# Copyright 2017 F. Alexander Wolf, P. Angerer, Theis Lab
# Revisions copyright 2024 R. Elkin
# All rights reserved.
# This file contains code derived, either in part or in whole,
# from the Scanpy library, which is governed by the original
# "BSD 3-Clause License".
# Please see the LICENSE file included as part of this package for
# more details.
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

from collections import defaultdict as ddict
from itertools import combinations
import itertools
import networkx as nx
import numpy as np
from operator import itemgetter
import pandas as pd
import pptree
import scipy as sp
from scipy.sparse import issparse
from sklearn.preprocessing import normalize

# from importlib import reload
# pptree = reload(pptree)
# import netflow.utils as utl
import netflow.utils as utl
# from importlib import reload
# reload(utl)
from .similarity import mutual_knn_edges
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
             smooth_corr=False, brute=True, split=True, connect_closest=False,
             mutual=False, k_mnn=3,
             verbose=None):
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
    root : {`None`, `int`, 'density', 'density_inv', 'ratio'}
        The root. If `None`, 'density' is used.

        Options:

        - `int` : index of observation
        - 'density' : select observation with minimal distance-density
        - 'density_inv' : select observation with maximal distance-density
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
    mutual : `bool` (default = `False`)
        If `True`, add ``k_mnn`` mutual nn edges. Otherwise, add single nn edge.
        When `False`, ``k_mnn`` is ignored.
    k_mnn : `int` (``0 < k_mnn < len(G)``)
        The number of nns to consider when extracting mutual nns.
        Note, this is ignored when ``mutual`` is `False`.
    connect_closest : `bool` (default = False)
            If `True`, connect branches by points with smallest distance between the branches.
            Otherwise, connect by continuum of ordering. 

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
                  connect_closest=connect_closest,
                  flavor=flavor, allow_kendall_tau_shift=allow_kendall_tau_shift,
                  smooth_corr=smooth_corr, brute=brute, split=split, verbose=verbose)
    G_pose = poser.branchings_segments(n_branches, until_branched=until_branched)
    G_pose.name = 'pose_' + label
    keeper.add_misc(poser, 'poser_' + label)
    keeper.add_graph(G_pose, 'pose_' + label)

    G_pose_nn = poser.construct_pose_nn_topology(G_pose, mutual=mutual, k_mnn=k_mnn)
    pre_label = 'pose_mnn_' if mutual else 'pose_nn_'
    G_pose_nn.name = pre_label + label
    keeper.add_graph(G_pose_nn, pre_label + label)


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

    if density_normalize:
        asym_label = "_".join([asym_label, "density_normalized"])
        sym_label = "_".join([sym_label, "density_normalized"])
        
    # if label is not None:
    #     asym_label = "_".join([asym_label, label])
    #     sym_label = "_".join([sym_label, label])

    keeper.add_misc(transitions_asym, asym_label)
    keeper.add_misc(transitions_sym, sym_label)


def compute_rw_transitions(keeper, similarity_key, do_save=True):
    """ Compute the row-stochastic transition matrix.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object.
    similarity_key : `str`
        Reference key to the `numpy.ndarray`, (n_observations, n_observations)
        symmetric similarity measure (with 1s on the diagonal) stored in the similarities
        in the keeper.
    do_save : `bool`
        If `True`, save to ``keeper``.

    Returns
    -------
    P : `numpy.ndarray` (n_observations, n_observations)
        The row-stochastic transition matrix (with 0s on the diagonals).
        If ``do_save`` is `True`, ``P`` is added to the ``keeper.misc`` with the key ``'transitions_rw_{similarity_key}'``
    """
    similarity = keeper.similarities[similarity_key].data
    P = normalize(similarity, "l1", axis=1)

    if do_save:
        P_label = f"transitions_rw_{similarity_key}"
        keeper.add_misc(P, P_label)

    return P


def compute_sym_diffusion_affinity_transitions(keeper, similarity_key, do_save=True):
    """ Compute the symmetric diffusion affinity transition matrix from
    https://github.com/KrishnaswamyLab/graphtools/blob/master/graphtools/base.py.

    .. math:: P_{ij} = K_{ij} * (d_i * d_j)^{-1/2}

    where :math:`d_i = \sum_r K_{ir}` is the degree (row sum) of observation :math:`i`.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object.
    similarity_key : `str`
        Reference key to the `numpy.ndarray`, (n_observations, n_observations)
        symmetric similarity measure (with 1s on the diagonal) stored in the similarities
        in the keeper.
    do_save : `bool`
        If `True`, save to ``keeper``.

    Returns
    -------
    P : `numpy.ndarray` (n_observations, n_observations)
        The symmetric diffusion affinity transition matrix (with 0s on the diagonals).
        If ``do_save`` is `True`, ``P`` is added to the ``keeper.misc`` with the key ``'transitions_sym_diff_aff_{similarity_key}'``
    """
    similarity = keeper.similarities[similarity_key].data
    row_degrees = similarity.sum(axis=1)[:, None]
    col_degrees = similarity.sum(axis=0)[None,:]
    
    P = (similarity / np.sqrt(row_degrees)) / np.sqrt(col_degrees)

    if do_save:
        P_label = f"transitions_sym_diff_aff_{similarity_key}"
        keeper.add_misc(P, P_label)

    return P



def compute_multiscale_VNE_transitions_from_similarity(keeper, similarity_key,
                                                       tau_max=None, do_save=True):
    """ Compute the multi-scale transition matrix based on the elbow of the Von Neumann Entropy (VNE)
    as described in GSPA and PHATE https://github.com/KrishnaswamyLab/spARC/blob/main/SPARC/vne.py,
    https://pdfs.semanticscholar.org/16ab/e92b7630d5b84b904bde97dad9b9fbce406c.pdf.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper object.
    similarity_key : `str`
        Reference key to the `numpy.ndarray`, (n_observations, n_observations)
        symmetric similarity measure (with 1s on the diagonal) stored in the similarities
        in the keeper.
    tau_max : `int`
        Max scale ``tau`` tested for VNE (default is 100).
    do_save : `bool`
        If `True`, save to ``keeper``.

    Returns
    -------
    P : `numpy.ndarray` (n_observations, n_observations)
        The symmetric VNE multi-scale transition matrix (with 0s on the diagonals).
        If ``do_save`` is `True`, ``P`` is added to the ``keeper.misc`` with the key ``'transitions_sym_multiscaleVNE_{similarity_key}'``
    P_asym : `numpy.ndarray` (n_observations, n_observations)
        The random-walk VNE multi-scale transition matrix (with 0s on the diagonals).
        If ``do_save`` is `True`, ``P_asym`` is added to the ``keeper.misc`` with the key ``'transitions_multiscaleVNE_{similarity_key}'``
    """
    P_sym_label = f"transitions_sym_multiscaleVNE_{similarity_key}"
    P_asym_label = f"transitions_multiscaleVNE_{similarity_key}"

    if P_sym_label in keeper.misc and P_asym_label in keeper.misc:
        P = keeper.misc[P_sym_label]
        P_asym = keeper.misc[P_asym_label]
    else:
        P_label = f"transitions_sym_diff_aff_{similarity_key}"
        if P_label in keeper.misc:
            P = keeper.misc[P_label]
        else:
            P = compute_sym_diffusion_affinity_transitions(keeper, similarity_key, do_save=False)
        vne = utl.von_neumann_entropy(P, tau_max=tau_max)
        tau = utl.find_knee_point(vne)

        # if not use_affinity_diffusion_matrix:
        #    P = compute_rw_transitions(keeper, similarity_key, do_save=False)    

        P_rw_label = f"transitions_rw_{similarity_key}"
        if P_rw_label in keeper.misc:
            P_asym = keeper.misc[P_rw_label]
        else:
            P_asym = compute_rw_transitions(keeper, similarity_key, do_save=False)

        P = np.linalg.matrix_power(P, tau)
        P_asym = np.linalg.matrix_power(P_asym, tau)

        if do_save:
            # if use_affinity_diffusion_matrix:
            #     P_label = f"transitions_sym_multiscaleVNE_{similarity_key}"
            # else:
            #     P_label = f"transitions_multiscaleVNE_{similarity_key}"

            try:
                keeper.add_misc(P, P_sym_label)
            except Exception as e:
                P = keeer.misc[P_sym_label]
                logger.warning(f"Returning pre-computed {P_sym_label}")
            try:
                keeper.add_misc(P_asym, P_asym_label)
            except Exception as e:
                P_asym = keeper.misc[P_asym_label]
                logger.warning(f"Returning pre-computed {P_sym_label}")

    return P, P_asym


def _dpt_from_augmented_sym_transitions(T, n_comps: int = 0, return_eigs=False):
    """ Return the diffusion pseudotime metric between observations,
    computed from the symmetric transitions.

    .. Note::

        - :math:`T` is the symmetric transition matrix
        - :math:`M(x,z) = \sum_{i=1}^{n-1} (\lambda_i * (1 - \lambda_i))\psi_i(x)\psi_i^T(z)`
        - :math:`dpt(x,z) = ||M(x, .) - M(y, .)||^2`
    
    Parameters
    ----------
    T : `numpy.ndarray`, (n_observations, n_observations)
        Symmetric transitions.
    n_comps
        Number of eigenvalues/vectors to be computed,
        set ``n_comps = 0`` to compute the whole spectrum.
        Alternatively, if set ``n_comps >= n_observations``,
        the whole spectrum will be computed.

    Returns
    -------
    dpt : `numpy.ndarray`, (n_observations, n_observations)
        Pairwise-observation Diffusion pseudotime distances.
    """
    evals, evecs = utl.compute_eigen(T, n_comps=n_comps, sort="decrease")
    if np.abs(evals[0] - 1.) > 1.8e-2: # 1e-2:
        raise ValueError(f"Largest eigenvalue is expected to be close to 1, found to be {np.round(evals[0], 4)}")
    if evals[1] > 1.0:
        raise ValueError(f"Expected second largest eigenvalue to be less than 1, found to be {np.round(evals[1], 4)}")
    EVALS = np.diag(evals[1:] / (1. - evals[1:]))
    M = evecs[:, 1:] @ EVALS @ evecs[:, 1:].T
    dpt = sp.spatial.distance.cdist(M, M)

    if return_eigs:
        return dpt, evals, evecs
    
    return dpt


def dpt_from_augmented_sym_transitions(keeper, key, n_comps: int = 0, save_eig=False):
    """ Compute the diffusion pseudotime metric between observations,
    computed from the symmetric transitions.

    .. Note::

        - :math:`T` is the symmetric transition matrix
        - :math:`M(x,z) = \sum_{i=1}^{n-1} (\lambda_i * (1 - \lambda_i))\psi_i(x)\psi_i^T(z)`
        - :math:`dpt(x,z) = ||M(x, .) - M(y, .)||^2`
    
    Parameters
    ----------
    key : `str`
        Reference ID for the symmetric transitions
        `numpy.ndarray`, (n_observations, n_observations)
        stored in ``keeper.misc``.
    n_comps
        Number of eigenvalues/vectors to be computed,
        set ``n_comps = 0`` to compute the whole spectrum.
        Alternatively, if set ``n_comps >= n_observations``,
        the whole spectrum will be computed.

    Returns
    -------
    dpt : `numpy.ndarray`, (n_observations, n_observations)
        Pairwise-observation Diffusion pseudotime distances are stored
        in keeper.distances[dpt_key] where ``dpt_key="dpt_from_{key}"``.
        If the full spectrum is not used (i.e., ``0 < n_comps < n_observations"``),
        then ``dpt_key="dpt_from_{key}_{n_comps}comps"``.
    """
    T = keeper.misc[key]
    
    dpt, evals, evecs = _dpt_from_augmented_sym_transitions(T, n_comps=n_comps, return_eigs=True)

    dpt_key = f"dpt_from_{key}"
    if 0 < n_comps < keeper.num_observations:
        dpt_key = "_".join([dpt_key, f"{n_comps}comps"])
    keeper.add_distance(dpt, dpt_key)

    if save_eig:
        keeper.add_misc(evals, dpt_key.replace('dpt_from', 'evals'))
        keeper.add_misc(evecs, dpt_key.replace('dpt_from', 'evecs'))
    

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

        Options:

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
    connect_closest : `bool` (default = False)
            If `True`, connect branches by points with smallest distance between the branches.
            Otherwise, connect by continuum of ordering. 
    """
    def __init__(self, keeper, key, root=None, root_as_tip=False,
                 min_branch_size=5, choose_largest_segment=False,
                 flavor='haghverdi16', allow_kendall_tau_shift=False,
                 smooth_corr=True, brute=True, split=True, connect_closest=False,
                 verbose=None):

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
        self.connect_closest = connect_closest

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
                raise ValueError("Unrecognized method for determining root, expected to be one of ['density', 'density_inv', 'ratio'].")
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
    

    def _detect_branch(self, Dseg: np.ndarray, tips: np.ndarray): # , connect_closest=False):
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

                if self.connect_closest:
                    for i_cur_seg, cur_seg in enumerate(ssegs[:-1]):
                         dseg_cur = Dseg[np.ix_(cur_seg, ssegs[-1])]
                         point_in_seg, point_in_trunk = np.unravel_index(np.argmin(dseg_cur), dseg_cur.shape)
                         ssegs_connects.append([[trunk, i_cur_seg], [undecided_cells[point_in_trunk],
                                                                     cur_seg[point_in_seg]]])
                else:
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

                if self.connect_closest:
                    for seg_a, seg_b in combinations(range(3), 2):                        
                        # i_cur_seg, cur_seg in enumerate(ssegs[:-1]):
                         dseg_cur = Dseg[np.ix_(ssegs[seg_a], ssegs[seg_b])]
                         point_in_seg_a, point_in_seg_b = np.unravel_index(np.argmin(dseg_cur), dseg_cur.shape)
                         closest_points[seg_a, seg_b] = ssegs[seg_b][point_in_seg_b]
                         closest_points[seg_b, seg_a] = ssegs[seg_a][point_in_seg_a]
                else:
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

                if self.connect_closest:
                    dseg_cur = Dseg[np.ix_(ssegs[0], ssegs[1])]
                    point_in_seg_0, point_in_seg_1 = np.unravel_index(np.argmin(dseg_cur), dseg_cur.shape)
                    ssegs_connects = [[[1, 0], [ssegs[1][point_in_seg_1],
                                                ssegs[0][point_in_seg_0]]]]
                else:
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

            if self.connect_closest:
                dseg_cur = Dseg[np.ix_(branch_seg, main_seg)]
                point_in_branch_seg, point_in_main_seg = np.unravel_index(np.argmin(dseg_cur), dseg_cur.shape)
                closest_cell_a = branch_seg[point_in_branch_seg]
                closest_cell_b = main_seg[point_in_main_seg]

                trunk=0
                ssegs_connects = [[[0, 1], [closest_cell_a,
                                            closest_cell_b]]]

            else:
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
            New segment (local with respect to original segment).
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

    
    def detect_branching(self, node): # , connect_closest=False):
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

        result = self._detect_branch(Dseg, tips3) # , connect_closest=connect_closest)
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


    def single_branch(self, until_branched=False): # , connect_closest=False):
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
                branched_flag = self.detect_branching(node) # , connect_closest=connect_closest)

            if not until_branched:
                # branched_flag = True
                break

        return branched_flag
                

    def detect_branches(self, n_branches, until_branched=False): # , connect_closest=False):
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

                branched_flag = self.single_branch(until_branched=until_branched) # , connect_closest=connect_closest)
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
            
            
    def branchings_segments(self, n_branches, until_branched=False, annotate=True): # , connect_closest=False):
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
        self.detect_branches(n_branches, until_branched=until_branched) # , connect_closest=connect_closest)

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
            The banched segments indexed by the node's unique identifier.
        annotate : `bool`
            If `True`, annotate edges with edge origin and distance.

        Returns
        -------
        G : `networkx.Graph`
            Graph where each node is a data point and edges reflect connections between them.
            If ``annotate`` is `True`, the following annotations are added:
        
            - Edges have attributes:

                - 'connection' : (str) 'intra-branch' or 'inter-branch'}
            - Nodes have attributes:

                - 'branch' : (`int`) -1, 0, 1, ... where -1 indicates the data point was not identified with a branch
                - 'undecided' : (bool) True if the data point is part of a trunk and False otherwise
                - 'name' : (str) Original label if given data was a dataframe, otherwise the same as the node id
                - 'unidentified' : (0 or 1) 1 if data point was ever not associated with any branch upon split, 0 otherwise.
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


    def construct_pose_nn_topology(self, G, mutual=False, k_mnn=3, annotate=True):
        """ Add nearest neighbor (nn) edges to POSE topology.

        .. note:: Mutual nns tend to be sparser than nns so allow to select more
                  than just the first nn if restricting to mutual neighbors.

        Parameters
        ----------
        G : `networkx.Graph`
            Nearest-neighbor edges are added to a copy of the POSE graph.
        mutual : `bool` (default = `False`)
            If `True`, add ``k_mnn`` mutual nn edges. Otherwise, add single nn edge.
            When `False`, ``k_mnn`` is ignored.
        k_mnn : `int` (``0 < k_mnn < len(G)``)
            The number of nns to consider when extracting mutual nns.
            Note, this is ignored when ``mutual`` is `False`.
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

        if mutual:
            nn_edges = mutual_knn_edges(d, n_neighbors=k_mnn)
            nn_edges = [tuple(sorted(ee)) for ee in nn_edges]  # TODO: this might be redundant if returned in sorted order already
        else:
            # nn = np.argmin(d, axis=0)        
            nn = np.argpartition(d, 1, axis=1)[:, 1]            
            nn_edges = [tuple(sorted([i,j])) for i, j in zip(range(d.shape[0]), nn)]
            nn_edges = list(set(nn_edges))
            
        
        if annotate:
            # nn_edges = [tuple(sorted([i,j])) for i, j in zip(range(d.shape[0]), nn)]
            # nn_edges = list(set(nn_edges))

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
            # nn_edges = [tuple([i,j]) for i, j in zip(range(d.shape[0]), nn)]
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
