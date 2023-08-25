import itertools
import networkx as nx
import numpy as np
import scipy.sparse.csgraph as scg
import time

def check_connected_graph(G):
    """ Raises an AssertionError if the graph is not connected"""
    assert nx.is_connected(G), "The graph must be connected."


def check_graph_no_self_loops(G):
    """ Raises an AssertionError if the graph has self-loops"""
    assert len(list(nx.selfloop_edges(G))) == 0, "No self-loops are allowed in the graph."


def check_graph_weights(G, component, weight, tol=1e-6):
    """ Raises an AssertionError if there are weights below the tolerance

    Parameters
    -----------
    G : networkx graph
        The graph.
    component : {'node', 'edge'}
        Component of graph the weights are attributed to.
    weight : str
        Graph attribute of the weight.
    tol : float, default = TOL
        The tolerance to check that no weights are below `tol`.
    """
    if component == 'node':
        weights = nx.get_node_attributes(G, weight).values()
    elif component == 'edge':
        weights = nx.get_edge_attributes(G, weight).values()
    else:
        return ValueError("Unrecognized value for component, must be one of ['node', 'edge'].")

    is_not_pos = min(weights) < TOL
    if is_not_pos:
        raise ValueError("Weights must be positive.")


def check_edges_in_graph(G, edges):
    """ Raises AssertionError if not all edges are in the graph. """
    assert all([edge in G.edges() for edge in edges]), "All edges must be in the graph."


def check_symmetric(A):
    """ Raises AssertionError if the matrix is not symmetric. """
    np.testing.assert_allclose(A, A.T, err_msg='Matrix is not symmetric.')


def check_distance_matrix(A):
    """ Raises AssertionError if the distance matrix is not non-negative and symmetric"""
    assert A.ndim == 2, "Distance matrix must be 2-dimensional."
    assert A.shape[0] == A.shape[1], "Distance matrix must have the same number of rows as columns."
    check_symmetric(A)
    assert np.min(A) >= 0., "Distance matrix must be non-negative."



