import numpy as np
import pandas as pd

import netflow.utils as utl

from .._logging import _gen_logger

logger = _gen_logger(__name__)


from collections import defaultdict as ddict

def edges_from_mutual_knn_indices(kmnn):
    """ Convert mutual k-nns to a list of edges.

    Parameters
    ----------
    kmnn : ``defaultdict`
        The mutual k-nn indices as returned from ``mutual_knn_indices``.

    Returns
    -------
    edges : `list`
        The list of edges corresponding to the mutual k-nns.
    """
    edges = list(itertools.chain(*[itertools.product([ix], [k for k in nns if k > ix]) for ix, nns in kmnn.items()]))
    return edges


def mutual_knn_edges(d, n_neighbors=None):
    """ Get edges between indices of mutual k-nearest neighbors (nn) from distance matrix.

    .. note:: Self is not included as one of the k-nns.

    Parameters
    ----------
    d : `numpy.ndarray`, (m, m)
        Symmetric distance matrix.
    n_neighbors : {`int`, `None`}
        Number of mutual nns to include (does not include self),
        ``n_neighbors > 0``. (Uses ``n_neighbors + 1``, since each obs is its closest neighbor).
        If `None`, all neighbors are used (same as k-nns since all neighbors are mutually included).
    
    Returns
    -------
    edges : `list`
        The list of edges corresponding to the mutual k-nns.
    """
    
    indices, distances = nf.similarity.get_knn_indices_distances(d, n_neighbors=n_neighbors)

    edges = []

    for ix, nbrs in enumerate(indices):
        for nbr in nbrs:
            if nbr <= ix:
                continue

            if ix in indices[nbr]:
                edges.append((ix, nbr))

    return edges



    
    
def mutual_knn_indices(d, n_neighbors=None):
    """ Get indices of mutual k-nearest neighbors (nn).

    Parameters
    ----------
    d : `numpy.ndarray`, (m, m)
        Symmetric distance matrix.
    n_neighbors : {`int`, `None`}
        Number of mutual nns to include (does not include self),
        ``n_neighbors > 0``. (Uses ``n_neighbors + 1``, since each obs is its closest neighbor).
        If `None`, all neighbors are used (same as k-nns since all neighbors are mutually included).
    
    Returns
    -------
    kmnn_indices : `defaultdict[`list`]` of the form ``{m : [up to ``n_neighbors`` mutual nns]}``
        Defaultdict keyed by row index referrencing the row indices of its mutual nns out of ``n_neighbors`` nns.
        Note, this does not include itself in output.
    """
    
    indices, distances = nf.similarity.get_knn_indices_distances(d, n_neighbors=n_neighbors)

    kmnn_indices = ddict(list)

    for ix, nbrs in enumerate(indices):
        for nbr in nbrs:
            # if (nbr >= ix) or (nbr in kmnn_indices[ix]):
            if nbr <= ix:
                continue

            if ix in indices[nbr]:
                kmnn_indices[ix].append(nbr)
                kmnn_indices[nbr].append(ix)
