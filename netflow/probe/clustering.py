import itertools 
import networkx as nx
import numpy as np
import pandas as pd

from collections import defaultdict as ddict
from sklearn.cluster import DBSCAN

def dbscan_clustering(d, eps=0.5, min_samples=3, metric="precomputed", **kwargs):
    """ Perform DBSCAN clustering. 
    
    Parameters
    ----------
    d : `pandas.DataFrame`
        The symmetric, observation-pairwise distances.
    {eps, min_samples, metric, kwargs}
        Key-word arguments passed to `sklearn.cluster.DBSCAN`

    Returns
    -------
    clusters : `pandas.Series`
        The cluster assignment for each observation in ``d``.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, 
                    metric=metric).fit(d)

    clusters = pd.Series(data=dbscan.labels_, index=d.index, name='dbscan')
    return clusters


def branch_clustering(d, branch_record, clustering, min_branch_size=6, **kwargs):
    """ Perform clustering on each branch.
    
    Note: outliers should be returned from ``clustering`` with the value -1.
    Currently, such outliers are treated as a cluster.
    
    Parameters
    ----------
    d : `pandas.DataFrame`
        The symmetric, observation-pairwise distances.
    branch_record : `pandas.Series`
        The branch assignment for each observation in ``d``.
    clustering : func
        A function that performs clustering from given data. Must accept data matrix 
        and key-word arguments)
    min_branch_size : `int`
        Branches with <= ``min_branch_size`` observations are not considered for further clustering.
    kwargs : `dict`
        Key-word arguments passed to ``clustering``.

    Returns
    -------
    records : `pandas.Series` 
        The new cluster index for each observation.
    branch_map : `dict`
        Maps the original branch index to indices of new child sub clusters.
    """
    records = {}
    for ix in branch_record.unique():
        obs = branch_record.index[branch_record==ix]
        if len(obs) <= min_branch_size:
            records[ix] = branch_record.loc[obs]
        else:
            records[ix] = clustering(d.loc[obs, obs], **kwargs)

    branch_counter = 0
    branch_map = {}
    for ix, br in records.items():
        br = br.replace(dict(zip(sorted(br.unique()), 
                                 [branch_counter + k for k in range(br.nunique())])))
        records[ix] = br
        branch_counter += br.nunique()
        branch_map[ix] = br.unique()

    records = pd.concat(list(records.values()), axis=0)
    return records, branch_map


def high_res_branch_graph(G):
    """ Extract higher resolution graph from POSE.
    
    Each node represents a branch from the POSE, and edges are placed between nodes 
    containing observations incident to edges between branches in the original POSE.
    
    Parameters
    ----------
    G : `nx.Graph`
        The graph returned from `netflow.TDA.construct_topology`.
        
    Returns
    -------
    Ghr : `nx.Graph`
        The high-res graph.
    """
    branch_record = pd.Series(dict(G.nodes.data(data='branch', 
                                                default=-1))).rename(index=dict(G.nodes.data(data='name')))   
    branch_record.name = 'branch'
    
    Ghr = nx.Graph()
    Ghr.add_nodes_from(branch_record.unique())

    br = branch_record.to_frame().groupby('branch')
    nx.set_node_attributes(Ghr, 
                           {i : list(j) for i, j in br.groups.items()}, 
                           name="members")
    nx.set_node_attributes(Ghr, {i: len(Ghr.nodes[i]['members']) for i in Ghr},
                           name="n_members")

    for ee in G.edges():
        if branch_record.loc[G.nodes[ee[0]]['name']] == branch_record.loc[G.nodes[ee[1]]['name']]:
            continue

        # put edge between super-nodes containing observations incident on the original edge:
        Ghr.add_edge(branch_record.loc[G.nodes[ee[0]]['name']], branch_record.loc[G.nodes[ee[1]]['name']])

    return Ghr


def high_res_clustered_branch_graph(G, clustering):
    """ Extract higher resolution graph from clustering on POSE branches.
    
    Each node represents a cluster from a POSE branch, and edges are placed between nodes 
    containing observations incident to edges between branches in the original POSE.
    
    Parameters
    ----------
    G : `nx.Graph`
        The graph returned from `netflow.TDA.construct_topology`.
    clustering : `pandas.Series`
        The cluster IDs, indexed by the observation labels.

    Returns
    -------
    Ghr : `nx.Graph`
        The high-res graph.
    """
    branch_record = pd.Series(dict(G.nodes.data(data='branch', 
                                                default=-1))).rename(index=dict(G.nodes.data(data='name')))   
    branch_record.name = 'branch'
    
    Ghr = nx.Graph()
    Ghr.add_nodes_from(clustering.unique())

    clustering = clustering.copy()
    clustering.name = 'cluster'
    br = clustering.to_frame().groupby('cluster')
    nx.set_node_attributes(Ghr, 
                           {i : list(j) for i, j in br.groups.items()}, 
                           name="members")
    nx.set_node_attributes(Ghr, {i: len(Ghr.nodes[i]['members']) for i in Ghr},
                           name="n_members")
    nx.set_node_attributes(Ghr, {i: branch_record.loc[Ghr.nodes[i]['members'][0]] for i in Ghr},
                           name="parent")
    
    for ee in G.edges():
        if clustering.loc[G.nodes[ee[0]]['name']] == clustering.loc[G.nodes[ee[1]]['name']]:
            continue

        # put edge between super-nodes containing observations incident on the original edge:
        Ghr.add_edge(clustering.loc[G.nodes[ee[0]]['name']], clustering.loc[G.nodes[ee[1]]['name']])

    return Ghr


def avg_cluster_edges(X, clustering, G=None):
    """ Return average value of X within and between clusters.
    
    Parameters
    ----------
    X : `pandas.DataFrame`
        The dobservation-pairwise data.
    clustering : `pandas.Series`
        The clustering IDs indexed by the observation labels.
    G : `networkx.Graph`
        (Optional) If provided, restrict to the average over over edges in the graph.
        Otherwise, the average is taken between all observations in the cluster.
        Expected to have node attribute 'name' corresponding to the name of each observation 
        (represented as nodes).

    Returns
    -------
    R : `pandas.DataFrame`
        The average values within and between clusters, where the index and columns 
        are the cluster references.
    """
    if G is not None:
        node_ids = {G.nodes[i]['name'] : i for i in G}
        
    clustering = clustering.copy()
    clustering.name = 'cluster'
     
    R = {i: ddict(float) for i in clustering.unique()}

    members = clustering.to_frame().groupby('cluster')
    members = {i : list(j) for i, j in members.groups.items()}
    
    # within clusters
    for ix, clus in members.items():
        if len(clus) == 1:
            R[ix][ix] = 0.
        else:
            if G is None:
                x = [X.loc[i, j] for (i,j) in itertools.combinations(clus, 2)]
            else:
                x = [X.loc[i, j] for (i,j) in itertools.combinations(clus, 2) if (node_ids[i], 
                                                                                  node_ids[j]) in G.edges()]
            x = np.asarray(x)
            R[ix][ix] = x.mean()
        
    # between clusters
    for ix_a, ix_b in itertools.combinations(clustering.unique(), 2):
        if G is None:
            x = X.loc[members[ix_a], members[ix_b]]
            R[ix_a][ix_b] = R[ix_b][ix_a] = np.mean(x.values)
        else:
            x = [X.loc[i, j] for (i, j) in itertools.product(members[ix_a], 
                                                             members[ix_b]) if (node_ids[i], 
                                                                                node_ids[j]) in G.edges()]
            x = np.asarray(x)
            R[ix_a][ix_b] = R[ix_b][ix_a] = x.mean()
                                                                              
    R = pd.DataFrame.from_dict(R)
    R = R.loc[sorted(R.columns), sorted(R.columns)]
    return R
             

def louvain(G, weight='inverted_distance', resolution=1., seed=0, **kwargs):
    """ Compute Louvain communities on graph, intended for POSE

    Louvain communities are computed via ``networkx.community.louvain_communities``

    Parameters
    ----------
    G : `networkx.Graph`
        The graph.
    weight : {`None`, `str`}
        The edge attribute of the value used as the weight. If None, set to 1
        for all edges (default value = 'inverted_distance').
    resolution : `float`
        Influences algorithm preference for larger (resolution value greater than 1)
        or smaller (resolution value smaller than 1) communities.
    seed : `int`
        Random generator state.
    kwargs : `dict`
        Keyword arguments passed to ``networkx.community.louvain_communities``.

    Returns
    -------
    lvp : `dict`
       Index of community each node is partitioned into, keyed by the nodes.
    """
    louvain_partition = nx.community.louvain_communities(G,
                                                         weight=weight,
                                                         resolution=resolution,
                                                         seed=seed,
                                                         **kwargs)
    lvp = {node: i for i, partition in enumerate(louvain_partition) for node in partition}
    # lvp = np.zeros(len(G))
    # for ix, cc in enumerate(louvain_partition):
    #     lvp[list(cc)] = ix
        
    # lvp = dict(zip(range(len(G)), lvp))
    return lvp


def louvain_paritioned(G, class_attr, louvain_attr=None,
                       weight='inverted_distance', resolution=1., seed=0,
                       **kwargs):
    """ Compute Louvain communities on graph and further partition restricted to existing classifier (e.g., branches intended for POSE)

    Louvain communities are computed via ``networkx.community.louvain_communities``

    Parameters
    ----------
    G : `networkx.Graph`
        The graph.
    class_attr : `str`
        The node attribute of the value of the pre-assigned class attribute against
        which the Louvain communities should be partitioned.
    louvain_attr : {`None`, `str`}
        (Optional) Node attribute where Louvain community indices are stored.
        If provided, first check if the attribute already exists in ``G`` to use
        pre-compouted Louvain community idices (if it exists, remaining argument
        values are ignored). Otherwise, the computed Louvain community indices are
        stored in this node attribute. If not provided, Louvain communities are
        computed and not stored.
    weight : {`None`, `str`}
        The edge attribute of the value used as the weight. If None, set to 1
        for all edges (default value = 'inverted_distance').
        (Ignored if pre-existing Louvain communiites were saved in ``louvain_attr``.)
    resolution : `float`
        Influences algorithm preference for larger (resolution value greater than 1)
        or smaller (resolution value smaller than 1) communities.
        (Ignored if pre-existing Louvain communiites were saved in ``louvain_attr``.)
    seed : `int`
        Random generator state.
        (Ignored if pre-existing Louvain communiites were saved in ``louvain_attr``.)
    kwargs : `dict`
        Keyword arguments passed to ``networkx.community.louvain_communities``.
        (Ignored if pre-existing Louvain communiites were saved in ``louvain_attr``.)

    Returns
    -------
    The following node attributes are added to the graph ``G`` :

        - f"{class_attr}_{louvain_attr}" : The class partitioned and Louvain community
          reference index in the form "{class-index}-{Louvain-index}"
          (``louvain_attr`` defaults to "lvp" if not provided).
        - louvain_attr - The Louvain community reference index, if provided.
    """
    ca = dict(nx.get_node_attributes(G, class_attr))


    if louvain_attr is not None:
        lvp = dict(nx.get_node_attributes(G, louvain_attr))
    else:
        lvp = {}
        
    if len(lvp) != len(G):
        lvp = louvain(G, weight=weight, resolution=resolution, seed=seed, **kwargs)
        # louvain_partition = nx.community.louvain_communities(G,
        #                                                      weight=weight,
        #                                                      resolution=resolution,
        #                                                      seed=seed,
        #                                                      **kwargs)
        # lvp = {node: i for i, partition in enumerate(louvain_partition) for node in partition}
        if louvain_attr is not None:
            nx.set_node_attributes(G, lvp, name=louvain_attr)

    louvain_str = louvain_attr if louvain_attr is not None else 'lvp'

    nx.set_node_attributes(G, {k: f"{ca[k]}-{lvp[k]}" for k in G},
                           name=f"{class_attr}-{louvain_str}")


    
            
    
    
