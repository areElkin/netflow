import networkx as nx
import pandas as pd

from netflow import InfoNet
from netflow.methods.metrics import norm_features_as_sym_dist
from netflow.pose.organization import compute_transitions, dpt_from_augmented_sym_transitions, TDA
from netflow.pose.similarity import sigma_knn
from netflow import similarity as nfs
from netflow._logging import _gen_logger, set_verbose
logger = _gen_logger(__name__)

from importlib import reload
from netflow.pose import organization as nfo
nfo = reload(nfo)
# from netflow.methods import classes as nfc
# nfc = reload(nfc)
# InfoNet = nfc.InfoNet
TDA = nfo.TDA
# logger.msg(f"*** UPDATED TDA ***")
# logger.warning(f"Imported new TDA: {help(TDA)}")

def _fuse_similarities_from_recipes(keeper, similarity_labels, recipes):
    """ Fuse similarities according to prescribed recipe.

    Currently, only performs unweighted average.

    .. Note:: An empty `list` is returned if ``recipes`` is `None`.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper.
    similarity_labels : `list`
        List of similarities as labeled in the keeper that may be  used
        to compute the fused similarities.
    recipes : `list` [list]
        List of recipes for fusion, where each listed item contains the indices
        of the similarities, as ordered in ``similarity_labels``, to be included
        in the fusion.

    Returns
    -------
    fused_similarity_labels : `list`
        A list of the fused similarity labels that are added to ``keeper.similarities``.        
    """
    if recipes is None:
        fused_similarity_labels = []
        return fused_similarity_labels
    
    fused_similarity_setup = {}
    for recipe in recipes:
        cur_labels = [similarity_labels[k] for k in recipe]
        fused_label = "fused_similarity_" + "_".join([k.split('similarity_')[-1] for k in cur_labels])
        fused_similarity_setup[fused_label] = cur_labels

    fused_similarity_labels = []
    for ll, labels in fused_similarity_setup.items():
        n = len(labels)
        dd = keeper.similarities[labels[0]].data
        for i in labels[1:]:
            dd = dd + keeper.similarities[i].data
        dd = dd / n
        keeper.add_similarity(dd, ll)
        fused_similarity_labels.append(ll)

    return fused_similarity_labels


def _pose_from_distance(keeper, distance_label, root=None, similarity_label=None, n_branches=5, min_branch_size=6,
                        choose_largest_segment=False, flavor='haghverdi16', allow_kendall_tau_shift=False,
                        smooth_corr=False, brute=True, verbose=None):
    """ Compute POSE from root with respect to distance.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper.
    distance_label : `str`
        The label of the distance in ``keeper`` to be used.
    root : {`None`, `int`, `str`}
        Root observation from which pesudo-ordering is computed.

        Options :

        - `None` : If ``similarity_label`` is provided, the observation with the largest density
            is used as the root. Otherwise, (previously: the observation with the smallest net 
            distance to all other observations is used) use observation furthest from observation
            with largest ratio of triangular distance to chord distance.
        - `int` : Index of root observation (``root > 0``).
        - `str` : Observation label.
    similarity_label : {`None`, `str`}
        If ``root`` is `None`, use the similarity to set the observation with the largest
        density as ``root``. If not provided, ``root`` is set as the observation with the
        smallest net distance to all other observations.
        This is ignored if ``root`` is not `None`.
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
    n_branches : `int`
        Number of branch splits to perform (``n_branches > 0``).

    Returns
    -------
    tda : `netflow.TDA`
        The TDA object.
    G_tda : `networkx.Graph`
        The POSE network topology.
    G_tda_nn : `networkx.Graph`
        The POSE network topology with nearest neighbor edges.
    """
    dd = keeper.distances[distance_label]

    if dd.label != distance_label:
        raise ValueError("Distance label does not match.")

    if root is None:
        # find root from density
        if similarity_label is None:
            root = keeper.distance_density_argmin(dd.label)
            # logger.msg(f"** Root from sim: root = {root}")
        else:
            # uncomment to use minimum density from distance:
            root = keeper.similarity_density_argmax(similarity_label)
                        
            
        # uncomment to find root as farthest point from point leading to largest triangular ratio:
        # d = dd.to_frame()
        # max_d_pairs = d.idxmax()  # 
        # sum_max_d_pairs = max_d_pairs.to_frame('a').T.apply(lambda x: d.loc[x.name] + d.loc[x.loc['a']])
        # tip3 = sum_max_d_pairs.idxmax()

        # max_d_pairs.name = 'tip2'
        # tip3.name = 'tip3'
        # tips = pd.concat([max_d_pairs, tip3], axis=1)
        # tri_ratio = tips.T.apply(lambda x: (d.loc[x.name, x.loc['tip3']] + d.loc[x.loc['tip2'], x.loc['tip3']]) / (d.loc[x.name, x.loc['tip2']])) # .hist(bins=100)

        # root_x = tri_ratio.idxmax()
        # root_lbl = d.loc[root_x].idxmax()
        # # root_lbl = tri_ratio.idxmax()
        # root = keeper.observation_index(root_lbl)
        # logger.msg(f"setting root = {root_lbl}: {root}")


    # logger.msg(f"root = {root} type = {type(root)}.")

    tda = TDA(keeper, dd.label, label=None,  min_branch_size=min_branch_size,
              choose_largest_segment=choose_largest_segment,
              flavor=flavor, allow_kendall_tau_shift=allow_kendall_tau_shift,
              root=root, smooth_corr=smooth_corr, brute=brute, verbose=verbose)

    tda.branchings_segments(n_branches)

    G_tda = tda.construct_topology()    
    nx.set_node_attributes(G_tda, {k: tda.pseudotime[k] for k in G_tda}, name="pseudo-distance to root")
    
    pos = nx.layout.kamada_kawai_layout(G_tda)
    # pos_d = pd.DataFrame(data=tda.distances).to_dict()
    # pos = nx.layout.kamada_kawai_layout(G_tda, dist=pos_d)
    # nx.set_edge_attributes(G_tda, {ee: tda.distances[ee[0], ee[1]] for ee in G_tda.edges()}, name="weight")
    # pos = nx.layout.kamada_kawai_layout(G_tda, weight='weight')
    # pos = nx.layout.kamada_kawai_layout(G_tda, weight='distance')
    nx.set_node_attributes(G_tda, pos, name='pos')

    G_tda_nn = tda.construct_pose_nn_topology(G_tda)
    # G_tda_mst = tda.construct_pose_mst_topology(G=G_tda)

    cur_key = "_".join(["POSE", dd.label, f"nbranches{n_branches}", f"min{min_branch_size}",
                        f"largestSeg{choose_largest_segment}", flavor, f"shift{allow_kendall_tau_shift}",
                        f"smooth{smooth_corr}", f"brute{brute}", f"root{root}"])
    # branch_record = pd.Series(dict(G_tda.nodes.data(data='branch', default=-1))).rename(index=dict(G_tda.nodes.data(data='name')))
    # branch_record.name = cur_key
    G_tda.name = cur_key
    G_tda_nn.name = f"NN_{cur_key}"
    # G_tda_mst.name = f"MST_{cur_key}"
    # keeper.add_graph(G_tda, cur_key)

    return tda, G_tda, G_tda_nn # , G_tda_mst
            

def run_pose(keeper, metrics_configs,
             sigma_config={'n_neighbors': 5, 'method': 'max', 'knn': False},
             density_normalize=True, fuse_recipes=None, root=None,
             n_branches=5, min_branch_size=6, choose_largest_segment=False,
             flavor='haghverdi16', allow_kendall_tau_shift=False,
             smooth_corr=False, brute=True, verbose=None):
    """ Run data to POSE pipeline.

    Specify data set(s) and metric(s) to be performed.

    If more than one data set and/or more than one metric is selected,
    a fused similarity is computed.

    The POSE is computed on the resulting fused metric.

    Parameters
    ----------
    keeper : `netflow.Keeper`
        The keeper.
    metrics_configs : `list`
        The data, feature list configuration for computing metrics. The configurations
        are provided as a `tuple` of `tuples` where each metric configuration is a tuple
        of the form
        ``('data_label', 'metric_name', 'graph_key', {'features': [], 'feature_set_name': str, **kwargs})`` 
        where the `dict` has the optional keywords:

        - 'features' : A `list` of feature that the metric should be computed on.
            The features can be the feature labels or feature indices as found in the
            ``keeper``. If not provided, all features are used.
        - `feature_set_name': A `str` used to identify the specified set of features
            (e.g., 'immune_signature'). If not specified, a default name is used.
            If ``'features'`` is not provided, this is ignored.
        - **kwargs : A `dict` with any optional key-word arguments that should be
            passed when computing the corresponding metric.

        Use ``('data-label', 'metric_name',)`` to use all features and default metric arguments.
    
        For example, a single metric configuration using all features and default
        metric arguments should be of the form
        metrics_configs = ``( ('data_label',  'metric-name', None, ), )``
    sigma_config : `dict`
        Configuration for computing sigmas.
    density_normalize : `bool`, default = True
        If `True`, use the density rescaling of Coifman and Lafon (2006): Then only the
        geometry of the data matters, not the sampled density when computing the diffusion
        distances.
    fuse_recipes : {`None`, `List`}
        Prescribe which metrics should be used to compused fused similarities. If `None`,
        and more than one metric was specified, all metrics are used; if only one metric was
        specified, no fusion is performed. Otherwise, if not `None`, each item should be a
        list of indices corresponding to their order in ``metrics_configs``.        
    root : {`None`, `int`, `str`]
        Root observation from which pesudo-ordering is computed.

        Options :

        - `None` : If ``similarity_label`` is provided, the observation with the largest density
            is used as the root. Otherwise, the observation with the smallest net distance to all
            other observations is used
        - `int` : Index of root observation (``root > 0``).
        - `str` : Observation label.
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
    n_branches : `int`
        Number of branch splits to perform (``n_branches > 0``).
    """
    # Compute distances
    feature_set_label_counter = 1
    distance_labels = []
    for config in metrics_configs.copy():
        data_label = config[0]
        metric_name = config[1]
        graph_key = config[2]
        
        if len(config) == 4:
            kwargs = config[3]
            if 'features' in kwargs:
                features = kwargs.pop('features')
                if 'feature_set_name' in kwargs:
                    feature_set_name = kwargs.pop('feature_set_name')
                else:
                    feature_set_name = f"feature_set_{feature_set_label_counter}"
                    feature_set_label_counter += 1
                    keeper.add_misc(features, feature_set_name)
            else:
                features = None
                feature_set_name = "all features"
        else:
            kwargs = {}
            features = None
            feature_set_name = "all features"

        inet = InfoNet(keeper, graph_key, layer=data_label)

        label = f"{data_label}_{metric_name}_{feature_set_name}"
        if graph_key is not None:
            label = label + f"_graph_{graph_key}"

        if metric_name.startswith('euc_nbhd'):
            norm_metric = metric_name.split('euc_nbhd_')[1]
            misc_label = f"{data_label}_euc_nbhd"
            inet.multiple_pairwise_observation_neighborhood_euc_distance(nodes=features,
                                                                         label=misc_label,
                                                                         **kwargs)

            norm_features_as_sym_dist(keeper, misc_label,
                                      label=label,
                                      features=features,
                                      method=norm_metric,
                                      is_distance=True) 


        elif metric_name == 'euc_profile':                
            inet.pairwise_observation_profile_euc_distance(features=features,
                                                           label=label,
                                                           **kwargs)

        elif metric_name.startswith('wass_nbhd'):
            norm_metric = metric_name.split('wass_nbhd_')[1]
            misc_label = f"{data_label}_wass_nbhd"
            if graph_key is not None:
                misc_label = misc_label + f"_graph_{graph_key}"

            if 'dhop' not in kwargs:
                dhop = inet.compute_graph_distances(weight=None)
            else:
                dhop = kwargs.pop('dhop')
            inet.multiple_pairwise_observation_neighborhood_wass_distance(nodes=features,
                                                                          graph_distances=dhop,
                                                                          label=label,
                                                                          **kwargs)

            norm_features_as_sym_dist(keeper, misc_label,
                                      label=label,
                                      features=features,
                                      method=norm_metric,
                                      is_distance=True)

        elif metric_name == "wass_profile":
            inet.pairwise_observation_profile_wass_distance(features=features,
                                                            graph_distances=None,
                                                            label=label,
                                                            **kwargs)
        else:
            raise ValueError("Unrecognized value for 'metric_name'")

        distance_labels.append(label)

    # compute sigmas for each distance computed
    similarity_labels = []
    if 'method' in sigma_config:
        sigma_method = sigma_config['method']
    else:
        sigma_method = 'max'

    if 'n_neighbors' in sigma_config:
        n_neighbors = sigma_config['n_neighbors']
    else:
        n_neighbors = 5

    if 'knn' in sigma_config:
        knn = sigma_config['knn']
    else:
        knn = False

    for d_label in distance_labels:
        dd = keeper.distances[d_label]
        # sigma_config={'n_neighbors': 5, 'method': 'max', 'knn': False}

        sigma_label = f"{sigma_method}{n_neighbors}nn_{dd.label}"

        nfs.sigma_knn(keeper, dd.label,
                      label=sigma_label,
                      n_neighbors=n_neighbors, method=sigma_method, return_nn=True)

        sim_label = f"similarity_{sigma_label}"
        similarity_labels.append(sim_label)

        nfs.distance_to_similarity(keeper, dd.label, n_neighbors, 'precomputed',
                                   label=sim_label,
                                   sigmas=f"sigmas_{sigma_label}",
                                   knn=knn,
                                   indices=f"nn_indices_{sigma_label}")

        # add similarity to keeper as distance
        keeper.add_distance(1. - keeper.similarities[sim_label].to_frame(), f"distance_from_{sim_label}")

    # compute fused similarities:        
    if fuse_recipes is None:
        if len(similarity_labels) > 1:
            fuse_recipes = [list(range(len(similarity_labels)))]
    fused_similarity_labels = _fuse_similarities_from_recipes(keeper, similarity_labels, fuse_recipes)

    # add fused similarity to keeper as distance
    for sim_label in fused_similarity_labels:
        keeper.add_distance(1. - keeper.similarities[sim_label].to_frame(), f"distance_from_{sim_label}")
    
    # compute diffusion distances
    for sim_label in similarity_labels + fused_similarity_labels:
        sim = keeper.similarities[sim_label]

        compute_transitions(keeper, sim.label, density_normalize=density_normalize)

        T_sym_label = f"transitions_sym_{sim.label}"
        dpt_from_augmented_sym_transitions(keeper, T_sym_label)


    # compute POSE
    POSE_distance_labels = [f"distance_from_{k}" for k in similarity_labels + fused_similarity_labels]
    POSE_distance_labels = POSE_distance_labels + [f"dpt_from_transitions_sym_{k}" for k in similarity_labels + fused_similarity_labels]
    
    for distance_label in POSE_distance_labels:
        if distance_label.startswith('distance_from'):
            sim_key = distance_label.split('distance_from_')[-1]
        elif distance_label.startswith('dpt_from_transitions_sym_'):
            sim_key = distance_label.split('dpt_from_transitions_sym_')[-1]
        else:
            raise AssertionError("Unrecognized similarity for current distance label")


        print(f"Root for distance = {distance_label} ==> {root}.")
        # tda, G_tda, G_tda_nn, G_tda_mst = _pose_from_distance(keeper, distance_label, root=root,
        tda, G_tda, G_tda_nn = _pose_from_distance(keeper, distance_label, root=root,
                                                   similarity_label=None, # sim_label,
                                                   n_branches=n_branches, min_branch_size=min_branch_size,
                                                   choose_largest_segment=choose_largest_segment, flavor=flavor,
                                                   allow_kendall_tau_shift=allow_kendall_tau_shift,
                                                   smooth_corr=smooth_corr, brute=brute, verbose=verbose)

        if 'root' not in G_tda.nodes[tda.root]:
            root_attr = {k: 1 if k == tda.root else 0 for k in G_tda}
            nx.set_node_attributes(G_tda, root_attr, name='root')
            nx.set_node_attributes(G_tda_nn, root_attr, name='root')
            # nx.set_node_attributes(G_tda_mst, root_attr, name='root')

        
        keeper.add_misc(tda, f"TDA_{G_tda.name}")
        keeper.add_graph(G_tda, G_tda.name)
        keeper.add_graph(G_tda_nn, G_tda_nn.name)
        # keeper.add_graph(G_tda_mst, G_tda_mst.name)
        


