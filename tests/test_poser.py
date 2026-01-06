"""
Tests POSER class for constructing the POSE.

This uses a minimal keeper stub and uses mocking to test POSER.detect_branching
tree updates without relying on the full DPT splitting logic.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ To run a single test module (e.g. test_poser.py):

$ cd netflow          # project directory
$ python -m unittest tests.test_poser

~ To run all test modules:

$ cd netflow          # project directory
$ python -m unittest -v
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
# from scipy.spatial.distance import cdist

from netflow.pose.organization import POSER, TreeNode
from netflow.keepers.keeper import Keeper


# class FakeDistance:
#     """Distance container mimicking keeper.distances[key] object interface."""
#     def __init__(self, data, labels):
#         self.data = np.array(data, dtype=float)
#         self._labels = list(labels)
#
#     def to_frame(self):
#         return pd.DataFrame(self.data, index=self._labels, columns=self._labels)
#
#
# class FakeKeeper:
#     """Minimal Keeper stub sufficient for POSER initialization and root selection."""
#     def __init__(self, D, labels, key="D"):
#         self.observation_labels = list(labels)
#         self.num_observations = len(labels)
#         self.distances = {key: FakeDistance(D, labels)}
#         self.misc = {}
#         self.similarities = {}
#
#     def observation_index(self, label):
#         return self.observation_labels.index(label)
#
#     def distance_density_argmin(self, key):
#         # deterministic choice for tests
#         return 0
#
#     def distance_density_argmax(self, key):
#         return self.num_observations - 1
#
#     def add_misc(self, arr, k):
#         self.misc[k] = arr
#
#     def add_distance(self, arr, k):
#         # store distance-like outputs the same way
#         self.distances[k] = FakeDistance(arr, self.observation_labels)

def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Compute dense Euclidean distance matrix for small test fixtures."""
    diffs = X.T[:, None, :] - X.T[None, :, :]
    dist_matrix = np.sqrt((diffs ** 2).sum(axis=2))

    # dist_matrix = cdist(X.T, X.T, metric='euclidean')
    return dist_matrix


class TestPOSERInit(unittest.TestCase):
    def test_init_root_density_default(self):
        """POSER(root=None) defaults to 'density' and uses keeper.distance_density_argmin."""
        D = np.array([
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 2, 0, 1],
            [3, 3, 1, 0],
        ], dtype=float)
        labels = ["a", "b", "c", "d"]
        # keeper = FakeKeeper(D, labels=["a", "b", "c", "d"])
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})

        poser = POSER(keeper, key="D", root=None, min_branch_size=3, smooth_corr=False)

        self.assertEqual(poser.root, 2)
        self.assertEqual(poser.distances.shape, (4, 4))
        self.assertIsNotNone(poser.tree.root)
        self.assertTrue(isinstance(poser.tree.root, TreeNode))

    def test_init_root_density_inv(self):
        """POSER(root="density_inv") uses keeper.distance_density_argmax."""
        D = np.array([
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 2, 0, 1],
            [3, 3, 1, 0],
        ], dtype=float)
        labels = ["a", "b", "c", "d"]
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})

        poser = POSER(keeper, key="D", root="density_inv", min_branch_size=3, smooth_corr=False)

        self.assertEqual(poser.root, 3)
        self.assertEqual(poser.distances.shape, (4, 4))
        self.assertIsNotNone(poser.tree.root)
        self.assertTrue(isinstance(poser.tree.root, TreeNode))

    def test_pseudo_dist_normalization(self):
        """POSER._set_pseudo_dist scales pseudo_dist so max finite non-inf value is 1."""
        D = np.array([
            [0, 2, 4],
            [2, 0, 6],
            [4, 6, 0],
        ], dtype=float)
        labels = ["a", "b", "c"]
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})
        # keeper = FakeKeeper(D, labels=["a", "b", "c"])

        poser = POSER(keeper, key="D", root=0, min_branch_size=3, smooth_corr=False)

        self.assertTrue(np.all(poser.pseudo_dist >= 0))
        self.assertAlmostEqual(np.max(poser.pseudo_dist), 1.0, places=10)

    def test_min_branch_size_validation_int(self):
        """POSER rejects integer min_branch_size <= 2."""
        D = np.eye(5)
        labels = list("abcde")
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})
        # keeper = FakeKeeper(D, labels=list("abcde"))
        with self.assertRaises(AssertionError):
            POSER(keeper, key="D", root=0, min_branch_size=2)

    def test_min_branch_size_validation_float(self):
        """POSER rejects float min_branch_size outside (0, 1)."""
        D = np.eye(5)
        # keeper = FakeKeeper(D, labels=list("abcde"))
        labels = list("abcde")
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})
        with self.assertRaises(AssertionError):
            POSER(keeper, key="D", root=0, min_branch_size=1.0)
        with self.assertRaises(AssertionError):
            POSER(keeper, key="D", root=0, min_branch_size=0.0)


class TestPOSERHelpers(unittest.TestCase):
    def test_identify_local_tips_updates_missing_first_tip(self):
        """identify_local_tips reassigns first tip to nearest point if original tip is not in newseg."""
        D = np.array([
            [0, 1, 5, 6],
            [1, 0, 4, 7],
            [5, 4, 0, 2],
            [6, 7, 2, 0],
        ], dtype=float)
        # keeper = FakeKeeper(D, labels=["a", "b", "c", "d"])
        labels = list("abcd")
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})
        poser = POSER(keeper, key="D", root=0, min_branch_size=3, smooth_corr=False)

        # newseg does not contain tip=0
        newseg = [2, 3]
        tips = poser.identify_local_tips(D, newseg, tip=0)

        self.assertIn(tips[0], newseg)
        self.assertIn(tips[1], newseg)
        self.assertEqual(tips.shape, (2,))

    def test_construct_topology_sets_expected_attributes(self):
        """_construct_topology annotates branch membership, node names, and edge attributes when annotate=True."""
        D = np.array([
            [0, 1, 5, 6],
            [1, 0, 4, 7],
            [5, 4, 0, 2],
            [6, 7, 2, 0],
        ], dtype=float)
        # keeper = FakeKeeper(D, labels=["a", "b", "c", "d"])
        labels = list("abcd")
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})
        poser = POSER(keeper, key="D", root=0, min_branch_size=3, smooth_corr=False)

        segs = {
            0: {"name": 0, "tips": np.array([0, 1]), "seg": np.array([0, 1]), "undecided": False},
            1: {"name": 1, "tips": np.array([2, 3]), "seg": np.array([2, 3]), "undecided": False},
        }
        # fabricate an inter-branch connection in the POSER tree
        poser.tree.node_connection = [[[0, 1], [1, 2]]]

        G = poser._construct_topology(segs, annotate=True)

        self.assertIsInstance(G, nx.Graph)
        self.assertIn("name", G.nodes[0])
        self.assertIn("branch", G.nodes[0])
        self.assertIn("unidentified", G.nodes[0])

        # edge annotations
        for u, v in G.edges():
            self.assertIn("connection", G.edges[(u, v)])
            self.assertIn("distance", G.edges[(u, v)])
            self.assertIn("inverted_distance", G.edges[(u, v)])

    def test_construct_pose_nn_topology_edge_origin_labels(self):
        """construct_pose_nn_topology labels edges as POSE/NN/POSE+NN depending on overlap."""
        D = np.array([
            [0, 1, 10, 10],
            [1, 0, 10, 10],
            [10, 10, 0, 1],
            [10, 10, 1, 0],
        ], dtype=float)
        # keeper = FakeKeeper(D, labels=["a", "b", "c", "d"])
        labels = list("abcd")
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})
        poser = POSER(keeper, key="D", root=0, min_branch_size=3, smooth_corr=False)

        # Start with a simple POSE graph containing one intra edge per pair
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        nx.set_node_attributes(G, {0: {"branch": 0}, 1: {"branch": 0}, 2: {"branch": 1}, 3: {"branch": 1}})
        G.add_edge(0, 1, connection="intra-branch")
        G.add_edge(2, 3, connection="intra-branch")

        Gnn = poser.construct_pose_nn_topology(G, mutual=False, annotate=True)

        # Should have edge_origin on all edges
        for e in Gnn.edges():
            self.assertIn("edge_origin", Gnn.edges[e])
            self.assertIn(Gnn.edges[e]["edge_origin"], {"POSE", "NN", "POSE + NN"})

    def test_construct_pose_mst_nn_topology_edge_origin_labels(self):
        """construct_pose_mst_nn_topology labels edges as POSE/NN/POSE+NN depending on overlap."""
        D = np.array([
            [0, 1, 10, 10],
            [1, 0, 10, 10],
            [10, 10, 0, 1],
            [10, 10, 1, 0],
        ], dtype=float)
        # keeper = FakeKeeper(D, labels=["a", "b", "c", "d"])
        labels = list("abcd")
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})
        poser = POSER(keeper, key="D", root=0, min_branch_size=3, smooth_corr=False)

        # Use a minimal MST-like backbone graph
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edge(0, 1)  # within cluster
        G.add_edge(1, 2)  # bridge
        G.add_edge(2, 3)  # within cluster

        Gnn = poser.construct_pose_mst_nn_topology(G, mutual=False, annotate=True)

        for e in Gnn.edges():
            self.assertIn("edge_origin", Gnn.edges[e])
            self.assertIn(Gnn.edges[e]["edge_origin"], {"POSE", "NN", "POSE + NN"})
            self.assertIn("distance", Gnn.edges[e])
            self.assertIn("inverted_distance", Gnn.edges[e])


class TestPOSERDetectBranchingUpdateLogic(unittest.TestCase):
    def test_detect_branching_updates_tree_and_connections(self):
        """detect_branching maps local segments to global, inserts trunk + branch nodes, and updates node_connection."""
        D = np.array([
            [0, 1, 2, 9, 9, 9],
            [1, 0, 2, 9, 9, 9],
            [2, 2, 0, 9, 9, 9],
            [9, 9, 9, 0, 1, 2],
            [9, 9, 9, 1, 0, 2],
            [9, 9, 9, 2, 2, 0],
        ], dtype=float)
        # keeper = FakeKeeper(D, labels=[f"x{i}" for i in range(6)])
        labels = [f"x{i}" for i in range(6)]
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})
        poser = POSER(keeper, key="D", root=0, min_branch_size=3, smooth_corr=False)

        node = poser.tree.root
        # detect_branching expects 3 tips; in real flow select_segment makes this.
        node.tips = np.array([0, 1, 2], dtype=int)

        # Return a controlled branching result in LOCAL coordinates:
        # - seg0 (trunk) = [0,1,2]
        # - seg1 (branch) = [3,4]
        # - undecided/unidentified local point = [5]
        fake_result = (
            [np.array([0, 1, 2]), np.array([3, 4])],               # ssegs
            [np.array([0, 2]), np.array([3, 4])],                  # ssegs_tips
            [[[0, 1], [2, 3]]],                                    # ssegs_connects (segments 0<->1 by points 2<->3)
            0,                                                     # trunk index
            True,                                                  # trunk_undecided
            np.array([5], dtype=int),                              # unidentified local points
        )

        with patch.object(poser, "_detect_branch", return_value=fake_result):
            updated = poser.detect_branching(node)

        self.assertTrue(updated)

        # unidentified points should be recorded globally
        self.assertIn(5, poser.unidentified_points)

        # Tree should now contain: original root + trunk node + one branch node
        self.assertGreaterEqual(len(poser.tree.nodes), 3)

        # node_connection should have been updated with counter-ids and global obs ids
        self.assertGreaterEqual(len(poser.tree.node_connection), 1)
        conn = poser.tree.node_connection[-1]
        self.assertEqual(len(conn[0]), 2)  # two node counters
        self.assertEqual(conn[1], [2, 3])  # global points from mapping

    def test_detect_branch_wolf17_bi_returns_valid_structure(self):
        """
        _detect_branch with flavor='wolf17_bi' returns well-formed segments/tips/connects and valid trunk index.

        NOTE: If this ever returns None due to edge cases, may need to tweak the toy Dseg
        to make the bipartition less ambiguous.
        """
        Dseg = np.array([
            [0, 1, 2, 9],
            [1, 0, 2, 9],
            [2, 2, 0, 9],
            [9, 9, 9, 0],
        ], dtype=float)

        # keeper = FakeKeeper(np.eye(4), labels=["a", "b", "c", "d"])
        labels = list("abcd")
        keeper = Keeper(distances={"D": pd.DataFrame(data=Dseg, index=labels, columns=labels)})
        poser = POSER(keeper, key="D", root=0, min_branch_size=3, smooth_corr=False, flavor="wolf17_bi", split=True)

        tips = np.array([0, 1, 3], dtype=int)
        out = poser._detect_branch(Dseg, tips)

        self.assertIsNotNone(out)

        ssegs, ssegs_tips, ssegs_connects, trunk, trunk_undecided, unidentified = out
        self.assertIsInstance(ssegs, list)
        self.assertTrue(all(len(seg) > 0 for seg in ssegs))
        self.assertTrue(0 <= trunk < len(ssegs))

        # indices must be in range
        for seg in ssegs:
            self.assertTrue(np.all((np.array(seg) >= 0) & (np.array(seg) < Dseg.shape[0])))

        for tt in ssegs_tips:
            self.assertTrue(np.all((np.array(tt) >= 0) & (np.array(tt) < Dseg.shape[0])))

    def test_detect_branch_wolf17_bi_partitions_by_closer_tip0_vs_tip1(self):
        """wolf17_bi: _detect_branch partitions points by whether they are closer to tips[0] than tips[1]."""
        # 1D points along a line: 0,1,2 closer to 0; 8,9,10 closer to 10
        x = np.array([0, 1, 2, 8, 9, 10], dtype=float)
        D = np.abs(x[:, None] - x[None, :])

        keeper = Keeper(distances={"D": D}) # _FakeKeeper(D)
        poser = POSER(
            keeper,
            key="D",
            root=0,
            min_branch_size=3,
            smooth_corr=False,
            flavor="wolf17_bi",
            split=True,
        )

        tips = np.array([0, 5, 2], dtype=int)  # tips[2] is ignored by wolf17_bi segmentation
        out = poser._detect_branch(D, tips)
        self.assertIsNotNone(out)

        ssegs, ssegs_tips, ssegs_connects, trunk, trunk_undecided, unidentified = out

        # Expect exactly two segments, no undecided trunk from nonunique/unidentified in this construction
        self.assertEqual(len(ssegs), 2)
        self.assertFalse(trunk_undecided)
        self.assertEqual(len(ssegs_connects), 1)

        seg0 = set(map(int, ssegs[0]))
        seg1 = set(map(int, ssegs[1]))
        self.assertTrue(seg0.isdisjoint(seg1))
        self.assertEqual(seg0 | seg1, set(range(D.shape[0])))

        # Validate partition logic: closer_to_tip0_than_tip1
        t0, t1 = tips[0], tips[1]
        closer_to_0 = set(np.where(D[t0] < D[t1])[0].astype(int))
        farther_from_0 = set(range(D.shape[0])) - closer_to_0

        self.assertEqual(seg0, closer_to_0)
        self.assertEqual(seg1, farther_from_0)

        # Validate returned tips format
        self.assertEqual(len(ssegs_tips), 2)
        for tt in ssegs_tips:
            self.assertEqual(len(tt), 2)

    def test_detect_branch_wolf17_tri_partitions_into_three_voronoi_segments(self):
        """wolf17_tri: _detect_branch partitions points into 3 unique segments based on pairwise distance comparisons."""
        # Tips at corners; extra points near each tip; one center point closer to tip0
        X = np.array(
            [
                [0.0, 0.0],  # 0 tip0
                [10.0, 0.0],  # 1 tip1
                [0.0, 10.0],  # 2 tip2
                [1.0, 1.0],  # 3 near tip0
                [9.0, 1.0],  # 4 near tip1
                [1.0, 9.0],  # 5 near tip2
                [3.0, 3.0],  # 6 closer to tip0 than tip1/tip2
            ],
            dtype=float,
        )
        D = _pairwise_euclidean(X.T)

        keeper = Keeper(distances={"D": D})  # _FakeKeeper(D)
        poser = POSER(
            keeper,
            key="D",
            root=0,
            min_branch_size=3,
            smooth_corr=False,
            flavor="wolf17_tri",
            split=True,
        )

        tips = np.array([0, 1, 2], dtype=int)
        out = poser._detect_branch(D, tips)
        self.assertIsNotNone(out)

        ssegs, ssegs_tips, ssegs_connects, trunk, trunk_undecided, unidentified = out

        self.assertEqual(len(ssegs), 3)
        self.assertFalse(trunk_undecided)
        self.assertEqual(len(ssegs_connects), 2)  # 3 segments => 2 inter-seg connects to trunk

        # Compute expected Voronoi-style partition used by wolf17_tri implementation:
        dist0, dist1, dist2 = D[tips[0]], D[tips[1]], D[tips[2]]
        expected_seg0 = set(np.where((dist0 < dist1) & (dist0 < dist2))[0].astype(int))
        expected_seg1 = set(np.where((dist0 >= dist1) & (dist1 < dist2))[0].astype(int))
        expected_seg2 = set(np.where((dist0 >= dist2) & (dist1 >= dist2))[0].astype(int))

        seg0 = set(map(int, ssegs[0]))
        seg1 = set(map(int, ssegs[1]))
        seg2 = set(map(int, ssegs[2]))

        self.assertEqual(seg0, expected_seg0)
        self.assertEqual(seg1, expected_seg1)
        self.assertEqual(seg2, expected_seg2)

        # Sanity: disjoint + cover all points
        self.assertTrue(seg0.isdisjoint(seg1))
        self.assertTrue(seg0.isdisjoint(seg2))
        self.assertTrue(seg1.isdisjoint(seg2))
        self.assertEqual(seg0 | seg1 | seg2, set(range(D.shape[0])))

        # Validate connects structure: [[[trunk, i], [u, v]], ...]
        self.assertTrue(0 <= trunk < 3)
        for conn in ssegs_connects:
            self.assertEqual(len(conn), 2)
            self.assertEqual(len(conn[0]), 2)
            self.assertEqual(len(conn[1]), 2)
            self.assertTrue(all(0 <= int(x) < 3 for x in conn[0]))
            self.assertTrue(all(0 <= int(x) < D.shape[0] for x in conn[1]))

        # Validate returned tips shape
        self.assertEqual(len(ssegs_tips), 3)
        for tt in ssegs_tips:
            self.assertEqual(len(tt), 2)

    def test_extract_branchings_reduces_tree(self):
        """extract_branchings returns a reduced Tree when fewer than existing branchings are requested."""
        D = np.array([
            [0, 1, 2, 9, 9, 9],
            [1, 0, 2, 9, 9, 9],
            [2, 2, 0, 9, 9, 9],
            [9, 9, 9, 0, 1, 2],
            [9, 9, 9, 1, 0, 2],
            [9, 9, 9, 2, 2, 0],
        ], dtype=float)
        # keeper = FakeKeeper(D, labels=[f"x{i}" for i in range(6)])
        labels = [f"x{i}" for i in range(6)]
        keeper = Keeper(distances={"D": pd.DataFrame(data=D, index=labels, columns=labels)})
        poser = POSER(keeper, key="D", root=0, min_branch_size=3, smooth_corr=False)

        # Force two "branched" nodes in branched_ordering by manually appending root
        # and simulating that root has children.
        root = poser.tree.root
        root.tips = np.array([0, 1, 2])
        child1 = TreeNode(name=0, data=np.array([0, 1, 2]), parent=root, nonunique=True, branchable=False,
                          is_trunk=True)
        child1.tips = np.array([0, 2])
        child2 = TreeNode(name=1, data=np.array([3, 4, 5]), parent=root, nonunique=False, branchable=False,
                          is_trunk=False)
        child2.tips = np.array([3, 5])
        poser.tree.insert(child1, parent=root)
        poser.tree.insert(child2, parent=root)

        poser.branched_ordering = [root, root]  # pretend 2 branchings exist (enough for reducer path)

        reduced = poser.extract_branchings(n_branches=1)

        self.assertIsNotNone(reduced.root)
        self.assertLessEqual(len(reduced.nodes), len(poser.tree.nodes))

