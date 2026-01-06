"""
Tests to ensure example datasets load.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ To run a single test module (e.g. test_keeper.py):

$ cd netflow          # project directory
$ python -m unittest tests.test_keeper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import tempfile

from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd

import netflow.keepers.keeper as ks


class TestDataKeeperAndDataView(TestCase):
    def setUp(self):
        self.obs = ["s1", "s2", "s3"]
        self.feats = ["f1", "f2"]
        self.values = np.array([[1.0, 2.0, 3.0],
                                [10.0, 20.0, 30.0]])
        self.df = pd.DataFrame(self.values, index=self.feats, columns=self.obs)

    # ---------- DataKeeper: initialization ----------
    def test_init_with_none(self):
        keeper = ks.DataKeeper()
        self.assertEqual(keeper.data, {})
        self.assertIsNone(keeper.observation_labels)
        self.assertIsNone(keeper.num_observations)

    def test_init_with_observation_labels_must_be_unique(self):
        with self.assertRaises(ValueError):
            ks.DataKeeper(observation_labels=["a", "a"])

    def test_init_with_dataframe_adds_under_default_label(self):
        keeper = ks.DataKeeper(data=self.df)
        self.assertIn("data", keeper)
        self.assertEqual(keeper.observation_labels, self.obs)
        self.assertEqual(keeper.num_observations, 3)

        np.testing.assert_allclose(keeper.data["data"], self.values)
        self.assertEqual(keeper.features_labels["data"], self.feats)
        self.assertEqual(keeper.num_features["data"], 2)

    def test_init_with_ndarray_adds_under_default_label(self):
        arr = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
        keeper = ks.DataKeeper(data=arr)
        self.assertIn("data", keeper)
        self.assertIsNone(keeper.observation_labels)
        self.assertEqual(keeper.num_observations, 3)
        self.assertIsNone(keeper.features_labels["data"])
        self.assertEqual(keeper.num_features["data"], 2)
        np.testing.assert_allclose(keeper.data["data"], arr)

    def test_init_with_dict_adds_each_label(self):
        df2 = pd.DataFrame(
            np.array([[7.0, 8.0, 9.0]]),
            index=["g1"],
            columns=self.obs,
        )
        keeper = ks.DataKeeper(data={"expr": self.df, "geno": df2})
        self.assertIn("expr", keeper)
        self.assertIn("geno", keeper)
        self.assertEqual(keeper.observation_labels, self.obs)
        self.assertEqual(keeper.num_observations, 3)
        self.assertEqual(keeper.num_features["expr"], 2)
        self.assertEqual(keeper.num_features["geno"], 1)

    def test_init_with_unrecognized_type_raises(self):
        with self.assertRaises(TypeError):
            ks.DataKeeper(data=[1, 2, 3])

    # ---------- DataKeeper: add_data validation ----------

    def test_add_data_duplicate_label_raises_key_error(self):
        keeper = ks.DataKeeper()
        keeper.add_data(self.df, "expr")
        with self.assertRaises(KeyError):
            keeper.add_data(self.df, "expr")

    def test_add_data_wrong_type_raises_value_error(self):
        keeper = ks.DataKeeper()
        with self.assertRaises(ValueError):
            keeper.add_data([1, 2, 3], "expr")  # not ndarray or DataFrame

    def test_add_data_inconsistent_num_observations_raises(self):
        keeper = ks.DataKeeper(observation_labels=["a", "b", "c"])
        bad = pd.DataFrame(np.ones((2, 2)), index=["x", "y"], columns=["a", "b"])
        with self.assertRaises(ValueError):
            keeper.add_data(bad, "bad")

    def test_add_data_dataframe_requires_unique_columns_when_labels_none(self):
        keeper = ks.DataKeeper()
        df_bad_cols = pd.DataFrame(
            np.ones((2, 3)),
            index=["f1", "f2"],
            columns=["s1", "s1", "s3"],  # duplicate
        )
        with self.assertRaises(ValueError):
            keeper.add_data(df_bad_cols, "expr")

    def test_add_data_dataframe_requires_unique_index(self):
        keeper = ks.DataKeeper()
        df_bad_idx = pd.DataFrame(
            np.ones((2, 3)),
            index=["f1", "f1"],  # duplicate
            columns=self.obs,
        )
        with self.assertRaises(ValueError):
            keeper.add_data(df_bad_idx, "expr")

    def test_add_data_second_dataset_must_match_num_observations(self):
        keeper = ks.DataKeeper(data=self.df)
        bad_arr = np.ones((5, 2))  # 2 obs, but keeper has 3
        with self.assertRaises(ValueError):
            keeper.add_data(bad_arr, "bad")

    # ---------- DataKeeper: mapping/iterating behaviors ----------

    def test_contains_getitem_and_missing_key(self):
        keeper = ks.DataKeeper(data=self.df)
        self.assertTrue("data" in keeper)
        view = keeper["data"]
        self.assertEqual(view.label, "data")

        with self.assertRaises(KeyError):
            _ = keeper["nope"]

    def test_iter_yields_dataviews(self):
        keeper = ks.DataKeeper(data={"a": self.df, "b": self.df})
        labels = sorted([v.label for v in keeper])
        self.assertEqual(labels, ["a", "b"])

    def test_keys_and_items(self):
        keeper = ks.DataKeeper(data={"a": self.df, "b": self.df})
        self.assertEqual(set(keeper.keys()), {"a", "b"})
        self.assertEqual(set(dict(keeper.items()).keys()), {"a", "b"})

    def test_observation_index(self):
        keeper = ks.DataKeeper(data=self.df)
        self.assertEqual(keeper.observation_index("s2"), 1)

    # ---------- DataView behaviors ----------

    def test_dataview_to_frame_roundtrip_matches_original(self):
        keeper = ks.DataKeeper(data=self.df)
        view = keeper["data"]
        df2 = view.to_frame()

        self.assertEqual(df2.columns.tolist(), self.obs)
        self.assertEqual(df2.index.tolist(), self.feats)
        np.testing.assert_allclose(df2.values, self.values)

    def test_dataview_feature_and_observation_index(self):
        keeper = ks.DataKeeper(data=self.df)
        view = keeper["data"]
        self.assertEqual(view.observation_index("s3"), 2)
        self.assertEqual(view.feature_index("f2"), 1)

    def test_dataview_subset_by_labels(self):
        keeper = ks.DataKeeper(data=self.df)
        view = keeper["data"]

        sub = view.subset(observations=["s3", "s1"], features=["f2"])
        self.assertEqual(sub.columns.tolist(), ["s3", "s1"])
        self.assertEqual(sub.index.tolist(), ["f2"])
        np.testing.assert_allclose(sub.values, np.array([[30.0, 10.0]]))

    def test_dataview_subset_by_indices(self):
        keeper = ks.DataKeeper(data=self.df)
        view = keeper["data"]

        sub = view.subset(observations=[2, 0], features=[1])
        self.assertEqual(sub.columns.tolist(), [2, 0])  # uses provided labels verbatim
        self.assertEqual(sub.index.tolist(), [1])
        np.testing.assert_allclose(sub.values, np.array([[30.0, 10.0]]))

    def test_dataview_subset_requires_observations_or_features(self):
        keeper = ks.DataKeeper(data=self.df)
        view = keeper["data"]
        with self.assertRaises(ValueError):
            view.subset()

    def test_dataview_standardize_rows_mean0_std1(self):
        keeper = ks.DataKeeper(data=self.df)
        view = keeper["data"]

        z = view.standardize()
        self.assertEqual(z.shape, self.df.shape)
        self.assertEqual(z.columns.tolist(), self.obs)
        self.assertEqual(z.index.tolist(), self.feats)

        # Each feature (row) should be standardized across observations
        means = z.values.mean(axis=1)
        stds = z.values.std(axis=1, ddof=0)  # StandardScaler uses population variance
        np.testing.assert_allclose(means, np.zeros_like(means), atol=1e-12)
        np.testing.assert_allclose(stds, np.ones_like(stds), atol=1e-12)

    # ---------- DataKeeper.subset integration ----------
    def test_datakeeper_subset_returns_new_keeper_with_subset(self):
        keeper = ks.DataKeeper(data={"expr": self.df})
        sub_keeper = keeper.subset(["s1", "s3"])

        self.assertEqual(sub_keeper.observation_labels, ["s1", "s3"])
        self.assertIn("expr", sub_keeper)

        sub_view = sub_keeper["expr"]
        sub_df = sub_view.to_frame()
        self.assertEqual(sub_df.columns.tolist(), ["s1", "s3"])
        self.assertEqual(sub_df.index.tolist(), self.feats)
        np.testing.assert_allclose(sub_df.values, self.df[["s1", "s3"]].values)

    # ---------- DataKeeper.standardize ----------
    def test_datakeeper_standardize(self):
        keeper = ks.DataKeeper(data=self.df)

        out = keeper.standardize("data")
        expected = keeper["data"].standardize()

        pd.testing.assert_frame_equal(out, expected)

    def test_datakeeper_standardize_delegation_with_kwargs(self):
        keeper = ks.DataKeeper(data=self.df)

        out = keeper.standardize("data", with_mean=False)
        expected = keeper["data"].standardize(with_mean=False)

        pd.testing.assert_frame_equal(out, expected)


class TestDistanceKeeperAndDistanceView(TestCase):
    def setUp(self):
        self.obs = ["a", "b", "c"]
        self.D = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ])
        self.df = pd.DataFrame(self.D, index=self.obs, columns=self.obs)

    # ------------------ DistanceKeeper: initialization ------------------

    def test_init_with_none(self):
        k = ks.DistanceKeeper()
        self.assertEqual(k.data, {})
        self.assertIsNone(k.observation_labels)
        self.assertIsNone(k.num_observations)

    def test_init_with_observation_labels_must_be_unique(self):
        with self.assertRaises(ValueError):
            ks.DistanceKeeper(observation_labels=["x", "x"])

    def test_init_with_dataframe_sets_labels_and_stores_values(self):
        orig_nan = ks.checks.check_matrix_no_nan
        orig_dist = ks.checks.check_distance_matrix

        with patch.object(ks.checks, "check_matrix_no_nan", wraps=orig_nan) as m_nan, \
                patch.object(ks.checks, "check_distance_matrix", wraps=orig_dist) as m_dist:
            k = ks.DistanceKeeper(data=self.df)

            self.assertIn("distance", k)
            self.assertEqual(k.observation_labels, self.obs)
            self.assertEqual(k.num_observations, 3)
            np.testing.assert_allclose(k.data["distance"], self.D)

            m_nan.assert_called_once()
            m_dist.assert_called_once()

    def test_init_with_dataframe_nan_fails_real_check(self):
        bad = self.df.copy()
        bad.iloc[0, 1] = np.nan

        orig_nan = ks.checks.check_matrix_no_nan
        orig_dist = ks.checks.check_distance_matrix

        with patch.object(ks.checks, "check_matrix_no_nan", wraps=orig_nan) as m_nan, \
                patch.object(ks.checks, "check_distance_matrix", wraps=orig_dist) as m_dist:
            with self.assertRaises(AssertionError):
                ks.DistanceKeeper(data=bad)

            m_nan.assert_called_once()
            # likely fails before distance check runs:
            self.assertEqual(m_dist.call_count, 0)

    def test_init_with_dataframe_invalid_distance_fails_real_check(self):
        bad = self.df.copy()
        bad.iloc[0, 1] = 999.0  # breaks symmetry if [1,0] unchanged

        orig_nan = ks.checks.check_matrix_no_nan
        orig_dist = ks.checks.check_distance_matrix

        with patch.object(ks.checks, "check_matrix_no_nan", wraps=orig_nan) as m_nan, \
                patch.object(ks.checks, "check_distance_matrix", wraps=orig_dist) as m_dist:
            with self.assertRaises(AssertionError):
                ks.DistanceKeeper(data=bad)

            m_nan.assert_called_once()
            m_dist.assert_called_once()

    @patch.object(ks.checks, "check_matrix_no_nan", autospec=True)
    @patch.object(ks.checks, "check_distance_matrix", autospec=True)
    def test_init_with_ndarray_sets_num_observations(self, m_dist, m_nan):
        k = ks.DistanceKeeper(data=self.D)
        self.assertIn("distance", k)
        self.assertIsNone(k.observation_labels)  # current behavior in provided code
        self.assertEqual(k.num_observations, 3)
        np.testing.assert_allclose(k.data["distance"], self.D)

        m_nan.assert_called_once()
        m_dist.assert_called_once()

    @patch.object(ks.checks, "check_matrix_no_nan", autospec=True)
    @patch.object(ks.checks, "check_distance_matrix", autospec=True)
    def test_init_with_dict_adds_each_label(self, m_dist, m_nan):
        k = ks.DistanceKeeper(data={"d1": self.df, "d2": self.df})
        self.assertIn("d1", k)
        self.assertIn("d2", k)
        self.assertEqual(k.num_observations, 3)
        # called once per add_data
        self.assertEqual(m_nan.call_count, 2)
        self.assertEqual(m_dist.call_count, 2)

    def test_init_with_unrecognized_type_raises(self):
        with self.assertRaises(TypeError):
            ks.DistanceKeeper(data=[1, 2, 3])

    # ------------------ DistanceKeeper.add_data: validation ------------------

    @patch.object(ks.checks, "check_matrix_no_nan", autospec=True)
    @patch.object(ks.checks, "check_distance_matrix", autospec=True)
    def test_add_data_duplicate_label_raises_keyerror(self, m_dist, m_nan):
        k = ks.DistanceKeeper()
        k.add_data(self.df, "d")
        with self.assertRaises(KeyError):
            k.add_data(self.df, "d")

    def test_add_data_wrong_type_raises_valueerror(self):
        k = ks.DistanceKeeper()
        with self.assertRaises(ValueError):
            k.add_data([1, 2, 3], "d")  # not ndarray/DataFrame

    def test_add_data_inconsistent_observation_labels_length_raises(self):
        k = ks.DistanceKeeper(observation_labels=["a", "b"])  # length 2
        with self.assertRaises(ValueError):
            k.add_data(self.df, "d")  # 3x3, mismatch with num_observations=2

    def test_add_data_requires_square_matrix(self):
        k = ks.DistanceKeeper(observation_labels=["a", "b", "c"])
        nonsquare = np.ones((3, 2))
        with self.assertRaises(ValueError):
            k.add_data(nonsquare, "d")

    def test_add_data_dataframe_requires_unique_columns_when_labels_none(self):
        k = ks.DistanceKeeper()
        bad = pd.DataFrame(
            self.D,
            index=self.obs,
            columns=["a", "a", "c"],  # duplicate column label
        )
        with self.assertRaises(ValueError):
            k.add_data(bad, "d")

    @patch.object(ks.checks, "check_matrix_no_nan", autospec=True)
    @patch.object(ks.checks, "check_distance_matrix", autospec=True)
    def test_add_data_dataframe_reorders_to_observation_labels(self, m_dist, m_nan):
        # DataFrame is in different order; keeper should reorder to observation_labels
        perm = ["b", "c", "a"]
        df_perm = self.df.loc[perm, perm]

        k = ks.DistanceKeeper(observation_labels=self.obs)
        k.add_data(df_perm, "d")

        # should match original D in obs order
        np.testing.assert_allclose(k.data["d"], self.D)

    @patch.object(ks.checks, "check_matrix_no_nan", autospec=True)
    @patch.object(ks.checks, "check_distance_matrix", autospec=True)
    def test_add_data_second_matrix_must_match_num_observations(self, m_dist, m_nan):
        k = ks.DistanceKeeper(data=self.df)
        bad = np.ones((4, 4))
        with self.assertRaises(ValueError):
            k.add_data(bad, "bad")

    # ------------------ DistanceKeeper: mapping/iterating behaviors ------------------

    def test_contains_getitem_and_missing_key(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data=self.df)

        self.assertTrue("distance" in k)
        view = k["distance"]
        self.assertEqual(view.label, "distance")

        with self.assertRaises(KeyError):
            _ = k["nope"]

    def test_iter_yields_distanceviews(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data={"d1": self.df, "d2": self.df})

        labels = sorted([v.label for v in k])
        self.assertEqual(labels, ["d1", "d2"])

    def test_keys_and_items(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data={"d1": self.df, "d2": self.df})

        self.assertEqual(set(k.keys()), {"d1", "d2"})
        self.assertEqual(set(dict(k.items()).keys()), {"d1", "d2"})

    def test_observation_index(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data=self.df)
        self.assertEqual(k.observation_index("b"), 1)

    # ------------------ add_stacked_data ------------------

    def test_add_stacked_data_calls_unstack_and_adds(self):
        k = ks.DistanceKeeper(observation_labels=self.obs)

        # pretend we have stacked upper-tri distances in a Series (actual content doesn't matter for this test)
        idx = pd.MultiIndex.from_tuples([("a", "b"), ("a", "c"), ("b", "c")])
        stacked = pd.Series([1.0, 2.0, 3.0], index=idx)

        unstacked_df = self.df  # what unstack_triu_ should return

        with patch.object(ks, "unstack_triu_", autospec=True, return_value=unstacked_df) as m_unstack, \
                patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k.add_stacked_data(stacked, label="dstack", diag=0.0)

        m_unstack.assert_called_once()
        self.assertIn("dstack", k)
        np.testing.assert_allclose(k.data["dstack"], self.D)

    # ------------------ DistanceKeeper.subset integration ------------------

    def test_distancekeeper_subset_by_labels(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data={"d": self.df})

        sub = k.subset(["c", "a"])
        self.assertEqual(sub.observation_labels, ["c", "a"])
        self.assertIn("d", sub)

        # expected slice
        exp = self.df.loc[["c", "a"], ["c", "a"]].values
        np.testing.assert_allclose(sub.data["d"], exp)

    def test_distancekeeper_subset_by_indices(self):
        # When observation_labels is None, subset should be by integer indices
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data={"d": self.D})

        sub = k.subset([2, 0])
        self.assertEqual(sub.observation_labels, [2, 0])
        exp = self.D[np.ix_([2, 0], [2, 0])]
        np.testing.assert_allclose(sub.data["d"], exp)

    # ------------------ DistanceView behaviors ------------------

    def test_distanceview_to_frame_roundtrip(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data=self.df)

        v = k["distance"]
        f = v.to_frame()
        self.assertEqual(f.index.tolist(), self.obs)
        self.assertEqual(f.columns.tolist(), self.obs)
        np.testing.assert_allclose(f.values, self.D)

    def test_distanceview_subset_symmetric_default(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data=self.df)

        v = k["distance"]
        sub = v.subset(observations_a=["b", "a"])
        self.assertEqual(sub.index.tolist(), ["b", "a"])
        self.assertEqual(sub.columns.tolist(), ["b", "a"])
        np.testing.assert_allclose(sub.values, self.df.loc[["b", "a"], ["b", "a"]].values)

    def test_distanceview_subset_rectangular(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data=self.df)

        v = k["distance"]
        sub = v.subset(observations_a=["a", "c"], observations_b=["b"])
        self.assertEqual(sub.index.tolist(), ["a", "c"])
        self.assertEqual(sub.columns.tolist(), ["b"])
        np.testing.assert_allclose(sub.values, self.df.loc[["a", "c"], ["b"]].values)

    def test_distanceview_observation_index(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data=self.df)

        v = k["distance"]
        self.assertEqual(v.observation_index("c"), 2)

    def test_distanceview_density(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.DistanceKeeper(data=self.df)

        v = k["distance"]
        dens = v.density()
        # density = column sums of distance matrix
        expected = self.df.sum(axis=0)
        pd.testing.assert_series_equal(dens, expected)

class TestGraphKeeper(TestCase):
    def setUp(self):
        self.g1 = nx.Graph()
        self.g1.add_edges_from([(1, 2), (2, 3)])

        self.g2 = nx.Graph()
        self.g2.add_edges_from([("a", "b")])

        # ---------- __init__ ----------

        def test_init_with_none(self):
            k = ks.GraphKeeper()
            self.assertEqual(k.graphs, {})
            self.assertEqual(list(k), [])

        def test_init_with_single_graph_uses_default_label_and_sets_name(self):
            k = ks.GraphKeeper(self.g1)

            self.assertIn("graph", k)
            self.assertIs(k["graph"], self.g1)
            self.assertEqual(self.g1.name, "graph")
            self.assertEqual(set(k.keys()), {"graph"})

        def test_init_with_dict_adds_all_labels_and_sets_names(self):
            graphs = {"g1": self.g1, "g2": self.g2}
            k = ks.GraphKeeper(graphs)

            self.assertEqual(set(k.keys()), {"g1", "g2"})
            self.assertIs(k["g1"], self.g1)
            self.assertIs(k["g2"], self.g2)
            self.assertEqual(self.g1.name, "g1")
            self.assertEqual(self.g2.name, "g2")

        def test_init_with_unrecognized_type_raises(self):
            with self.assertRaises(TypeError):
                ks.GraphKeeper(graphs=["not", "a", "graph"])

        # ---------- add_graph ----------

        def test_add_graph_adds_graph_sets_name_and_is_retrievable(self):
            k = ks.GraphKeeper()
            k.add_graph(self.g1, "my_graph")

            self.assertIn("my_graph", k)
            self.assertIs(k["my_graph"], self.g1)
            self.assertEqual(self.g1.name, "my_graph")

        def test_add_graph_duplicate_label_raises_keyerror(self):
            k = ks.GraphKeeper()
            k.add_graph(self.g1, "dup")
            with self.assertRaises(KeyError):
                k.add_graph(self.g2, "dup")

        def test_add_graph_wrong_type_raises_valueerror(self):
            k = ks.GraphKeeper()
            with self.assertRaises(ValueError):
                k.add_graph(graph={"not": "a graph"}, label="bad")

        # ---------- container behaviors ----------

        def test_getitem_missing_key_raises_keyerror(self):
            k = ks.GraphKeeper()
            with self.assertRaises(KeyError):
                _ = k["missing"]

        def test_iter_yields_graph_objects(self):
            k = ks.GraphKeeper({"g1": self.g1, "g2": self.g2})
            yielded = list(iter(k))

            # __iter__ yields graphs (values), not keys
            self.assertEqual(len(yielded), 2)
            self.assertIn(self.g1, yielded)
            self.assertIn(self.g2, yielded)

        def test_keys_and_items(self):
            k = ks.GraphKeeper({"g1": self.g1, "g2": self.g2})

            self.assertEqual(set(k.keys()), {"g1", "g2"})
            items = dict(k.items())
            self.assertEqual(set(items.keys()), {"g1", "g2"})
            self.assertIs(items["g1"], self.g1)
            self.assertIs(items["g2"], self.g2)

        def test_graphs_property_returns_underlying_dict(self):
            k = ks.GraphKeeper({"g1": self.g1})
            self.assertIsInstance(k.graphs, dict)
            self.assertIs(k.graphs["g1"], self.g1)


class TestKeeper(TestCase):
    def setUp(self):
        self.obs = ["a", "b", "c"]
        self.feats = ["f1", "f2"]

        self.X = np.array([[1.0, 2.0, 3.0],
                           [10.0, 20.0, 30.0]])
        self.df_data = pd.DataFrame(self.X, index=self.feats, columns=self.obs)

        self.D = np.array([[0.0, 1.0, 2.0],
                           [1.0, 0.0, 3.0],
                           [2.0, 3.0, 0.0]])
        self.df_dist = pd.DataFrame(self.D, index=self.obs, columns=self.obs)

        self.S = np.array([[1.0, 0.2, 0.1],
                           [0.2, 1.0, 0.4],
                           [0.1, 0.4, 1.0]])
        self.df_sim = pd.DataFrame(self.S, index=self.obs, columns=self.obs)

# -------------------- init / label inference --------------------

    def test_init_infers_observation_labels_from_data_dataframe(self):
        k = ks.Keeper(data=self.df_data)
        self.assertEqual(k.observation_labels, self.obs)
        self.assertEqual(k.num_observations, 3)

    def test_init_infers_default_labels_from_ndarray(self):
        k = ks.Keeper(data=self.X)
        self.assertEqual(k.observation_labels, ["X0", "X1", "X2"])
        self.assertEqual(k.num_observations, 3)

    def test_init_infers_observation_labels_from_distances_when_no_data(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
             patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.Keeper(distances=self.df_dist)

        self.assertEqual(k.observation_labels, self.obs)
        self.assertEqual(k.num_observations, 3)

    def test_init_duplicate_observation_labels_raises(self):
        with self.assertRaises(ValueError):
            ks.Keeper(observation_labels=["a", "a", "b"])

    def test_init_creates_outdir_if_missing(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "results"
            self.assertFalse(outdir.exists())
            k = ks.Keeper(outdir=outdir)
            self.assertTrue(outdir.exists())
            self.assertTrue(outdir.is_dir())
            self.assertEqual(k.outdir, outdir)

    # -------------------- adders (wiring) --------------------

    def test_add_data_initializes_obs_labels_when_empty(self):
        k = ks.Keeper()
        k.add_data(self.X, label="expr")

        self.assertEqual(k.num_observations, 3)
        self.assertEqual(k.observation_labels, ["X0", "X1", "X2"])
        self.assertIn("expr", k.data)

    def test_add_distance_initializes_other_keepers_when_empty(self):
        k = ks.Keeper()
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
             patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k.add_distance(self.D, label="d1")

        self.assertEqual(k.num_observations, 3)
        self.assertEqual(k.observation_labels, ["X0", "X1", "X2"])
        self.assertIn("d1", k.distances)
        self.assertEqual(k.data.observation_labels, ["X0", "X1", "X2"])
        self.assertEqual(k.similarities.observation_labels, ["X0", "X1", "X2"])

    def test_add_graph_routes_to_graph_keeper(self):
        k = ks.Keeper()
        G = nx.Graph()
        G.add_edge("u", "v")
        k.add_graph(G, "g1")
        self.assertIn("g1", k.graphs)
        self.assertIs(k.graphs["g1"], G)
        self.assertEqual(G.name, "g1")

    def test_add_misc_duplicate_label_raises(self):
        k = ks.Keeper()
        k.add_misc(pd.DataFrame({"x": [1]}), "meta")
        with self.assertRaises(KeyError):
            k.add_misc(pd.DataFrame({"x": [2]}), "meta")

    # -------------------- subset --------------------

    def test_subset_defaults_exclude_misc_and_graphs(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
             patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.Keeper(data=self.df_data, distances=self.df_dist, similarities=self.df_sim)

        k.add_misc(pd.DataFrame({"m": [1, 2, 3]}, index=self.obs), "meta")
        G = nx.Graph()
        G.add_edge("a", "b")
        k.add_graph(G, "g")

        sub = k.subset(["c", "a"])

        self.assertEqual(sub.observation_labels, ["c", "a"])
        self.assertEqual(sub.num_observations, 2)
        self.assertEqual(sub.misc, {})
        self.assertEqual(sub.graphs.graphs, {})

    def test_subset_can_keep_misc_and_graphs(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
             patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.Keeper(data=self.df_data, distances=self.df_dist, similarities=self.df_sim)

        k.add_misc(pd.DataFrame({"m": [1, 2, 3]}, index=self.obs), "meta")
        G = nx.Graph()
        G.add_edge("a", "b")
        k.add_graph(G, "g")

        sub = k.subset(["a", "b"], keep_misc=True, keep_graphs=True)
        self.assertIn("meta", sub.misc)
        self.assertIn("g", sub.graphs)

    # -------------------- density helpers --------------------

    def test_distance_density_and_argmin_argmax(self):
        with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
             patch.object(ks.checks, "check_distance_matrix", autospec=True):
            k = ks.Keeper(distances=self.df_dist)

        dens = k.distance_density("distance")
        self.assertEqual(dens.loc["a"], 3.0)
        self.assertEqual(dens.loc["b"], 4.0)
        self.assertEqual(dens.loc["c"], 5.0)

        self.assertEqual(k.distance_density_argmin("distance"), 0)
        self.assertEqual(k.distance_density_argmax("distance"), 2)

    # -------------------- load_* integration tests (csv/tsv/xlsx) --------------------

    def test_load_data_csv_tsv_xlsx(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            cases = [
                ("csv", ","),
                ("tsv", "\t"),
                ("xlsx", None),
            ]

            for ext, delim in cases:
                with self.subTest(ext=ext):
                    k = ks.Keeper()

                    fp = td / f"data.{ext}"
                    if ext == "xlsx":
                        self.df_data.to_excel(fp, index=True)
                        k.load_data(fp, label="expr", file_format=None, index_col=0)  # read_excel kwargs
                    else:
                        self.df_data.to_csv(fp, sep=delim, header=True, index=True)
                        k.load_data(fp, label="expr", delimiter=delim, header=0, index_col=0)

                    got = k.data["expr"].to_frame()
                    if ext == "xlsx":
                        # NOTE: pd.read_excel will default to load as int and not float if possible.
                        # If dtype matters, this should be hardcoded into the load method.
                        pd.testing.assert_frame_equal(got, self.df_data, check_dtype=False)
                    else:
                        pd.testing.assert_frame_equal(got, self.df_data)

    def test_load_data_using_file_path_and_file_format_appending(self):
        # Exercises load_from_file branch: file_format not None => appends extension
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fp = td / "expr.csv"
            self.df_data.to_csv(fp, sep=",", header=True, index=True)

            k = ks.Keeper()
            k.load_data(
                file_name="expr",   # no extension
                file_path=td,
                file_format="csv",  # should append -> expr.csv
                delimiter=",",
                header=0,
                index_col=0,
                label="expr",
            )
            pd.testing.assert_frame_equal(k.data["expr"].to_frame(), self.df_data)

    def test_load_distance_csv_tsv_xlsx(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            cases = [
                ("csv", ","),
                ("tsv", "\t"),
                ("xlsx", None),
            ]

            for ext, delim in cases:
                with self.subTest(ext=ext):
                    fp = td / f"dist.{ext}"
                    if ext == "xlsx":
                        self.df_dist.to_excel(fp, index=True)
                    else:
                        self.df_dist.to_csv(fp, sep=delim, header=True, index=True)

                    with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                         patch.object(ks.checks, "check_distance_matrix", autospec=True):
                        k = ks.Keeper()
                        if ext == "xlsx":
                            k.load_distance(fp, label="d1")  # load_distance passes header/index_col
                        else:
                            k.load_distance(fp, label="d1", delimiter=delim)

                    np.testing.assert_allclose(k.distances.data["d1"], self.D)
                    self.assertEqual(k.observation_labels, self.obs)

    def test_load_similarity_csv_tsv_xlsx(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            cases = [
                ("csv", ","),
                ("tsv", "\t"),
                ("xlsx", None),
            ]

            for ext, delim in cases:
                with self.subTest(ext=ext):
                    fp = td / f"sim.{ext}"
                    if ext == "xlsx":
                        self.df_sim.to_excel(fp, index=True)
                    else:
                        self.df_sim.to_csv(fp, sep=delim, header=True, index=True)

                    with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                            patch.object(ks.checks, "check_distance_matrix", autospec=True):
                        k = ks.Keeper()
                        if ext == "xlsx":
                            k.load_similarity(fp, label="s1")
                        else:
                            k.load_similarity(fp, label="s1", delimiter=delim)

                    np.testing.assert_allclose(k.similarities.data["s1"], self.S)
                    self.assertEqual(k.observation_labels, self.obs)

    def test_load_graph_edgelist_csv_tsv_xlsx(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            E = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})

            cases = [
                ("csv", ","),
                ("tsv", "\t"),
                ("xlsx", None),
            ]

            for ext, delim in cases:
                with self.subTest(ext=ext):
                    fp = td / f"edges.{ext}"
                    if ext == "xlsx":
                        E.to_excel(fp, index=False)
                    else:
                        E.to_csv(fp, sep=delim, header=True, index=False)

                    k = ks.Keeper()
                    if ext == "xlsx":
                        k.load_graph(fp, label="g1")  # read_excel via load_from_file
                    else:
                        k.load_graph(fp, label="g1", delimiter=delim)

                    G = k.graphs["g1"]
                    self.assertTrue(G.has_edge("a", "b"))
                    self.assertTrue(G.has_edge("b", "c"))

    # -------------------- save_* integration tests (txt/csv/tsv) --------------------

    def test_save_data_distance_similarity_misc_txt_csv_tsv(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td) / "out"

            with patch.object(ks.checks, "check_matrix_no_nan", autospec=True), \
                    patch.object(ks.checks, "check_distance_matrix", autospec=True):
                k = ks.Keeper(data=self.df_data, distances=self.df_dist, similarities=self.df_sim, outdir=outdir)

            misc_df = pd.DataFrame({"m": [1, 2, 3]}, index=self.obs)
            k.add_misc(misc_df, "meta")

            for ext, delim in [("csv", ","), ("tsv", "\t"), ("txt", ",")]:
                with self.subTest(ext=ext):
                    k.save_data("data", file_format=ext, delimiter=delim)
                    got = pd.read_csv(outdir / f"data_data.{ext}", sep=delim, header=0, index_col=0)
                    pd.testing.assert_frame_equal(got, self.df_data)

                    k.save_distance("distance", file_format=ext, delimiter=delim)
                    got = pd.read_csv(outdir / f"distance_distance.{ext}", sep=delim, header=0, index_col=0)
                    pd.testing.assert_frame_equal(got, self.df_dist)

                    # similarities: default label in DistanceKeeper is also 'distance' for single matrix init
                    k.save_similarity("distance", file_format=ext, delimiter=delim)
                    got = pd.read_csv(outdir / f"similarity_distance.{ext}", sep=delim, header=0, index_col=0)
                    pd.testing.assert_frame_equal(got, self.df_sim)

                    k.save_misc("meta", file_format=ext, delimiter=delim)
                    got = pd.read_csv(outdir / f"misc_meta.{ext}", sep=delim, header=0, index_col=0)
                    pd.testing.assert_frame_equal(got, misc_df)

    def test_save_raises_when_outdir_none(self):
        k = ks.Keeper(data=self.df_data)  # outdir None
        with self.assertRaises(ValueError):
            k.save_data("data", file_format="csv")

    def test_save_raises_when_file_exists(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            k = ks.Keeper(data=self.df_data, outdir=outdir)

            # Pre-create the expected output file
            (outdir / "data_data.csv").write_text("already exists")

            with self.assertRaises(ValueError):
                k.save_data("data", file_format="csv")



