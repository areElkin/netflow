"""
Tests transitions for multi-scale and scale-free distance computations.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ To run a single test module (e.g. test_transitions.py):

$ cd netflow          # project directory
$ python -m unittest tests.test_transitions

~ To run all test modules:

$ cd netflow          # project directory
$ python -m unittest -v
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from netflow.keepers.keeper import Keeper
from netflow.pose.organization import (
    _compute_transitions,
    compute_transitions,
    compute_rw_transitions,
    compute_sym_diffusion_affinity_transitions,
    compute_multiscale_VNE_transitions_from_similarity,
)


# class FakeKeeper:
#     """Minimal Keeper stub sufficient for transition-matrix functions."""
#     def __init__(self, similarities):
#         self.similarities = similarities  # dict: key -> obj(data=ndarray)
#         self.misc = {}
#
#     def add_misc(self, arr, key):
#         self.misc[key] = arr


class TestTransitions(unittest.TestCase):
    def test__compute_transitions_shapes_and_diagonal(self):
        """_compute_transitions returns (asym, sym) of correct shape and 1 diagonal."""
        W = np.array(
            [
                [1.0, 0.1, 0.2],
                [0.1, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ],
            dtype=float,
        )

        asym, sym = _compute_transitions(W, density_normalize=True)

        self.assertEqual(asym.shape, W.shape)
        self.assertEqual(sym.shape, W.shape)
        self.assertTrue(np.allclose(np.diag(asym), 0.0))
        self.assertTrue(np.allclose(np.diag(sym), 0.0))

        asym, sym = _compute_transitions(W, density_normalize=False)

        self.assertEqual(asym.shape, W.shape)
        self.assertEqual(sym.shape, W.shape)
        self.assertTrue(np.allclose(np.diag(asym), 0.0))
        self.assertTrue(np.allclose(np.diag(sym), 0.0))

    def test_compute_transitions_saves_expected_keys(self):
        """compute_transitions writes transitions into keeper.misc with expected key names and suffixes."""
        W = np.array(
            [
                [1.0, 0.1, 0.2],
                [0.1, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ],
            dtype=float,
        )

        # keeper = FakeKeeper(similarities={"S": SimpleNamespace(data=W)})
        keeper = Keeper(similarities={"S": W})

        compute_transitions(keeper, "S", density_normalize=True)

        self.assertIn("transitions_asym_S_density_normalized", keeper.misc)
        self.assertIn("transitions_sym_S_density_normalized", keeper.misc)

        A = keeper.misc["transitions_asym_S_density_normalized"]
        B = keeper.misc["transitions_sym_S_density_normalized"]
        self.assertEqual(A.shape, W.shape)
        self.assertEqual(B.shape, W.shape)

        # test with density_normalize=False
        keeper = Keeper(similarities={"S": W})

        compute_transitions(keeper, "S", density_normalize=False)

        self.assertIn("transitions_asym_S", keeper.misc)
        self.assertIn("transitions_sym_S", keeper.misc)

        A = keeper.misc["transitions_asym_S"]
        B = keeper.misc["transitions_sym_S"]
        self.assertEqual(A.shape, W.shape)
        self.assertEqual(B.shape, W.shape)

    def test_compute_rw_transitions_row_stochastic(self):
        """compute_rw_transitions returns row-stochastic P and saves it when do_save=True."""
        W = np.array(
            [
                [1.0, 0.1, 0.2],
                [0.1, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ],
            dtype=float,
        )

        keeper = Keeper(similarities={"S": W})

        P = compute_rw_transitions(keeper, "S", do_save=True)

        self.assertIn("transitions_rw_S", keeper.misc)
        self.assertTrue(np.allclose(P.sum(axis=1), 1.0))

        # test with enforced nonlazy walk:
        keeper = Keeper(similarities={"S": W})

        P = compute_rw_transitions(keeper, "S", allow_lazy_rw=False, do_save=True)

        self.assertIn("transitions_rw_nonlazyrw_S", keeper.misc)
        self.assertTrue(np.allclose(P.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(np.diag(P), 0.0))

    def test_compute_sym_diffusion_affinity_transitions(self):
        """compute_sym_diffusion_affinity_transitions returns symmetric affinity normalization and saves it."""
        W = np.array(
            [
                [1.0, 0.1, 0.2],
                [0.1, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ],
            dtype=float,
        )

        keeper = Keeper(similarities={"S": W})

        P = compute_sym_diffusion_affinity_transitions(keeper, "S", do_save=True)

        self.assertIn("transitions_sym_diff_aff_S", keeper.misc)
        self.assertEqual(P.shape, W.shape)

        # test with enforced nonlazy walk:
        keeper = Keeper(similarities={"S": W})

        P = compute_sym_diffusion_affinity_transitions(keeper, "S", allow_lazy_rw=False, do_save=True)

        self.assertIn("transitions_sym_diff_aff_nonlazyrw_S", keeper.misc)
        self.assertEqual(P.shape, W.shape)
        self.assertTrue(np.allclose(np.diag(P), 0.0))


class TestMultiscaleVNE(unittest.TestCase):
    @patch("netflow.pose.organization.utl.find_knee_point", return_value=2)
    @patch("netflow.pose.organization.utl.von_neumann_entropy", return_value=np.array([1.0, 0.5, 0.25]))
    def test_multiscale_vne_saves_and_caches(self, *_):
        """
        compute_multiscale_VNE_transitions_from_similarity saves both
        sym and rw outputs and reuses cache if present.
        """
        K = np.array(
            [
                [1.0, 0.1, 0.2],
                [0.1, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ],
            dtype=float,
        )

        keeper = Keeper(similarities={"S": K})
        P_sym_1, P_rw_1 = compute_multiscale_VNE_transitions_from_similarity(
            keeper, "S", tau_max=10, do_save=True,
        )

        self.assertIn("transitions_sym_multiscaleVNE_S", keeper.misc)
        self.assertIn("transitions_multiscaleVNE_S", keeper.misc)

        P_sym_2, P_rw_2 = compute_multiscale_VNE_transitions_from_similarity(
            keeper, "S", tau_max=10, do_save=True
        )

        self.assertTrue(np.allclose(P_sym_1, P_sym_2))
        self.assertTrue(np.allclose(P_rw_1, P_rw_2))