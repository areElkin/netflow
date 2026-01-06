"""
Tests Tree and TreeNode classes for representing the branching process
for constructing the POSE backbone.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~ To run a single test module (e.g. test_tree.py):

$ cd netflow          # project directory
$ python -m unittest tests.test_tree

~ To run all test modules:

$ cd netflow          # project directory
$ python -m unittest -v
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from netflow.pose.organization import Tree, TreeNode


class TestTreeNode(unittest.TestCase):
    def test_is_root_and_is_leaf(self):
        """TreeNode.is_root and TreeNode.is_leaf reflect presence/absence of parent/children."""
        root = TreeNode(name="root", data=[1, 2, 3])
        self.assertTrue(root.is_root())
        self.assertTrue(root.is_leaf())

        child = TreeNode(name="child", data=[4])
        root.add_child(child)

        self.assertTrue(root.is_root())
        self.assertFalse(root.is_leaf())
        self.assertFalse(child.is_root())
        self.assertTrue(child.is_leaf())
        self.assertIs(child.parent, root)

    def test_depth(self):
        """TreeNode.depth returns 0 at root and increases by 1 per ancestor."""
        root = TreeNode(name="root", data=[0])
        a = TreeNode(name="a", data=[1])
        b = TreeNode(name="b", data=[2])
        root.add_child(a)
        a.add_child(b)

        self.assertEqual(root.depth(), 0)
        self.assertEqual(a.depth(), 1)
        self.assertEqual(b.depth(), 2)

    def test_add_child_type_check(self):
        """TreeNode.add_child raises TypeError when passed a non-TreeNode."""
        root = TreeNode(name="root", data=[0])
        with self.assertRaises(TypeError):
            root.add_child("not_a_node")


class TestTree(unittest.TestCase):
    def test_insert_root_first(self):
        """Tree.insert requires the first inserted node to be the root (no parent)."""
        t = Tree()
        root = TreeNode(name=0, data=[0, 1, 2])
        t.insert(root)
        self.assertIs(t.root, root)
        self.assertEqual(len(t.nodes), 1)
        self.assertEqual(root._counter, 0)

    def test_insert_with_parent(self):
        """Tree.insert attaches child to parent and assigns unique counters."""
        t = Tree()
        root = TreeNode(name=0, data=[0, 1])
        t.insert(root)

        child = TreeNode(name=1, data=[2, 3])
        t.insert(child, parent=root)

        self.assertEqual(len(t.nodes), 2)
        self.assertEqual(root.children[0], child)
        self.assertIs(child.parent, root)
        self.assertEqual(root._counter, 0)
        self.assertEqual(child._counter, 1)

    def test_insert_parent_requires_existing_root(self):
        """Tree.insert raises if a parent is given before a root exists."""
        t = Tree()
        parent = TreeNode(name=0, data=[0])
        child = TreeNode(name=1, data=[1])
        with self.assertRaises(AssertionError):
            t.insert(child, parent=parent)

    def test_get_node_from_name_returns_none_if_missing(self):
        """Tree.get_node_from_name returns None when no node with the specified name exists."""
        t = Tree()
        root = TreeNode(name="root", data=[0])
        t.insert(root)
        self.assertIsNone(t.get_node_from_name("does_not_exist"))

    def test_disp_does_not_error(self):
        """Tree.disp runs without raising, for a minimal tree."""
        t = Tree()
        root = TreeNode(name="root", data=[0])
        t.insert(root)
        # Just ensure no exception is raised
        t.disp()

    def test_search_by_name_bottom_up_and_top_down(self):
        """Tree.search returns deepest or shallowest node index when names repeat at different depths."""
        t = Tree()
        root = TreeNode(name="A", data=[0])
        t.insert(root)

        child = TreeNode(name="A", data=[1])
        t.insert(child, parent=root)

        shallow_ix = t.search("A", bottom_up=False)
        deep_ix = t.search("A", bottom_up=True)

        self.assertEqual(shallow_ix, 0)
        self.assertEqual(deep_ix, 1)

    def test_search_data(self):
        """Tree.search_data finds the node containing a data value; prefers deepest when bottom_up=True."""
        t = Tree()
        root = TreeNode(name=0, data=[0, 1, 2])
        t.insert(root)
        child = TreeNode(name=1, data=[2])
        t.insert(child, parent=root)

        self.assertEqual(t.search_data(2, bottom_up=True), 1)
        self.assertEqual(t.search_data(2, bottom_up=False), 0)
        self.assertEqual(t.search_data(999), -1)

    def test_all_data_and_leaves(self):
        """Tree.all_data returns sorted unique union of node.data; leaf helpers return correct nodes."""
        t = Tree()
        root = TreeNode(name=0, data=[0, 1, 2])
        t.insert(root)

        c1 = TreeNode(name=1, data=[3, 4])
        c2 = TreeNode(name=2, data=[5])
        t.insert(c1, parent=root)
        t.insert(c2, parent=root)

        self.assertEqual(t.all_data(), [0, 1, 2, 3, 4, 5])
        leaves = t.get_leaves()
        self.assertEqual(set([n.name for n in leaves]), {1, 2})

    def test_co_branch_indicator(self):
        """Tree.co_branch_indicator returns symmetric matrix with 1 for pairs co-occurring in same leaf."""
        t = Tree()
        root = TreeNode(name=0, data=[0, 1, 2, 3])
        t.insert(root)

        c1 = TreeNode(name=1, data=[0, 1])
        c2 = TreeNode(name=2, data=[2, 3])
        t.insert(c1, parent=root)
        t.insert(c2, parent=root)

        M = t.co_branch_indicator()
        self.assertIsInstance(M, pd.DataFrame)
        self.assertTrue((M.values.T == M.values).all())
        self.assertEqual(M.loc[0, 1], 1)
        self.assertEqual(M.loc[0, 2], 0)
        self.assertEqual(M.loc[2, 3], 1)
        self.assertEqual(M.loc[0, 0], 0)

