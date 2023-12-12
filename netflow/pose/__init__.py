"""
**POSE** : **P**\ seudo-\ **O**\ rganization **\ S**\ ch\ **E**\ ma

A library for constructing the pseudo-organization schema of the observations.

This draws on topological data analaysis and lineage tracing to construct
the branching connectivity between data points (i.e., observations).

While topological data analysis provides topological information about the 
underlying space of a data set, such as the distance function, POSE seeks to
extract the topological structure of the pseudo-organization of the data points
from a distance function. The goal is to construct the pseudo-organization schema (POSE) 
which results in a network where each data point (or observation) is represented 
as a node and edges between nodes indicate the pseudo-organization and branching.

"""

