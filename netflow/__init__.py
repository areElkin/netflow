"""
The :mod:`netflow` module includes ...


To do:
======
- Currently, __version__ must be manually updated in _version.py and setup.py.
  This should be automated to ensure agreement.
- Add option to `netflow.keepers` to load data from file (need to specify format -- should depend on common use cases)
- Possibly add class to `netflow.keepers` to handle Graphs?
- Possibly add class to `netlfow.keepers` to handle single or multiple feature data set(s), distance(s), similarit(y/ies), graph(s)
  with the same observations and features.
- In `netflow.keepers`, add way to select data directly from :class:`netflow.keepers.Keeper`.
- May be issue with ``netflow.keepers.Keeper`` in how ``observation_labels`` are determined. Should require to be set at
  initialization or else, always just use index (e.g., ignore if later add a DataFrame with labels).
  Note: this means you cannot initialize a blank `Keeper`.
- In `netflow.keepers.Keeper`, add option for adding feature data, distance, similarity, etc that doesn't use
  `netflow.keepers.DataKeeper`, etc. internal method, to ensure same observation labels, etc.
- In `netflow.keepers.Keeper`, add option for directory and how to save data
- In `netflow.keeeprs.DataKeeper`, add option for normalization, and subset and subset normalization
- In `netflow.keepers.DistanceKeeper`, add option for to similarity?
- In `netflow.pose.organization`, separate out multiscale analysis to it's own form and possibly should be under `netflow.methods` instead
- `netflow.keepers.keeper.DataView` can be better leveraged in `netflow.methods.classes` and for computing neighborhoods and wass distance, etc.
"""

# __version__ = "0.0.dev"
from ._version import __version__


# from netflow.keepers import keeper
from netflow.keepers.keeper import Keeper
from netflow.methods.classes import InfoNet
from netflow.pose import organization, similarity
# from netflow import pose

