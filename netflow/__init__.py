"""
The :mod:`netflow` module includes ...


To do:
======
- Currently, __version__ must be manually updated in _version.py and setup.py.
  This should be automated to ensure agreement.
- Add option to `netflow.keepers` to load data from file for other file formats -- should depend on common use cases)
- Clean branching implementation
- Additional capabilities
"""

# __version__ = "0.0.dev"
from ._version import __version__


# from netflow.keepers import keeper
from netflow.keepers.keeper import Keeper
from netflow.methods.classes import InfoNet
from netflow.pose import organization, similarity
from netflow.probe.jupyter_app import render_pose
from netflow.probe import visualization
# from netflow import pose



