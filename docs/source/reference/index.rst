.. _netflow-api:

.. netflow documentation master file, created by
   sphinx-quickstart on Tue Nov  7 13:12:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. template largely taken from scipy

***********
netflow API
***********

The main **netflow** namespace is comprised of subpackages. Each subpackage pertains
to a particular aspect of analysis. These are summarized in the following table:

API definition
--------------

* :py:mod:`netflow`

* :py:mod:`netflow.metrics`

* :py:mod:`netflow.classes`

* :py:mod:`netflow.pseudotime`

* :py:mod:`netflow.utils`

* :py:mod:`netflow.checks`

* :py:mod:`netflow.app`

* :py:mod:`netflow.keepers`

  - :py:mod:`netflow.keepers.keeper`

* :py:mod:`netflow.pose`

  - :py:mod:`netflow.pose.organization`

  - :py:mod:`netflow.pose.similarity`

* :py:mod:`netflow.methods`

  - :py:mod:`netflow.methods.classes`

  - :py:mod:`netflow.methods.metrics`

* :py:mod:`netflow.probe`

  - :py:mod:`netflow.probe.visualization`

..
   add :hidden: below to hide in main section and only
   include toc in side bar

..
   add :numbered: to have the TOC numbered

..
   add :caption: Contents: to have title of TOC
.. toctree::
   :maxdepth: 1
   :hidden:   

   netflow <main_namespace>   

..
   The following are hidden from appearing in a list on the main page
   but still remain in the side panel.
   To include them in the main page, include them in the previous
   toctree and remove  :hidden:
   
..
   toctree::
   .:maxdepth: 1
   .:hidden:
   .:titlesonly:
      
   netflow.metrics <generated/netflow.metrics>
   netflow.classes <generated/netflow.classes>
   netflow.pseudotime <generated/netflow.pseudotime>
   netflow.utils <generated/netflow.utils>
   netflow.checks <generated/netflow.checks>
   netflow.app <generated/netflow.app>
   netflow.keepers <generated/netflow.keepers>
   netflow.pose <generated/netflow.pose>   
   netflow.methods <generated/netflow.methods>
   netflow.probe <generated/netflow.probe>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
