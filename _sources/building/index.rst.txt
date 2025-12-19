.. _building-install:

#####################################
Installation (Linux/MacOS/Windows)
#####################################

****************************
Via conda (recommended)
****************************
Requirements: Miniconda_. 

.. _Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/main 

Conda provides pre-built binaries for Graphviz, which Netflow depends on, eliminating the need to compile from source.  


TODO: UPDATE WITH IMPROVED INSTALL INSTRUCTIONS AND NOT JUST FOR CONDA


Create Conda environment
==========================
.. code-block:: console

  $ conda create -n <env> python=3.10

Specify desired environment name <env>.


Activate Conda environment
==============================

After creating the environment, it can be activated by:

.. code-block:: console

   $ conda activate <env>


Install Graphviz
==============================
.. code-block:: console

   $ conda install -c conda-forge graphviz pygraphviz


Install Netflow via PIP
==============================
.. code-block:: console

   $ python -m pip install git+https://github.com/areElkin/netflow.git


Add Conda environment to Jupyter notebook
=========================================

After activating the environment, type the following:

.. code-block:: console

   $ python -m ipykernel install --user --name=<env>

After using the above command, the conda environment <env> should appear in Jupyter notebooks.


Future installation options
===========================


.. toctree::
   :maxdepth: 6
   :numbered:
   :caption: Contents:

   usage <usage>	    


   
