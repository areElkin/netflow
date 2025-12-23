.. _building-install:

#####################################
Installation 
#####################################
Requires Python 3.10.x

****************************
Via venv/pip (recommended)
****************************

Create Python virtual environment
===========================================

Linux/MacOS
------------

.. code-block:: console

 $ python3.10 -m venv </path/to/env_name>

Specify desired path and name env_name


Windows (CMD)
-------------

* Install Python_. 3.10.x
.. _Python: https://www.python.org/downloads/

* Navigate to directory <your_path> where the virtual environment is to be installed.

.. code-block:: console
   > cd "<your_path>" 

* Run:

.. code-block:: console

 > python3.10 -m venv <env_name>

Activate environment and install dependencies
=============================================

Linux / MacOS:
--------------

.. code-block:: console

 $ source </path/to/env_name>/bin/activate
 $ python -m pip install git+https://github.com/areElkin/netflow.git


Windows
----------

.. code-block:: console
 > .\<env>\Scripts\activate.bat
 (env) > python -m pip install git+https://github.com/areElkin/netflow.git


*********************************
Via conda (Liinux/MacOS/Windows)
*********************************
Requirements: Miniconda_. 

.. _Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/main 


Create Conda environment
==========================
.. code-block:: console

  $ conda create -n <env_name> python=3.10

Specify desired environment name <env>.


Activate Conda environment
==============================

After creating the environment, it can be activated by:

.. code-block:: console

   $ conda activate <env_name>


Install Netflow via PIP
==============================
.. code-block:: console

   $ python -m pip install git+https://github.com/areElkin/netflow.git


Adding Python virtual environment to Jupyter notebook
=====================================================

After activating the environment, type the following:

.. code-block:: console

   $ python -m ipykernel install --user --name=<env_name>

After using the above command, the venv/Conda environment <env_name> should appear in Jupyter notebooks.


.. toctree::
   :maxdepth: 6
   :numbered:
   :caption: Contents:

   usage <usage>	    


   
