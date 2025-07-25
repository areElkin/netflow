.. _building-install:

************
Installation
************

Currently, only installed via conda. 

TODO: UPDATE WITH IMPROVED INSTALL INSTRUCTIONS AND NOT JUST FOR CONDA

TODO: upload to PYPI for easy installation


Create conda environment
========================

Note: conda environment creation was tested on MacOS.
TODO: provide instructions for other operating systems

via env.yml
-----------

First, optionally set the environment name. The default environment name is `nf_env`.
To rename it:

Option 1: Manually rename the environment:

   - Open env.yml and replace `nf_env` with the new environment name `<env>`
     on the first and last line 


Option 2: Edit the file from the terminal:

   - If on macOS (or using BSD):

     .. code-block:: console

	$ sed -i '' -e '1s/nf_env/<env>/' -e '$s/nf_env/<env>/' env.yml

   - If on Linux (or GNU):

     .. code-block:: console

	$ sed -i -e '1s/nf_env/<env>/' -e '$s/nf_env/<env>/' env.yml

where `<env>` should be replaced with the desired environment name.


Second, set the prefix to the location where environments are stored on your system.
The prefix is specified on the last line of env.yml. To set the prefix:

Option 1: Manually set the prefix:

   - Open env.yml and replace the prefix on the last line.

Option 2: Edit the file from the terminal:

   On MacOS, environments are typically stored to `/Users/<user-name>/opt/anaconda3/envs/<env>`.
   If this is the case, replace `<user-name>` accordingly. (Note, `<env>` will have been updated
   in step 1, if you renamed the environemnt. Otherwise, it is `nf_env`.)

   For example, considering the default environment `nf_env`, replacing `<user-name>` to
   `my_user_name` can be done as follows: 

   - If on macOS (or using BSD):

     .. code-block:: console

	$ sed -i '' -e '$s/<user-name>/my_user_name/' env.yml

   - If on Linux (or GNU):

     .. code-block:: console

	$ sed -i -e '$s/<user-name>/my_user_name/' env.yml

Lastly, create the conda environment as follows:

.. code-block:: console

   $ conda env create -f env.yml


Activate the conda environment
==============================

After creating the environment, it can be activated by:

.. code-block:: console

   $ conda activate <env>
 

Add conda environment to Jupyter notebook
=========================================

After activating the environment, type the following:

.. code-block:: console

   $ python -m ipykernel install --user --name=<env>

After using the above command, the conda environment <env> should appear in Jupyter notebooks.




Future installation options
===========================

Note: pip install has not yet been tested.

.. toctree::
   :maxdepth: 6
   :numbered:
   :caption: Contents:

   usage <usage>	    


   
