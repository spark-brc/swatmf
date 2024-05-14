======
swatmf
======

.. image:: https://img.shields.io/pypi/v/swatmf?color=blue
   :target: https://pypi.python.org/pypi/swatmf
   :alt: PyPI Version
.. image:: https://img.shields.io/pypi/l/swatmf
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: PyPI - License
.. image:: https://zenodo.org/badge/304147230.svg
   :target: https://zenodo.org/badge/latestdoi/304147230



`swatmf` is a set of python modules for SWAT-MODFLOW model (Bailey et al., 2016) parameter estimation and uncertainty analysis with the open-source suite PEST (Doherty 2010a and 2010b, and Doherty and other, 2010).

===========================================
Uncertainty Analysis for SWAT-MODFLOW model
===========================================


Get data and jupyter notebooks
------------------------------

You essentially have 2 options:

Easy way
--------

- `Download the data zip file <https://github.com/spark-brc/swatmf/archive/refs/heads/main.zip>`_
- Unzip `swatmf-main.zip` to a prefered location.


Hard way (Dev mode)
-------------------

- You will need to install Git if you don't have it installed already. Downloads are available at [the link](https://git-scm.com/download). On windows, be sure to select the option that installs command-line tools  
- For Git, you will need to set up SSH keys to work with Github. To do so:
    - Go to GitHub.com and set up an account
    - On Windows, open Git Bash (on Mac/Linux, just open a terminal) and set up ssh keys if you haven't already. To do this, simply type ssh-keygen in git bash/terminal and accept all defaults (important note - when prompted for an optional passphrase, just hit return.)  
- Follow the `instructions <https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/>`_ to set up the SSH keys with your GitHub account.
- Clone the materials from GitHub.
    - Open a git bash shell from the start menu (or, on a Mac/Linux, open a terminal)
    - Navigate to the folder you made to put the course materials
    - Clone the materials by executing the following in the git bash or terminal window:


.. code-block:: bash

   git clone https://github.com/spark-brc/swatmf.git


============
Installation
============

To execute jupyter notebook, we need the Miniconda environment.

1. Miniconda Python:
--------------------

- If you don't already have conda installed, please download Miniconda for your operating system from https://conda.io/en/latest/miniconda.html (choose the latest version for your operating system, 64-bit). You should not need elevated rights to install this.
- Run the installer and select "only my user" when prompted. This will allow you to work with your python installation directly.

2. Set Environment and install libraries:
-----------------------------------------

- After installation, go to the START menu and select "Miniconda Prompt" to open a DOS box.
- Type the following command:



.. code-block:: bash

   conda install -c conda-forge mamba


- Using the `cd <https://www.computerhope.com/issues/chusedos.htm>`_ command in the Miniconda DOS box, navigate to the location where you have `environment.yml` the file and type: 

.. code-block:: bash

   mamba env create -f environment_swatmf.yml

and hit ENTER.

After your virtual environment setup is complete, change the environment to `swatmf`:  

.. code-block:: bash

   conda activate swatmf

- Launch jupyter notebook 

.. code-block:: bash

   jupyter notebook


A browser window with a Jupyter notebook instance should open. Yay!




.. rubric:: Brief overview of the API

.. code-block:: python

   from swatmf import swatmf_pst_utils

   >>> prj_dir = "project directory"
   >>> swatmfwd = "SWAT-MODFLOW model"
   >>> swatwd = "SWAT model"
   >>> swatmf_pst_utils.init_setup(prj_dir, swatmfwd, swatwd))

   Creating 'backup' folder ... passed
   Creating 'echo' folder ... passed
   Creating 'sufi2.in' folder ... passed
   'Absolute_SWAT_Values.txt' file copied ... passed
   'pestpp-glm' file copied ... passed
   'pestpp-ies.exe' file copied ... passed
   'pestpp-ies.exe' file copied ... passed
   'forward_run.py' file copied ... passed

