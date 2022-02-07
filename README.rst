swatmf
------

.. image:: https://img.shields.io/pypi/v/swatmf?color=blue
   :target: https://pypi.python.org/pypi/swatmf
   :alt: PyPI Version
.. image:: https://img.shields.io/pypi/l/swatmf
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: PyPI - License
.. image:: https://zenodo.org/badge/304147230.svg
   :target: https://zenodo.org/badge/latestdoi/304147230

`swatmf` is a set of python modules for SWAT-MODFLOW model (Bailey et al., 2016) parameter estimation and uncertainty analysis with the open-source suite PEST (Doherty 2010a and 2010b, and Doherty and other, 2010).

.. rubric:: Installation

.. code-block:: python
   
   >>> pip install swatmf


.. rubric:: Brief overview of the API

.. code-block:: python

   from swatmf import swatmf_pst_utils

   >>> wd = "User-SWAT-MODFLOW working directory"
   >>> swat_wd = "User-SWAT working directory"
   >>> swatmf_pst_utils.init_setup(wd, swat_wd)

   Creating 'backup' folder ... passed
   Creating 'echo' folder ... passed
   Creating 'sufi2.in' folder ... passed
   'beopest64.exe' file copied ... passed
   'i64pest.exe' file copied ... passed
   'i64pwtadj1.exe' file copied ... passed
   'forward_run.py' file copied ... passed

