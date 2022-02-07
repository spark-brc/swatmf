swatmf
===
swatmf is a set of python modules for SWAT-MODFLOW model (Bailey et al., 2016) parameter estimation and uncertainty analysis with the open-source suite PEST (Doherty 2010a and 2010b, and Doherty and other, 2010).
```shell
pip install swatmf
```

```python
from swatmf import swatmf_pst_utils, swatmf_pst_par, swatmf_viz

wd = "User-SWAT-MODFLOW working directory"
swat_wd = "User-SWAT working directory"
swatmf_pst_utils.init_setup(wd, swat_wd)
```
>>> 
```bash
Creating 'backup' folder ... passed
Creating 'echo' folder ... passed
Creating 'sufi2.in' folder ... passed
'beopest64.exe' file copied ... passed
'i64pest.exe' file copied ... passed
'i64pwtadj1.exe' file copied ... passed
'forward_run.py' file copied ... passed
```