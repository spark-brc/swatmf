"""swatmf: a set of python modules for SWAT-MODFLOW model (Bailey et al., 2016)
parameter estimation and uncertainty analysis with the open-source suite PEST (Doherty 2010a
and 2010b, and Doherty and other, 2010).
"""

from .swatmf_utils import *
from .swatmf_viz import *
from .swatmf_pst_utils import *
from .swatmf_pst_par import *
# from .swatmf_pst_stats import *


# from .ev import ErrVar
# from .en import Ensemble, ParameterEnsemble, ObservationEnsemble

# # from .mc import MonteCarlo
# # from .inf import Influence
# from .mat import Matrix, Jco, Cov
# from .pst import Pst, pst_utils
# from .utils import (
#     helpers,
#     gw_utils,
#     optimization,
#     geostats,
#     pp_utils,
#     os_utils,
#     smp_utils,
#     metrics,
# )
# from .plot import plot_utils
# from .logger import Logger

# from .prototypes import *

# from ._version import get_versions

# __version__ = get_versions()["version"]
# __all__ = [
#     "LinearAnalysis",
#     "Schur",
#     "ErrVar",
#     "Ensemble",
#     "ParameterEnsemble",
#     "ObservationEnsemble",
#     "Matrix",
#     "Jco",
#     "Cov",
#     "Pst",
#     "pst_utils",
#     "helpers",
#     "gw_utils",
#     "geostats",
#     "pp_utils",
#     "os_utils",
#     "smp_utils",
#     "plot_utils",
#     "metrics",
# ]
# # del get_versions