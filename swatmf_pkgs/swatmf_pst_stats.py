"""SWAT-MODFLOW PEST support statistics: 12/31/2019 created by Seonggyu Park
   last modified day: 4:31 12/31/2019 by Seonggyu Park
"""

import pandas as pd
import numpy as np
import scipy.stats
import math
import os


def create_param_unc(pst_file, unc_file=None, sampl_n=None):
    """create a parameter uncertainty file from an existing *.pst file

    Args:
        - pst_file (`str`): path and name of existing *.pst file
        - unc_file (`str`): name of parameter uncertainty file
                            If `None`, then `param.unc` is used.
                            Defult is `None`.
        - sampl_n ('int'): sample number from normal distribution
                            If `None`, then `1000` is used.
                            Defult is `None`.
    Returns:
        `pandas.DataFrame`: a dataframe of log standard deviation for each parameter
        `param.unc file`

    Example:
        sm_pst_stats.create_param_unc('my.pst', 'my.unc', 2000)

    """

    if not os.path.exists(pst_file):
        raise Exception("'{}' file not found".format(pst_file))
    if unc_file is None:
        unc_file = 'param.unc'
    if sampl_n is None:
        sampl_n = 1000

    with open(pst_file) as f:
        content = f.readlines()
    idxs = [x for x in range(len(content)) if (
                                            '* parameter data' in content[x].lower() or
                                            '* observation groups' in content[x].lower())]
    data = [x.split() for x in content[idxs[0]+1:idxs[1]]]

    df = pd.DataFrame(data)
    df = df.iloc[:, [0, 4, 5]]
    df = df.rename(columns={0: 'parnme', 4: 'parlbnd', 5: 'parubnd'})
    df = df.set_index('parnme')

    # Calculate the mean of the range for each parameter
    df['mu'] = (df['parlbnd'].astype(float) + df['parubnd'].astype(float)) * 0.5

    # Rough estimate of standard deviation for each parameter
    df['sigma'] = (df['parubnd'].astype(float) - df['parlbnd'].astype(float)) / 4

    lower_95s = []
    upper_95s = []
    log_sds = []
    for i in df.index:
        sampl = np.random.normal(df.loc[i, 'mu'], df.loc[i, 'sigma'], sampl_n)
        h = df.loc[i, 'sigma'] * scipy.stats.t.ppf((1 + 0.95) / 2., len(sampl) - 1)
        lower_95 = df.loc[i, 'mu'] - h
        upper_95 = df.loc[i, 'mu'] + h
        log_sd = 0.25 * (math.log10(upper_95) - math.log10(lower_95))
        lower_95s.append(lower_95)
        upper_95s.append(upper_95)
        log_sds.append(log_sd)
    df['lower_95'] = lower_95s
    df['upper_95'] = upper_95s
    df['log_sd'] = log_sds

    df.index = df.index.map(lambda x: '{0:20s}'.format(x))
    with open(unc_file, "w", newline='') as f:
        f.write("START STANDARD_DEVIATION" + "\n")
        f.write("std_multiplier 1.0" + "\n")
        df['log_sd'].to_csv(f, sep='\t', encoding='utf-8', index=True, header=False, float_format='%.10f')
        f.write("END STANDARD_DEVIATION" + "\n")
    print('{} file has been created...'.format(unc_file))

    return df
