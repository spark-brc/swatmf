import os
import pandas as pd
import numpy as np


class Hg(object):


    def __init__(self, sim_start, cal_start, cal_end):

        self.hg_sub_interaction =None
        
        self.sim_start = sim_start
        self.cal_start = cal_start
        self.cal_end = cal_end

        # get sub nums


    def read_hg_sub_inter(self, filename=None):
        if filename is None:
            filename = 'swatmf_out_SWAT_rivhg'

        # Open "swatmf_out_MF_gwsw" file
        y = ("GW/SW", "for", "Positive:", "Negative:", "Day:", "Daily", "Subbasin,")  # Remove unnecssary lines

        with open(filename, "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
        data1 = [float(x.split()[1]) for x in data]  # make each line a list
        x = np.reshape(data1, (2557, 15))
        df = pd.DataFrame(x, columns=['sub{:03d}'.format(i) for i in range(1, 16)])
        df.index = pd.date_range(self.sim_start, periods=len(df))
        hg_gw_ct7 = df[self.cal_start:self.cal_end]*1000
        hg_gw_ct0617 = hg_gw_ct7.sum(axis=1).resample('A').sum()

