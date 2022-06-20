import os
import pandas as pd
import numpy as np
from .. import utils



class Hg(object):

    def __init__(self, wd):
        os.chdir(wd)
        self.hg_sub_interaction =None
        startDate, endDate, startDate_warmup, endDate_warmup = utils.define_sim_period()
        self.sim_start = startDate
        self.sim_end = endDate
        self.sim_start_warm = startDate_warmup
        self.sim_end_warm = endDate_warmup
        self.dates = pd.date_range(self.sim_start, self.sim_end)
        # get sub nums
        infig = 'fig.fig'
        with open(infig, "r") as f:
            self.sub_ids = [int(x.strip().split()[3]) for x in f if x.strip() and x.strip().startswith('subbasin')] # get only line with subbasin
        self.sub_num = len(self.sub_ids)

    @property
    def hg_sub_inter(self):
        filename = 'swatmf_out_SWAT_rivhg'
        # Open "swatmf_out_MF_gwsw" file
        y = ("GW/SW", "for", "Positive:", "Negative:", "Day:", "Daily", "Subbasin,")  # Remove unnecssary lines
        with open(filename, "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
        data1 = [float(x.split()[1]) for x in data]  # make each line a list
        x = np.reshape(data1, (len(self.dates),  self.sub_num))
        df = pd.DataFrame(x, columns=['sub{:03d}'.format(i) for i in self.sub_ids])
        df.index = pd.date_range(self.sim_start, periods=len(df))
        return df

    @property
    def hg_sub_contr(self):
        hg_sub_file = 'output-mercury.sub'
        hg_sub_df = pd.read_csv(hg_sub_file,
                            delim_whitespace=True,
                            skiprows=2,
                            usecols=["SUB", "Hg0SurfqDs", "Hg2SurfqDs", "MHgSurfqDs", "Hg0SedYlPt", "Hg2SedYlPt", "MHgSedYlPt"],
                            index_col=0
                        )
        hg_sub_dff = pd.DataFrame()
        for s in self.sub_ids:
            hg_str_sed_df = hg_sub_df.loc[hg_sub_df["SUB"] == int(s)]
            hg_str_sed_df.index = pd.date_range(self.sim_start_warm, periods=len(hg_str_sed_df))
            hg_str_sed_df = hg_str_sed_df.drop('SUB', axis=1)
        hg_sub_dff = pd.concat([hg_str_sed_df.iloc[:, 0:3].sum(axis=1), hg_str_sed_df.iloc[:, 3:].sum(axis=1)], axis=1)
        hg_sub_dff.rename(columns={0:"surf_mg", 1:"sed_mg"}, inplace=True)
        return hg_sub_dff

    def hg_yield(self, rch_id):
        hg_rch_file = 'output-mercury.rch'
        hg_rch_df = pd.read_csv(hg_rch_file,
                            delim_whitespace=True,
                            skiprows=2,
                            usecols=["RCH", "Hg0DmgOut", "Hg2DmgOut", "MeHgDmgOut", "Hg0PmgOut", "Hg2PmgOut", "MeHgPmgOut"],
                            index_col=0
                        )
        hg_rch_df = hg_rch_df.loc[hg_rch_df["RCH"] == int(rch_id)]
        hg_rch_df.index = pd.date_range(self.sim_start_warm, periods=len(hg_rch_df))
        hg_rch_df = hg_rch_df.drop('RCH', axis=1)
        return hg_rch_df
