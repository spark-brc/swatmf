import os
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from .. import utils
from .. import swatmf_pst_utils



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
    def hg_sub_inter2(self):
        df = pd.DataFrame(index=self.dates, columns=[c for c in range(1, self.sub_num+1)])
        filename = 'swatmf_out_SWAT_rivhg'
        # Open "swatmf_out_MF_gwsw" file
        y = ("GW/SW", "for", "Positive:", "Negative:", "Daily", "Subbasin,")  # Remove unnecssary lines
        with open(filename, "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
        date = [x.strip().split() for x in data if x.strip().startswith("Day:")] # Collect only lines with dates
        onlyDate = [int(x[1]) for x in date]
        data1 = [x.split() for x in data]  # make each line a list
        # sdate = datetime.datetime.strptime('01/01/2010', "%m/%d/%Y") # Change startDate format
        for d in tqdm(onlyDate):
            count=0
            for num, line in enumerate(data1, 1):
                if line[0] == "Day:" in line and line[1] == str(d) in line:
                    ii = num # Starting line
            count = int(ii)
            cols = []
            vals = []
            if d < len(onlyDate):
                while data1[count][0]!='Day:':
                    cols.append(data1[count][0])
                    vals.append(float(data1[count][1]))
                    count +=1
                for k in range(len(cols)):
                    df.loc[df.index[(d)-1], int(cols[k])] = vals[k]
            elif d == len(onlyDate):
                for i in range(len(data1[count:])):
                    cols.append(data1[count][0])
                    vals.append(float(data1[count][1]))
                    count +=1
                for k in range(len(cols)):
                    df.loc[df.index[int(d)-1], int(cols[k])] = vals[k]
        print("   finished ...")
        return df

    @property
    def hg_sub_gw_sw_inter2(self):
        df = pd.DataFrame(index=self.dates, columns=[c for c in range(1, self.sub_num+1)])
        filename = 'swatmf_out_SWAT_gwsw'
        # Open "swatmf_out_MF_gwsw" file
        y = ("Groundwater/Surface", "for", "Positive:", "Negative:", "Daily", "Subbasin,")  # Remove unnecssary lines
        with open(filename, "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
        date = [x.strip().split() for x in data if x.strip().startswith("Day:")] # Collect only lines with dates
        onlyDate = [int(x[1]) for x in date]
        data1 = [x.split() for x in data]  # make each line a list
        # sdate = datetime.datetime.strptime('01/01/2010', "%m/%d/%Y") # Change startDate format
        for d in tqdm(onlyDate):
            count=0
            for num, line in enumerate(data1, 1):
                if line[0] == "Day:" in line and line[1] == str(d) in line:
                    ii = num # Starting line
            count = int(ii)
            cols = []
            vals = []
            if d < len(onlyDate):
                while data1[count][0]!='Day:':
                    cols.append(data1[count][0])
                    vals.append(float(data1[count][1]))
                    count +=1
                for k in range(len(cols)):
                    df.loc[df.index[(d)-1], int(cols[k])] = vals[k]
            elif d == len(onlyDate):
                for i in range(len(data1[count:])):
                    cols.append(data1[count][0])
                    vals.append(float(data1[count][1]))
                    count +=1
                for k in range(len(cols)):
                    df.loc[df.index[int(d)-1], int(cols[k])] = vals[k]
        print("   finished ...")
        return df


    @property
    def hg_riv_grid_inter(self):
        filename = 'swatmf_out_RT_rivhg'
        # Open "swatmf_out_MF_gwsw" file
        y = ("Groundwater/Surface", "for", "Positive:", "Negative:", "Day:", "Daily")  # Remove unnecssary lines
        with open(filename, "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines
        data1 = [float(x.split()[3]) for x in data]  # make each line a list
        x = np.reshape(data1, (len(self.dates),  268))
        df = pd.DataFrame(x, columns=['grid_ids{:04d}'.format(i) for i in range(1, 269)])
        df.index = pd.date_range(self.sim_start, periods=len(df))
        return df


    @property
    def hg_sub_contr(self):
        hg_sub_file = 'output-mercury.sub'
        hg_sub_df = pd.read_csv(hg_sub_file,
                            delim_whitespace=True,
                            skiprows=2,
                            usecols=[
                                "SUB",
                                "Hg0SurfqDs", "Hg2SurfqDs", "MHgSurfqDs",
                                "Hg0LatlqDs", "Hg2LatlqDs", "MHgLatlqDs",
                                "Hg0PercqDs", "Hg2PercqDs", "MHgPercqDs",
                                "Hg0SedYlPt", "Hg2SedYlPt", "MHgSedYlPt",
                                ],
                            index_col=0
                        )
        hg_sub_dff = pd.DataFrame()
        for s in self.sub_ids:
            hg_str_sed_df = hg_sub_df.loc[hg_sub_df["SUB"] == int(s)]
            hg_str_sed_df.index = pd.date_range(self.sim_start_warm, periods=len(hg_str_sed_df))
            hg_str_sed_df = hg_str_sed_df.drop('SUB', axis=1)
            hg_sub_dff = hg_sub_dff.add(hg_str_sed_df, fill_value=0)
        contr_df = pd.concat(
            [
                hg_sub_dff.iloc[:, 0:3].sum(axis=1), 
                hg_sub_dff.iloc[:, 3:6].sum(axis=1),
                hg_sub_dff.iloc[:, 6:9].sum(axis=1),
                hg_sub_dff.iloc[:, 9:12].sum(axis=1),
                ], axis=1
                )
        contr_df.rename(
            columns={0:"surf_mg", 1:"lat_mg", 2:"perco_mg", 3:"sed_mg"}, inplace=True)
        contr_df['tot_contr'] = contr_df["surf_mg"] + contr_df["lat_mg"] + contr_df["perco_mg"] + contr_df["sed_mg"]
        return contr_df

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

    def hg_rch(self, rch_ids):
        hg_rch_file = 'output-mercury.rch'
        hg_rch_df = pd.read_csv(hg_rch_file,
                            delim_whitespace=True,
                            skiprows=2,
                            usecols=["RCH", "Hg2PmgSto"],
                            index_col=0
                        )
        hg_df = hg_rch_df.loc["REACH"]
        hg_wt_sims = pd.DataFrame()
        for i in rch_ids:
            hg_dff = hg_df.loc[hg_df["RCH"] == int(i)]
            hg_dff.index = pd.date_range(self.sim_start_warm, periods=len(hg_dff))
            hg_dfs = hg_dff.rename({'Hg2PmgSto': 'sub{:03d}'.format(i)}, axis=1)
            hg_wt_sims = pd.concat([hg_wt_sims, hg_dfs.loc[:, 'sub{:03d}'.format(i)]], axis=1)
        hg_wt_sims.index = pd.to_datetime(hg_wt_sims.index)
        return hg_wt_sims

    def hg_sed(self, sed_ids):
        hg_rch_file = 'output-mercury.rch'
        # sim_start =  sim_start[:-4] + str(int(sim_start[-4:])+ int(warmup))
        hg_sed_df = pd.read_csv(hg_rch_file,
                            delim_whitespace=True,
                            skiprows=2,
                            usecols=["RCH", "SedTHgCppm"],
                            index_col=0
                        )
        hg_df = hg_sed_df.loc["REACH"]
        hg_sed_sims = pd.DataFrame()
        for i in sed_ids:
            hg_dff = hg_df.loc[hg_df["RCH"] == int(i)]
            hg_dff.index = pd.date_range(self.sim_start_warm, periods=len(hg_dff))
            hg_dff = hg_dff.rename({'SedTHgCppm': 'sub{:03d}'.format(i)}, axis=1)
            hg_sed_sims = pd.concat([hg_sed_sims, hg_dff.loc[:, 'sub{:03d}'.format(i)]], axis=1)
        hg_sed_sims.index = pd.to_datetime(hg_sed_sims.index)
        return hg_sed_sims

    def gw_levels(self, grids, obd_cols, elevs=None):
        swatmf_pst_utils.extract_depth_to_water(grids, self.sim_start_warm, self.sim_end_warm)
        gw_tot = pd.DataFrame()
        if elevs is None:
            for g, o in zip(grids, obd_cols):
                sim_df = pd.read_csv(
                                    'dtw_{}.txt'.format(g),
                                    delim_whitespace=True,
                                    index_col=0,
                                    parse_dates=True,
                                    header=None
                                    )
                sim_df.rename(columns = {1:'sim'}, inplace = True)
                obd_df = pd.read_csv(
                                    'dtw_day.obd',
                                    sep='\t',
                                    usecols=['date', o],
                                    index_col=0,
                                    parse_dates=True,
                                    na_values=[-999, '']
                                    )
                obd_df.rename(columns = {o:'obd'}, inplace = True)
                so_df = pd.concat([sim_df, obd_df], axis=1)
                so_df['grid'] = 'g{}'.format(g)
                so_df.dropna(inplace=True)
                gw_tot = pd.concat([gw_tot, so_df], axis=0)
        else:
            for g, o, elev in zip(grids, obd_cols, elevs):
                sim_df = pd.read_csv(
                                    'dtw_{}.txt'.format(g),
                                    delim_whitespace=True,
                                    index_col=0,
                                    parse_dates=True,
                                    header=None
                                    )
                sim_df.rename(columns = {1:'sim'}, inplace = True)
                obd_df = pd.read_csv(
                                    'dtw_day.obd',
                                    sep='\t',
                                    usecols=['date', o],
                                    index_col=0,
                                    parse_dates=True,
                                    na_values=[-999, '']
                                    )
                obd_df.rename(columns = {o:'obd'}, inplace = True)
                so_df = pd.concat([sim_df, obd_df], axis=1)
                so_df['grid'] = 'g{}'.format(g)
                so_df['elev'] = elev
                so_df.dropna(inplace=True)
                gw_tot = pd.concat([gw_tot, so_df], axis=0)
            gw_tot['sim_wt'] = gw_tot['sim'] + gw_tot['elev']
            gw_tot['obd_wt'] = gw_tot['obd'] + gw_tot['elev']
        return gw_tot