""" PEST support visualizations: 02/09/2021 created by Seonggyu Park
    last modified day: 02/21/2021 by Seonggyu Park
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# from hydroeval import evaluator, nse, rmse, pbias
import numpy as np
import math
import datetime as dt
from tqdm import tqdm
from swatmf import analyzer


# NOTE: swat output handler
class SWATMFout(object):

    def __init__(self, wd):
        os.chdir(wd)
        if os.path.isfile("file.cio"):
            cio = open("file.cio", "r")
            lines = cio.readlines()
            skipyear = int(lines[59][12:16])
            self.iprint = int(lines[58][12:16]) #read iprint (month, day, year)
            styear = int(lines[8][12:16]) #begining year
            styear_warmup = int(lines[8][12:16]) + skipyear #begining year with warmup
            edyear = styear + int(lines[7][12:16])-1 # ending year
            edyear_warmup = styear_warmup + int(lines[7][12:16])-1 - int(lines[59][12:16])#ending year with warmup
            if skipyear == 0:
                FCbeginday = int(lines[9][12:16])  #begining julian day
            else:
                FCbeginday = 1  #begining julian day
            FCendday = int(lines[10][12:16])  #ending julian day
            cio.close()
            self.stdate = dt.datetime(styear, 1, 1) + dt.timedelta(FCbeginday - 1)
            self.eddate = dt.datetime(edyear, 1, 1) + dt.timedelta(FCendday - 1)
            self.stdate_warmup = dt.datetime(styear_warmup, 1, 1) + dt.timedelta(FCbeginday - 1)
            self.eddate_warmup = dt.datetime(edyear_warmup, 1, 1) + dt.timedelta(FCendday - 1)



    # scratches for QSWATMOD
    # read data first
    # stream discharge output.rch
    def read_stf_obd(self, obd_file):
        return pd.read_csv(obd_file,
            index_col=0,
            header=0,
            parse_dates=True,
            na_values=[-999, ""]
        )

    def read_output_rch_data(self, colNum=6):
        return pd.read_csv(
            "output.rch",
            sep=r'\s+',
            skiprows=9,
            usecols=[1, 3, colNum],
            names=["date", "filter", "stf_sim"],
            index_col=0
        )

    def update_index(self, df):
        startDate = self.stdate_warmup
        if self.iprint == 1:
            df.index = pd.date_range(startDate, periods=len(df.stf_sim))
        elif self.iprint == 0:
            df = df[df['filter'] < 13]
            df.index = pd.date_range(startDate, periods=len(df.stf_sim), freq="M")
        else:
            df.index = pd.date_range(startDate, periods=len(df.stf_sim), freq="A")
        return df

    def get_stf_sim(colNum=6):
        return pd.read_csv(
            "output.rch",
            sep=r'\s+',
            skiprows=9,
            usecols=[1, 3, colNum],
            names=["date", "filter", "stf_sim"],
            index_col=0
        )        

    def get_stf_sim_obd(self, obd_file, obd_col, subnum):
        strObd = self.read_stf_obd(obd_file)
        output_rch = self.read_output_rch_data()
        df = output_rch.loc[subnum]
        df = self.update_index(df)
        df2 = pd.concat([df, strObd[obd_col]], axis=1)
        df3 = df2.dropna()
        return df3

    # read groundwater levels from
    # 431, 4011
    # let's give dataframe not series
    def get_gw_sim(self, dtw_format=True):
        mf_obs = pd.read_csv(
                            "modflow.obs",
                            sep=r'\s+',
                            skiprows = 2,
                            usecols = [2, 3, 4],
                            # index_col = 0,
                            names = ["layer", "grid_id", "mf_elev"],)
        mf_obs["grid_layer"] = "dtw_sim" + mf_obs['grid_id'].astype(str) + "lyr" + mf_obs["layer"].astype(str)

        # need to change grid id info to allow multi-layer outputs
        grid_lyr_lst = mf_obs.loc[:, "grid_layer"].tolist()
        output_wt = pd.read_csv(
                            "swatmf_out_MF_obs",
                            sep=r'\s+',
                            skiprows = 1,
                            names = grid_lyr_lst)
        
        # '''
        if dtw_format is True:
            dtw_df = pd.DataFrame()
            for grid_id in grid_lyr_lst:
                dtw_list = output_wt.loc[:, str(grid_id)] - float(mf_obs["mf_elev"].loc[mf_obs["grid_layer"]==grid_id])
                dtw_df = pd.concat(
                    [dtw_df, pd.DataFrame({str(grid_id):dtw_list})], 
                    axis=1, ignore_index=True)
            dtw_df.columns = grid_lyr_lst
            dtw_df.index = pd.date_range(self.stdate, periods=len(dtw_df))
            return dtw_df
        else:
            output_wt.index = pd.date_range(self.stdate, periods=len(output_wt))
            return output_wt
        # '''
    
    def get_gw_obd(self, ts=None):
        if ts is None:
            mfobd_file = "dtw_day.obd.csv"
        if ts == "month":
            mfobd_file = "dtw_mon.obd.csv"
        return pd.read_csv(
                        mfobd_file,
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""])
    
    def get_gw_sim_obd(self, grid_id, obd_col, ts=None, dtw_format=True):
        gw_obd = self.get_gw_obd(obd_col, ts=ts)
        gw_sim = self.get_gw_sim(grid_id, dtw_format=dtw_format)
        df =  pd.concat([gw_sim, gw_obd], axis=1).dropna()
        return df


    # read waterbalance data from output.std
    def get_std_data(self):
        startDate = self.stdate_warmup
        eddate = self.eddate_warmup
        eYear = eddate.strftime("%Y")
        with open("output.std", "r") as infile:
            lines = []
            y = ("TIME", "UNIT", "SWAT", "(mm)")
            for line in infile:
                data = line.strip()
                if len(data) > 100 and not data.startswith(y):  # 1st filter
                    lines.append(line)           
        dates = []
        for line in lines:  # 2nd filter
            try:
                date = line.split()[0]
                if (date == eYear):  # Stop looping
                    break
                elif(len(str(date)) == 4):  # filter years
                    continue
                else:
                    dates.append(line)
            except:
                pass
        date_f, prec, surq, latq, gwq, swgw, perco, tile, sw, gw = [], [], [], [], [], [], [], [], [], []
        for i in range(len(dates)): # 3rd filter and obtain necessary data
            if (int(dates[i].split()[0]) == 1) and (int(dates[i].split()[0]) - int(dates[i - 1].split()[0]) == -30):
                continue
            elif (int(dates[i].split()[0]) < int(dates[i-1].split()[0])) and (int(dates[i].split()[0]) != 1):
                continue
            else:
                date_f.append(int(dates[i].split()[0]))
                prec.append(float(dates[i].split()[1]))
                surq.append(float(dates[i].split()[2]))
                latq.append(float(dates[i].split()[3]))
                gwq.append(float(dates[i].split()[4]))
                swgw.append(float(dates[i].split()[5]))
                # perco.append(float(dates[i].split()[6]))
                perco.append(float(dates[i].split()[7]))  # SM3 uses reach !SP
                tile.append(float(dates[i].split()[8]))  # not use it for now
                sw.append(float(dates[i].split()[10]))
                gw.append(float(dates[i].split()[11]))
        names = ["prec", "surq", "latq", "gwq", "swgw", "perco", "tile", "sw", "gw"]
        data = pd.DataFrame(
            np.column_stack([prec, surq, latq, gwq, swgw, perco, tile, sw, gw]),
            columns=names)
        data.index = pd.date_range(startDate, periods=len(data))
        return data
    
    def get_gwsw_sub_df(self):
        stdate = self.stdate 
        tot_feats = 257
        infile = "swatmf_out_SWAT_gwsw_monthly"
        y = ("Monthly", "month:")
        with open(infile, "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)]
        data1 = [x.split()[1] for x in data]
        data_array = np.reshape(
            data1, (int(len(data1)/tot_feats), tot_feats), 
            # order='F'
            )
        # column_names = [i for i in range(1, 258)]
        df_ = pd.DataFrame(
            data_array, 
            # columns=column_names
            )
        df_.index = pd.date_range(stdate, periods=len(df_), freq="ME")
        dff = df_["1/1/2010":"12/31/2019"].astype(float)
        mbig_df = dff.groupby(dff.index.month).mean().T
        mbig_df["subid"] = mbig_df.index+1
        mbig_df.to_csv("swat_gwsw_avg_mon.csv", index=False)
        return mbig_df


    def get_head_avg_m_df(self):
        stdate = self.stdate 
        tot_feats = 74095
        # Open "swatmf_out_MF_head" file
        y = ("Monthly", "Yearly") # Remove unnecssary lines
        filename = "swatmf_out_MF_head_monthly"
        with open(filename, "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines     
        date = [x.strip().split() for x in data if x.strip().startswith("month:")] # Collect only lines with dates  
        onlyDate = [x[1] for x in date] # Only date
        data1 = [x.split() for x in data] # make each line a list
        dateList = pd.date_range(stdate, periods=len(onlyDate), freq ='M').strftime("%b-%Y").tolist()
        selectedSdate = 'Jan-2010'
        selectedEdate = 'Dec-2019'

        # Reverse step
        dateSidx = dateList.index(selectedSdate)
        dateEidx = dateList.index(selectedEdate)
        dateList_f = dateList[dateSidx:dateEidx+1]

        big_df = pd.DataFrame()
        datecount = 0
        for selectedDate in tqdm(dateList_f):
            # Reverse step
            dateIdx = dateList.index(selectedDate)
            #only
            onlyDate_lookup = onlyDate[dateIdx]
            dtt = dt.datetime.strptime(selectedDate, "%b-%Y")
            year = dtt.year
            layerN = "1"
            for num, line in enumerate(data1, 1):
                if ((line[0] == "month:" in line) and (line[1] == onlyDate_lookup in line) and (line[3] == str(year) in line)):
                    ii = num # Starting line
            count = 0
            while not ((data1[count+ii][0] == 'layer:') and (data1[count+ii][1] == layerN)):
                count += 1
            stline =count+ii+1

            mf_rchs = []
            hdcount = 0
            while hdcount < tot_feats:
                for kk in range(len(data1[stline])):
                    mf_rchs.append(float(data1[stline][kk]))
                    hdcount += 1
                stline += 1
            s = pd.Series(mf_rchs, name=dt.datetime.strptime(selectedDate, "%b-%Y").strftime("%Y-%m-%d"))
            big_df = pd.concat([big_df, s], axis=1)
            datecount +=1


        big_df = big_df.T
        big_df.index = pd.to_datetime(big_df.index)
        mbig_df = big_df.groupby(big_df.index.month).mean()
        mbig_df_t = mbig_df.T
        mbig_df_t["grid_id"] = mbig_df_t.index + 1
        mbig_df_t.to_csv("mf_head_avg_mon.csv", index=False)
        print("finished ...")

    def get_recharge_avg_m_df(self):
        stdate = self.stdate 
        tot_feats = 74095

        # Open "swatmf_out_MF_head" file
        y = ("Monthly", "Yearly") # Remove unnecssary lines
        filename = "swatmf_out_MF_recharge_monthly"
        # self.layer = QgsProject.instance().mapLayersByName("mf_nitrate_monthly")[0]
        with open(os.path.join(wd, filename), "r") as f:
            data = [x.strip() for x in f if x.strip() and not x.strip().startswith(y)] # Remove blank lines     
        date = [x.strip().split() for x in data if x.strip().startswith("month:")] # Collect only lines with dates  
        onlyDate = [x[1] for x in date] # Only date
        data1 = [x.split() for x in data] # make each line a list
        dateList = pd.date_range(stdate, periods=len(onlyDate), freq ='ME').strftime("%b-%Y").tolist()

        selectedSdate = 'Jan-2010'
        selectedEdate = 'Dec-2019'
        # Reverse step
        dateSidx = dateList.index(selectedSdate)
        dateEidx = dateList.index(selectedEdate)
        dateList_f = dateList[dateSidx:dateEidx+1]

        big_df = pd.DataFrame()
        datecount = 0
        for selectedDate in tqdm(dateList_f):
            # Reverse step
            dateIdx = dateList.index(selectedDate)
            #only
            onlyDate_lookup = onlyDate[dateIdx]
            dtt = dt.datetime.strptime(selectedDate, "%b-%Y")
            year = dtt.year
            layerN = "1"
            for num, line in enumerate(data1, 1):
                if ((line[0] == "month:" in line) and (line[1] == onlyDate_lookup in line) and (line[3] == str(year) in line)):
                    ii = num # Starting line
            mf_rchs = []
            hdcount = 0
            while hdcount < tot_feats:
                for kk in range(len(data1[ii])):
                    mf_rchs.append(float(data1[ii][kk]))
                    hdcount += 1
                ii += 1
            s = pd.Series(mf_rchs, name=dt.datetime.strptime(selectedDate, "%b-%Y").strftime("%Y-%m-%d"))
            big_df = pd.concat([big_df, s], axis=1)
            datecount +=1
 
        big_df = big_df.T
        big_df.index = pd.to_datetime(big_df.index)
        mbig_df = big_df.groupby(big_df.index.month).mean()
        mbig_df_t = mbig_df.T
        mbig_df_t["grid_id"] = mbig_df_t.index + 1
        mbig_df_t.to_csv("mf_rch_avg_mon.csv", index=False)
        print("finished ...")



def okvg_temp():
    wd = "C:\\Users\\seonggyu.park\\Downloads\\qswatmod_prj\\2nd_cali\\okvg_062320_pest\\SWAT-MODFLOW"
    m1 = SWATMFout(wd)
    df =  m1.get_recharge_avg_m_df()
    print(df)


def read_morris_msn(wd, pst_name):
    df = pd.read_csv(
            os.path.join(wd, pst_name.replace(".pst",".msn")),
            index_col='parameter_name'
            )
    # df.loc[df['sen_std_dev'].str.contains('-nan'), 'sen_std_dev'] = 0
    # df = df.astype(float)
    print(df) 
    return df 




# def plot_tot():
if __name__ == '__main__':
    wd = "D:\\Projects\\Watersheds\\Koksilah\\analysis\\koksilah_git\\koki_zon_rw_morris"
    pst_name = "koki_zon_rw_morris.pst"
    # read_morris_msn(wd, pst_name)
    analyzer.plot_sen_morris(read_morris_msn(wd, pst_name))