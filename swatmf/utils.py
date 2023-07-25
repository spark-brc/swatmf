""" SWATMF support functions: 02/09/2021 created by Seonggyu Park
    last modified day: 03/21/2021 by Seonggyu Park
"""

import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from hydroeval import evaluator, nse, rmse, pbias
# import numpy as np
import numpy as np
# import h5py as hdf
import os 
import datetime

def define_sim_period():
    if os.path.isfile("file.cio"):
        cio = open("file.cio", "r")
        lines = cio.readlines()
        skipyear = int(lines[59][12:16])
        iprint = int(lines[58][12:16]) #read iprint (month, day, year)
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
        stdate = datetime.datetime(styear, 1, 1) + datetime.timedelta(FCbeginday - 1)
        eddate = datetime.datetime(edyear, 1, 1) + datetime.timedelta(FCendday - 1)
        stdate_warmup = datetime.datetime(styear_warmup, 1, 1) + datetime.timedelta(FCbeginday - 1)
        eddate_warmup = datetime.datetime(edyear_warmup, 1, 1) + datetime.timedelta(FCendday - 1)

        startDate = stdate.strftime("%m/%d/%Y")
        endDate = eddate.strftime("%m/%d/%Y")
        startDate_warmup = stdate_warmup.strftime("%m/%d/%Y")
        endDate_warmup = eddate_warmup.strftime("%m/%d/%Y")
        # duration = (eddate - stdate).days

        # ##### 
        # start_month = stdate.strftime("%b")
        # start_day = stdate.strftime("%d")
        # start_year = stdate.strftime("%Y")
        # end_month = eddate.strftime("%b")
        # end_day = eddate.strftime("%d")
        # end_year = eddate.strftime("%Y")
        return startDate, endDate, startDate_warmup, endDate_warmup

def delete_duplicate_river_grids(wd, riv_fname):
    with open(os.path.join(wd, riv_fname), "r") as fp:
        lines = fp.readlines()
        new_lines = []
        for line in lines:
            #- Strip white spaces
            line = line.strip()
            if line not in new_lines:
                new_lines.append(line)

    output_file = "{}_fixed".format(riv_fname)
    with open(os.path.join(wd, output_file), "w") as fp:
        fp.write("\n".join(new_lines))
    print('done!')


def get_all_scenario_lists(wd):
    os.chdir(wd)
    scn_nams = [name for name in os.listdir(".") if os.path.isdir(name)]
    full_paths = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]
    return scn_nams, full_paths    


def all_strs(wd, sub_number, start_date, obd_nam, time_step=None):
    scn_nams, full_paths = get_all_scenario_lists(wd)
    if time_step is None:
        time_step = "D"
        strobd_file = "swat_rch_day.obd"
    else:
        time_step = "M"
        strobd_file = "swat_rch_mon.obd"
    tot_df = pd.DataFrame()
    for scn_nam, p in zip(scn_nams, full_paths):
        os.chdir(p)
        print("Folder changed to {}".format(p))
        df = pd.read_csv(
                    os.path.join("output.rch"),
                    delim_whitespace=True,
                    skiprows=9,
                    usecols=[1, 3, 6],
                    names=["date", "filter", "str_sim"],
                    index_col=0)
        df = df.loc[sub_number]
        if time_step == 'M':
            df = df[df["filter"] < 13]
        df.index = pd.date_range(start_date, periods=len(df.str_sim), freq=time_step)

        df.rename(columns = {'str_sim':'{}_sub_{}'.format(scn_nam, sub_number)}, inplace = True)
        tot_df = pd.concat(
            [tot_df, df['{}_sub_{}'.format(scn_nam, sub_number)]], axis=1,
            sort=False
            )
    print('Finished!')
    return tot_df


def all_seds(wd, sub_number, start_date, obd_nam, time_step=None):
    scn_nams, full_paths = get_all_scenario_lists(wd)
    if time_step is None:
        time_step = "D"
        strobd_file = "swat_rch_day.obd"
    else:
        time_step = "M"
        strobd_file = "swat_rch_mon.obd"
    tot_df = pd.DataFrame()
    for scn_nam, p in zip(scn_nams, full_paths):
        os.chdir(p)
        print("Folder changed to {}".format(p))
        df = pd.read_csv(
                    os.path.join("output.rch"),
                    delim_whitespace=True,
                    skiprows=9,
                    usecols=[1, 3, 10],
                    names=["date", "filter", "str_sim"],
                    index_col=0)
        df = df.loc[sub_number]
        if time_step == 'M':
            df = df[df["filter"] < 13]
        df.index = pd.date_range(start_date, periods=len(df.str_sim), freq=time_step)

        df.rename(columns = {'str_sim':'{}_sub_{}'.format(scn_nam, sub_number)}, inplace = True)
        tot_df = pd.concat(
            [tot_df, df['{}_sub_{}'.format(scn_nam, sub_number)]], axis=1,
            sort=False
            )
    print('Finished!')
    return tot_df


def str_df(rch_file, start_date, rch_num, obd_nam, time_step=None):
    
    if time_step is None:
        time_step = "D"
        strobd_file = "swat_rch_day.obd"
    else:
        time_step = "M"
        strobd_file = "swat_rch_mon.obd."
    output_rch = pd.read_csv(
                        rch_file, delim_whitespace=True, skiprows=9,
                        usecols=[0, 1, 8], names=["idx", "sub", "simulated"], index_col=0
                        )
    df = output_rch.loc["REACH"]
    str_obd = pd.read_csv(
                        strobd_file, sep=r'\s+', index_col=0, header=0,
                        parse_dates=True, delimiter="\t",
                        na_values=[-999, ""]
                        )
    # Get precipitation data from *.DYL
    prep_file = 'sub{}.DLY'.format(rch_num)
    with open(prep_file) as f:
        content = f.readlines()    
    year = content[0][:6].strip()
    mon = content[0][6:10].strip()
    day = content[0][10:14].strip()
    prep = [float(i[32:38].strip()) for i in content]
    prep_stdate = "/".join((mon,day,year))
    prep_df =  pd.DataFrame(prep, columns=['prep'])
    prep_df.index = pd.date_range(prep_stdate, periods=len(prep))
    prep_df = prep_df.replace(9999, np.nan)
    if time_step == "M":
        prep_df = prep_df.resample('M').mean()
    df = df.loc[df['sub'] == int(rch_num)]
    df = df.drop('sub', axis=1)
    df.index = pd.date_range(start_date, periods=len(df), freq=time_step)
    df = pd.concat([df, str_obd[obd_nam], prep_df], axis=1)
    plot_df = df[df['simulated'].notna()]
    return plot_df



def get_stats(df):
    df_stat = df.dropna()

    sim = df_stat.iloc[:, 0].to_numpy()
    obd = df_stat.iloc[:, 1].to_numpy()
    df_nse = evaluator(nse, sim, obd)
    df_rmse = evaluator(rmse, sim, obd)
    df_pibas = evaluator(pbias, sim, obd)
    r_squared = (
        ((sum((obd - obd.mean())*(sim-sim.mean())))**2)/
        ((sum((obd - obd.mean())**2)* (sum((sim-sim.mean())**2))))
        )
    return df_nse, df_rmse, df_pibas, r_squared



def obds_df(strobd_file, wt_obd_file):
    str_obd = pd.read_csv(
                        strobd_file, sep=r'\s+', index_col=0, header=0,
                        parse_dates=True, delimiter="\t",
                        na_values=[-999, ""]
                        )
    wt_obd = pd.read_csv(
                        'MODFLOW/' + wt_obd_file, sep=r'\s+', index_col=0, header=0,
                        parse_dates=True, delimiter="\t",
                        na_values=[-999, ""]
                        )
    if strobd_file == 'swat_rch_mon.obd':
        str_obd = str_obd.resample('M').mean()
    if wt_obd_file == 'modflow_mon.obd':
        wt_obd = wt_obd.resample('M').mean()

    df = pd.concat([str_obd, wt_obd], axis=1)
    return df
    

def wt_df(start_date, grid_id, obd_nam, time_step=None, prep_sub=None):
    
    if time_step is None:
        time_step = "D"
        mfobd_file = "modflow_day.obd"
    else:
        time_step = "M"
        mfobd_file = "modflow_mon.obd."

    mf_obs = pd.read_csv(
                        "MODFLOW/modflow.obs",
                        delim_whitespace=True,
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        "MODFLOW/" + mfobd_file,
                        sep='\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")

    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    output_wt = pd.read_csv(
                        "MODFLOW/apexmf_out_MF_obs",
                        delim_whitespace=True,
                        skiprows = 1,
                        names = grid_id_lst,)
    output_wt = output_wt[str(grid_id)] - float(mf_obs.loc[int(grid_id)])
    output_wt.index = pd.date_range(start_date, periods=len(output_wt))

    if time_step == 'M':
        output_wt = output_wt.resample('M').mean()
    if prep_sub is not None:
        # Get precipitation data from *.DYL
        prep_file = 'sub{}.DLY'.format(prep_sub)
        with open(prep_file) as f:
            content = f.readlines()    
        year = content[0][:6].strip()
        mon = content[0][6:10].strip()
        day = content[0][10:14].strip()
        prep = [float(i[32:38].strip()) for i in content]
        prep_stdate = "/".join((mon,day,year))
        prep_df =  pd.DataFrame(prep, columns=['prep'])
        prep_df.index = pd.date_range(prep_stdate, periods=len(prep))
        prep_df = prep_df.replace(9999, np.nan)
        # if time_step == "M":
        prep_df = prep_df.resample('M').mean()
        output_wt = pd.concat([output_wt, mfobd_df[obd_nam], prep_df], axis=1)
    else:
        output_wt = pd.concat([output_wt, mfobd_df[obd_nam]], axis=1)
    output_wt = output_wt[output_wt[str(grid_id)].notna()]

    return output_wt        


def wt_tot_df(sim_start, df_start, df_end, grid_ids, obd_nams, time_step=None):
    """combine all groundwater outputs to provide a dataframe for 1 to 1 plot

    Args:
        start_date (str): simulation start date 
        grid_ids (list): list of grid ids used for plot
        obd_nams (list): list of column names in observed data and in accordance with grid ids
        time_step (str, optional): simulation time step (day, month, annual). Defaults to None.

    Returns:
        dataframe: dataframe for all simulated depth to water and observed data
    """
    if time_step is None:
        time_step = "D"
        mfobd_file = "modflow_day.obd"
    else:
        time_step = "M"
        mfobd_file = "modflow_mon.obd."
    # read obs and obd files to get grid ids, elev, and observed values
    mf_obs = pd.read_csv(
                        "MODFLOW/modflow.obs",
                        delim_whitespace=True,
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        "MODFLOW/" + mfobd_file,
                        sep='\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")
    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    # read simulated water elevation
    output_wt = pd.read_csv(
                        "MODFLOW/apexmf_out_MF_obs",
                        delim_whitespace=True,
                        skiprows = 1,
                        names = grid_id_lst,)
    # append data to big dataframe
    tot_df = pd.DataFrame()
    for grid_id, obd_nam in zip(grid_ids, obd_nams):
        df = output_wt[str(grid_id)] - float(mf_obs.loc[int(grid_id)]) # calculate depth to water
        df.index = pd.date_range(sim_start, periods=len(df))
        df = df[df_start:df_end]
        if time_step == 'M':
            df = df.resample('M').mean()
        df = pd.concat([df, mfobd_df[obd_nam]], axis=1) # concat sim with obd
        df = df.dropna() # drop nan
        new_cols ={x:y for x, y in zip(df.columns, ['sim', 'obd'])} #replace col nams with new nams
        tot_df = tot_df.append(df.rename(columns=new_cols))  
    return tot_df


def read_output_mgt(wd):
    with open(os.path.join(wd, 'output.mgt'), 'r') as f:
        content = f.readlines()
    subs = [int(i[:5]) for i in content[5:]]
    hrus = [int(i[5:10]) for i in content[5:]]
    yrs = [int(i[10:16]) for i in content[5:]]
    mons = [int(i[16:22]) for i in content[5:]]
    doys = [int(i[22:28]) for i in content[5:]]
    areas = [float(i[28:39]) for i in content[5:]]
    cfp = [str(i[39:55]).strip() for i in content[5:]]
    opt = [str(i[55:70]).strip() for i in content[5:]]
    irr = [-999 if i[150:160].strip() == '' else float(i[150:160]) for i in content[5:]]
    mgt_df = pd.DataFrame(
        np.column_stack([subs, hrus, yrs, mons, doys, areas, cfp, opt, irr]),
        columns=['sub', 'hru', 'yr', 'mon', 'doy', 'area_km2', 'cfp', 'opt', 'irr_mm'])
    mgt_df['irr_mm'] = mgt_df['irr_mm'].astype(float)
    mgt_df['irr_mm'].replace(-999, np.nan, inplace=True)
    return mgt_df


def read_output_hru(wd):
    with open(os.path.join(wd, 'output.hru'), 'r') as f:
        content = f.readlines()
    lulc = [(i[:4]) for i in content[9:]]
    hrus = [str(i[10:19]) for i in content[9:]]
    subs = [int(i[19:24]) for i in content[9:]]
    mons = [(i[29:34]) for i in content[9:]]
    areas = [float(i[34:44]) for i in content[9:]]
    irr = [float(i[74:84]) for i in content[9:]]

    hru_df = pd.DataFrame(
        np.column_stack([lulc, hrus, subs, mons, areas, irr]),
        columns=['lulc', 'hru', 'sub', 'mon', 'area_km2', 'irr_mm'])

    conv_types = {'hru':str, 'sub':int, 'mon':float, 'area_km2':float, 'irr_mm':float}
    hru_df = hru_df.astype(conv_types)
    hru_df = hru_df.loc[hru_df['mon'] < 13]
    hru_df['mon'] = hru_df['mon'].astype(int)
    hru_df['irr_m3'] = (hru_df['area_km2']*1000000) * (hru_df['irr_mm']*0.001)

    return hru_df


def read_output_sub(wd):
    with open(os.path.join(wd, 'output.sub'), 'r') as f:
        content = f.readlines()
    subs = [int(i[6:10]) for i in content[9:]]
    mons = [float(i[19:24]) for i in content[9:]]
    preps = [float(i[34:44]) for i in content[9:]]
    # pets = [float(i[54:64]) for i in content[9:]]
    ets = [float(i[64:74]) for i in content[9:]]
    sws = [float(i[74:84]) for i in content[9:]]
    percs = [float(i[84:94]) for i in content[9:]]
    surqs = [float(i[94:104]) for i in content[9:]]
    gwqs = [float(i[104:114]) for i in content[9:]]
    latq = [float(i[184:194]) for i in content[9:]] 
    sub_df = pd.DataFrame(
        np.column_stack([subs, mons, preps, sws, latq, surqs, ets, percs, gwqs]),
        columns=["subs","mons", "precip", "sw", "latq", "surq", "et", "perco", "gwq"])

    # conv_types = {'hru':str, 'sub':int, 'mon':float, 'area_km2':float, 'irr_mm':float}
    # hru_df = hru_df.astype(conv_types)
    sub_df = sub_df.loc[sub_df['mons'] < 13]
    sub_df['mons'] = sub_df['mons'].astype(int)
    sub_df['subs'] = sub_df['subs'].astype(int)

    return sub_df


# NOTE: let's implement it in QSWATMOD
def export_gwsw_swatToExcel(wd, startDate, scdate, ecdate, nsubs):

    filename = "swatmf_out_SWAT_gwsw_monthly"
    data = np.loadtxt(
                    os.path.join(wd, filename),
                    skiprows=2,
                    comments=["month:", "Layer"])
    df = np.reshape(data[:, 1], (int(len(data)/nsubs), nsubs))
    df2 = pd.DataFrame(df)
    df2.index = pd.date_range(startDate, periods=len(df[:,0]), freq='M')
    df2 = df2[scdate:ecdate]
    mdf = df2.groupby(df2.index.month).mean()
    mdf = mdf.T
    mdf.columns = [
                'Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

    mdf.insert(0, "rchno", [x+1 for x in mdf.index] , True)
    mdf.to_excel('{}.xlsx'.format(filename), index=False)
    print(mdf)


# def cvtHd5ToArray(wd, f):

#     hf = hdf.File(os.path.join(wd, f), 'r')
#     # data = hf.get('Arrays/top1').value
#     data = np.array(hf["River": 1])
#     hf.close()
#     return data

def cvt_bas_array(wd, infile, nrows, ncols):

    with open(os.path.join(wd, infile), 'r') as f:
        data = []
        for line in f.readlines():
            if not line.startswith("#"):
                data.append(line.replace('\n', '').split())

    ii = 2 # starting line
    bas = []
    while data[ii][0] != "INTERNAL":
        for j in range(len(data[ii])):
            bas.append(int(data[ii][j]))
        ii += 1
    bas = np.array(bas).reshape([nrows, ncols])
    np.savetxt(os.path.join(wd, 'stuff.dat'), bas.astype(int), fmt='%i',delimiter='\t')
    print(bas)



if __name__ == '__main__':
    wd = "D:/Projects/Watersheds/Gumu/Analysis/APEX-MODFLOWs/qsm_50_rt_test/qsm_50/SWAT-MODFLOW"
    infile = "mf_50.bas"
    nrows = 123
    ncols = 62

    # print(os.path.abspath(swatmf.__file__))
    cvt_bas_array(wd, infile, nrows, ncols)