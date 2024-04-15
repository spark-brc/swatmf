""" PEST support utilities: 12/4/2019 created by Seonggyu Park
"""

# from lib2to3.pgen2.token import NEWLINE
import pandas as pd
import numpy as np
import time
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT
import os
import shutil
import socket
import multiprocessing as mp
import csv
from tqdm import tqdm
from termcolor import colored
# from colorama import init
# from colorama import Fore, Style


opt_files_path = os.path.join(
                    os.path.dirname(os.path.abspath( __file__ )),
                    'opt_files')
foward_path = os.path.dirname(os.path.abspath( __file__ ))


def create_swatmf_con(
                prj_dir, 
                swatmf_model, sim_start, warmup, cal_start, cal_end,
                subs=None, grids=None, riv_parm=None,
                baseflow=None,
                time_step=None,
                pp_included=None,
                grids_lyrs=None,
                avg_grids=None

                # depth_to_water=None, 
                ):
    """create swatmf.con file containg SWAT-MODFLOW model PEST initial settings

    Args:
        prj_dir (`str`): Optimization project working directory
        swatmf_model (`str`): SWAT-MODFLOW working directory
        subs (`list`): reach numbers to be extracted
        grids (`list`): grid numbers to be extracted
        sim_start (`str`): simulation start date e.g. '1/1/2000'
        warmup(`int`): warm-up period
        cal_start (`str`): calibration start date e.g., '1/1/2001'
        cal_end (`str`): calibration end date e.g., '12/31/2005'
        time_step (`str`, optional): model time step. Defaults to None ('day'). e.g., 'day', 'month', 'year'
        riv_parm (`str`, optional): river parameter activation. Defaults to None ('n').
        depth_to_water (`str`, optional): extracting simulated depth to water activation. Defaults to None ('n').
        baseflow (`str`, optional): extracting baseflow ratio activation. Defaults to None ('n').

    Returns:
        dataframe: return SWAT-MODFLOW PEST configure settings as dataframe and exporting it as swatmf.con file.
    """
    if subs is None:
        subs = 'n'
    if grids is None:
        grids = 'n'
    if riv_parm is None:
        riv_parm = 'n'
    else:
        riv_parm = 'y'
    if time_step is None:
        time_step = 'day'
    if baseflow is None:
        baseflow = 'n'
    else:
        baseflow = 'y'
    if pp_included is None:
        pp_included = 'n'
    # if depth_to_water is None:
    #     depth_to_water = 'n'
    if grids_lyrs is None:
        grids_lyrs = 'n'

    if avg_grids is None:
        avg_grids = 'n'
    col01 = [
        'prj_dir',
        'swatmf_model', 'sim_start', 'warm-up', 'cal_start', 'cal_end',
        'subs', 'grids',
        'riv_parm', 'baseflow',
        'time_step',
        'pp_included',
        'grids_lyrs',
        'avg_grids'
        ]
    col02 = [
        prj_dir,
        swatmf_model, sim_start, warmup, cal_start, cal_end, 
        subs, grids,
        riv_parm, baseflow,
        time_step,
        pp_included,
        grids_lyrs,
        avg_grids
        ]
    df = pd.DataFrame({'names': col01, 'vals': col02})

    main_opt_path = os.path.join(prj_dir, 'main_opt')
    if not os.path.isdir(main_opt_path):
        os.makedirs(main_opt_path)
    with open(os.path.join(prj_dir, 'main_opt', 'swatmf.con'), 'w', newline='') as f:
        f.write("# swatmf.con created by swatmf\n")
        df.to_csv(
            f, sep='\t',
            encoding='utf-8', 
            index=False, header=False)
    return df

def init_setup(prj_dir, swatmfwd, swatwd):
    filesToCopy = [
        "Absolute_SWAT_Values.txt",
        "i64pwtadj1.exe",
        "pestpp-glm.exe",
        "pestpp-ies.exe",
        "pestpp-opt.exe",
        "pestpp-sen.exe",
        "model.in",
        "SUFI2_LH_sample.exe",
        "Swat_Edit.exe",
        "swatmf_rel230922.exe"
        ]
    suffix = ' passed'
    print(" Creating 'main_opt' folder in working directory ...",  end='\r', flush=True)

    main_opt_path = os.path.join(prj_dir, 'main_opt')

    if not os.path.isdir(main_opt_path):
        os.makedirs(main_opt_path)
    filelist = [f for f in os.listdir(swatmfwd) if os.path.isfile(os.path.join(swatmfwd, f))]
    for i in tqdm(filelist):
        # print(i)
        # if os.path.getsize(os.path.join(swatwd, i)) != 0:
        shutil.copy2(os.path.join(swatmfwd, i), main_opt_path)
    print(" Creating 'main_opt' folder ..." + colored(suffix, 'green'))
    # create backup
    print(" Creating 'backup' folder ...",  end='\r', flush=True)
    if not os.path.isdir(os.path.join(main_opt_path, 'backup')):
        os.makedirs(os.path.join(main_opt_path, 'backup'))
        filelist = [f for f in os.listdir(swatwd) if os.path.isfile(os.path.join(swatwd, f))]
        
        # filelist =  os.listdir(swatwd)
        for i in tqdm(filelist):
            # print(i)
            # if os.path.getsize(os.path.join(swatwd, i)) != 0:
            shutil.copy2(os.path.join(swatwd, i), os.path.join(main_opt_path, 'backup'))
    print(" Creating 'backup' folder ..." + colored(suffix, 'green'))

    # create echo
    print(" Creating 'echo' folder ...",  end='\r', flush=True)
    if not os.path.isdir(os.path.join(main_opt_path, 'echo')):
        os.makedirs(os.path.join(main_opt_path, 'echo'))
    print(" Creating 'echo' folder ..." + colored(suffix, 'green'))
    # create sufi2
    print(" Creating 'sufi2.in' folder ...",  end='\r', flush=True)
    if not os.path.isdir(os.path.join(main_opt_path, 'sufi2.in')):
        os.makedirs(os.path.join(main_opt_path, 'sufi2.in'))
    print(" Creating 'sufi2.in' folder ..."  + colored(suffix, 'green'))

    for j in filesToCopy:
        if not os.path.isfile(os.path.join(main_opt_path, j)):
            shutil.copy2(os.path.join(opt_files_path, j), os.path.join(main_opt_path, j))
            print(" '{}' file copied ...".format(j) + colored(suffix, 'green'))
    if not os.path.isfile(os.path.join(main_opt_path, 'forward_run.py')):
        shutil.copy2(os.path.join(foward_path, 'forward_run.py'), os.path.join(main_opt_path, 'forward_run.py'))
        print(" '{}' file copied ...".format('forward_run.py') + colored(suffix, 'green'))
    os.chdir(main_opt_path)       

def extract_day_stf(channels, start_day, warmup, cali_start_day, cali_end_day):
    """extract a daily simulated streamflow from the output.rch file,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2005'

    Example:
        sm_pst_utils.extract_month_stf('path', [9, 60], '1/1/1993', '1/1/1993', '12/31/2000')
    """
    rch_file = 'output.rch'
    start_day =  start_day[:-4] + str(int(start_day[-4:])+ int(warmup))

    rch_file = 'output.rch'
    for i in channels:
        sim_stf = pd.read_csv(
                        rch_file,
                        sep=r'\s+',
                        skiprows=9,
                        usecols=[1, 3, 6],
                        names=["date", "filter", "stf_sim"],
                        index_col=0)

        sim_stf_f = sim_stf.loc[i]
        sim_stf_f = sim_stf_f.drop(['filter'], axis=1)
        sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.stf_sim))
        sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
        sim_stf_f.to_csv('stf_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        print('stf_{:03d}.txt file has been created...'.format(i))
    print('Finished ...')


def extract_month_stf(channels, start_day, warmup, cali_start_day, cali_end_day):
    """extract a simulated streamflow from the output.rch file,
       store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2005'

    Example:
        sm_pst_utils.extract_month_stf('path', [9, 60], '1/1/1993', '1/1/1993', '12/31/2000')
    """
    rch_file = 'output.rch'
    start_day =  start_day[:-4] + str(int(start_day[-4:]) + int(warmup))
    for i in channels:
        sim_stf = pd.read_csv(
                        rch_file,
                        delim_whitespace=True,
                        skiprows=9,
                        usecols=[1, 3, 6],
                        names=["date", "filter", "stf_sim"],
                        index_col=0)

        sim_stf_f = sim_stf.loc[i]
        sim_stf_f = sim_stf_f[sim_stf_f['filter'] < 13]
        sim_stf_f = sim_stf_f.drop(['filter'], axis=1)
        sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.stf_sim), freq='M')
        sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
        sim_stf_f.to_csv('stf_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        print('stf_{:03d}.txt file has been created...'.format(i))
    print('Finished ...')


def extract_month_baseflow(channels, start_day, cali_start_day, cali_end_day):
    """ extract a simulated baseflow rates from the output.sub file,
        store it in each channel file.

    Args:
        - sub_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2005'

    Example:
        sm_pst_utils.extract_month_baseflow('path', [9, 60], '1/1/1993', '1/1/1993', '12/31/2000')
    """
    sub_file = 'output.sub' 
    gwqs = []
    subs = []
    for i in channels:
        sim_stf = pd.read_csv(
                        sub_file,
                        delim_whitespace=True,
                        skiprows=9,
                        usecols=[1, 3, 10, 11, 19],
                        names=["date", "filter", "surq", "gwq", "latq"],
                        index_col=0,
                        dtype={'filter': str})
        
        sim_stf_f = sim_stf.loc[i]
        # sim_stf_f["filter"]= sim_stf_f["filter"].astype(str) 
        sim_stf_f = sim_stf_f[sim_stf_f['filter'].astype(str).map(len) < 13]
        sim_stf_f = sim_stf_f.drop(['filter'], axis=1)
        sim_stf_f.index = pd.date_range(start_day, periods=len(sim_stf_f.surq), freq='M')
        sim_stf_f = sim_stf_f[cali_start_day:cali_end_day]
        # sim_stf_f.to_csv('gwq_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        
        sim_stf_f['surq'] = sim_stf_f['surq'].astype(float)
        sim_stf_f['bf_rate'] = sim_stf_f['gwq']/ (sim_stf_f['surq'] + sim_stf_f['latq'] + sim_stf_f['gwq'])
        sim_stf_f.loc[sim_stf_f['gwq'] < 0, 'bf_rate'] = 0     
        bf_rate = sim_stf_f['bf_rate'].mean()
        # bf_rate = bf_rate.item()
        subs.append('bfr_{:03d}'.format(i))
        gwqs.append(bf_rate)
        print('Average baseflow rate for {:03d} has been calculated ...'.format(i))
    # Combine lists into array
    bfr_f = np.c_[subs, gwqs]
    with open('baseflow_ratio.out', "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for item in bfr_f:
            writer.writerow([(item[0]),
                '{:.4f}'.format(float(item[1]))
                ])
    print('Finished ...\n')


def extract_depth_to_water(grid_ids, start_day, end_day, time_step="day"):
    """extract a simulated depth to water using modflow.obs and swatmf_out_MF_obs,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2000'

    Example:
        pest_utils.extract_depth_to_water('path', [9, 60], '1/1/1993', '12/31/2000')
    """
    if not os.path.exists('swatmf_out_MF_obs'):
        raise Exception("'swatmf_out_MF_obs' file not found")
    if not os.path.exists('modflow.obs'):
        raise Exception("'modflow.obs' file not found")
    mf_obs_grid_ids = pd.read_csv(
                        'modflow.obs',
                        sep=r'\s+',
                        usecols=[3, 4],
                        skiprows=2,
                        header=None
                        )
    col_names = mf_obs_grid_ids.iloc[:, 0].tolist()

    # set index by modflow grid ids
    mf_obs_grid_ids = mf_obs_grid_ids.set_index([3])

    mf_sim = pd.read_csv(
                        'swatmf_out_MF_obs', skiprows=1, sep=r'\s+',
                        names=col_names,
                        # usecols=grid_ids,
                        )
    mf_sim.index = pd.date_range(start_day, periods=len(mf_sim))
    if time_step == "day":
        mf_sim = mf_sim[start_day:end_day]
    if time_step == "month":
        mf_sim = mf_sim[start_day:end_day].resample('M').mean()
    for i in grid_ids:
        elev = mf_obs_grid_ids.loc[i].values  # use land surface elevation to get depth to water
        (mf_sim.loc[:, i] - elev).to_csv(
                        'dtw_{}.txt'.format(i), sep='\t', encoding='utf-8',
                        index=True, header=False, float_format='%.7e'
                        )
        print('dtw_{}.txt file has been created...'.format(i))
    print('Finished ...')


def extract_depth_to_water_layer(dtw_df, start_day, end_day, time_step="day"):
    """create depth to water extracted file with layer info

    :param dtw_df: depth to water dataframe created by handler.get_gw_sim function
    :type dtw_df: dataframe
    :param start_day: sim startdate
    :type start_day: str
    :param end_day: sim enddate
    :type end_day: str
    :param time_step: time step, defaults to "day"
    :type time_step: str, optional
    """
    if time_step == "day":
        mf_sim = dtw_df[start_day:end_day]
    if time_step == "month":
        mf_sim = dtw_df[start_day:end_day].resample('M').mean()
    for col in mf_sim.columns:
        mf_sim.loc[:, col].to_csv(
                        '{}.txt'.format(col), sep='\t', encoding='utf-8',
                        index=True, header=False, float_format='%.7e'
                        )    
        print('{}.txt file has been created...'.format(col))
    print('Finished ...')


def extract_avg_depth_to_water(
                grid_ids, start_day, 
                avg_stdate, avg_eddate,
                time_step="day"
                ):
    """extract a simulated depth to water using modflow.obs and swatmf_out_MF_obs,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1985'
        - end_day ('str'): simulation end day e.g. '12/31/2000'

    Example:
        pest_utils.extract_depth_to_water('path', [9, 60], '1/1/1993', '12/31/2000')
    """
    if not os.path.exists('swatmf_out_MF_obs'):
        raise Exception("'swatmf_out_MF_obs' file not found")
    if not os.path.exists('modflow.obs'):
        raise Exception("'modflow.obs' file not found")
    mf_obs_grid_ids = pd.read_csv(
                        'modflow.obs',
                        sep=r'\s+',
                        usecols=[3, 4],
                        skiprows=2,
                        header=None
                        )
    col_names = mf_obs_grid_ids.iloc[:, 0].tolist()

    # set index by modflow grid ids
    mf_obs_grid_ids = mf_obs_grid_ids.set_index([3])

    mf_sim = pd.read_csv(
                        'swatmf_out_MF_obs', skiprows=1, sep=r'\s+',
                        names=col_names,
                        # usecols=grid_ids,
                        )
    mf_sim.index = pd.date_range(start_day, periods=len(mf_sim))
    # mf_sim = mf_sim[avg_stdate:avg_eddate]
    if time_step == "day":
        mf_sim = mf_sim[avg_stdate:avg_eddate].mean()
    # if time_step == "month":
    #     mf_sim = mf_sim[avg_stdate:avg_eddate].resample('M').mean()
    df_avg = pd.DataFrame()
    dtws = []
    for i in grid_ids:
        elev = mf_obs_grid_ids.loc[i, 4]
        dtws.append(mf_sim.loc[i] - elev)
    df_avg = pd.DataFrame()
    df_avg['grid_ids'] = grid_ids
    df_avg['sims'] = dtws
    df_avg.to_csv("dtw_avg.txt", sep='\t', index=False, header=False, float_format='%.7e')

        # use land surface elevation to get depth to water
        # (mf_sim.loc[:, i] - elev).to_csv(
        #                 'dtw_{}.txt'.format(i), sep='\t', encoding='utf-8',
        #                 index=True, header=False, float_format='%.7e'
        #                 )
        # print('dtw_{}.txt file has been created...'.format(i))
    # print('Finished ...')

def mf_avg_obd_to_ins(wt_file=None):
    """extract a simulated groundwater levels from the  file,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1993'
        - end_day ('str'): simulation end day e.g. '12/31/2000'

    Example:
        pest_utils.extract_month_str('path', [9, 60], '1/1/1993', '12/31/2000')
    """ 
    if wt_file is None:
        wt_file = "dtw_avg.txt"
    mf_obd_file = "dtw_avg.obd.csv"
    col_name = "avg_dtw"
    
    mf_obd = pd.read_csv(
                        mf_obd_file,
                        index_col=0,
                        )
    wt_sim = pd.read_csv(
                        wt_file,
                        delim_whitespace=True,
                        names=['grid_id', 'dtw_avg_sim'],
                        index_col=0,
                        )
    result = pd.concat([mf_obd, wt_sim], axis=1)
    result['ins'] = (
                    'l1 w !{}_'.format(col_name) + result.index.map(str) + '!'
                    )
    result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])
    with open(wt_file+'.ins', "w", newline='') as f:
        f.write("pif ~" + "\n")
        result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    print('{}.ins file has been created...'.format(wt_file))
    return result['{}_ins'.format(col_name)]


def stf_obd_to_ins(srch_file, col_name, cal_start, cal_end, time_step=None):
    """extract a simulated streamflow from the output.rch file,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1993'
        - end_day ('str'): simulation end day e.g. '12/31/2000'
        - time_step (`str`): day, month, year

    Example:
        pest_utils.extract_month_stf('path', [9, 60], '1/1/1993', '12/31/2000')
    """ 
    if time_step is None:
        time_step = 'day'
        stfobd_file = 'stf_day.obd.csv'
    if time_step == 'month':
        stfobd_file = 'stf_mon.obd.csv'


    stf_obd = pd.read_csv(
                        stfobd_file,
                        # sep='\t',
                        usecols=['date', col_name],
                        index_col=0,
                        parse_dates=True,
                        na_values=[-999, '']
                        )
    stf_obd = stf_obd[cal_start:cal_end]

    stf_sim = pd.read_csv(
                        srch_file,
                        delim_whitespace=True,
                        names=["date", "stf_sim"],
                        index_col=0,
                        parse_dates=True)

    result = pd.concat([stf_obd, stf_sim], axis=1)

    result['tdate'] = pd.to_datetime(result.index)
    result['month'] = result['tdate'].dt.month
    result['year'] = result['tdate'].dt.year
    result['day'] = result['tdate'].dt.day

    if time_step == 'day':
        result['ins'] = (
                        'l1 w !{}_'.format(col_name) + result["year"].map(str) +
                        result["month"].map('{:02d}'.format) +
                        result["day"].map('{:02d}'.format) + '!'
                        )
    elif time_step == 'month':
        result['ins'] = 'l1 w !{}_'.format(col_name) + result["year"].map(str) + result["month"].map('{:02d}'.format) + '!'
    else:
        print('are you performing a yearly calibration?')
    result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])

    with open(srch_file+'.ins', "w", newline='') as f:
        f.write("pif ~" + "\n")
        result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    print('{}.ins file has been created...'.format(srch_file))
    return result['{}_ins'.format(col_name)]


def mf_obd_to_ins(wt_file, col_name, cal_start, cal_end, time_step="day"):
    """extract a simulated groundwater levels from the  file,
        store it in each channel file.

    Args:
        - rch_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1993'
        - end_day ('str'): simulation end day e.g. '12/31/2000'

    Example:
        pest_utils.extract_month_str('path', [9, 60], '1/1/1993', '12/31/2000')
    """ 
    if time_step == "day":
        mf_obd_file = "dtw_day.obd.csv"
    if time_step == "month":
        mf_obd_file = "dtw_mon.obd.csv"


    mf_obd = pd.read_csv(
                        mf_obd_file,
                        usecols=['date', col_name],
                        index_col=0,
                        na_values=[-999, ""],
                        parse_dates=True,
                        )
    mf_obd = mf_obd[cal_start:cal_end]

    wt_sim = pd.read_csv(
                        wt_file,
                        delim_whitespace=True,
                        names=["date", "stf_sim"],
                        index_col=0,
                        parse_dates=True)

    result = pd.concat([mf_obd, wt_sim], axis=1)

    result['tdate'] = pd.to_datetime(result.index)
    result['day'] = result['tdate'].dt.day
    result['month'] = result['tdate'].dt.month
    result['year'] = result['tdate'].dt.year
    result['ins'] = (
                    'l1 w !{}_'.format(col_name) + result["year"].map(str) +
                    result["month"].map('{:02d}'.format) +
                    result["day"].map('{:02d}'.format) + '!'
                    )
    result['{}_ins'.format(col_name)] = np.where(result[col_name].isnull(), 'l1', result['ins'])

    with open(wt_file+'.ins', "w", newline='') as f:
        f.write("pif ~" + "\n")
        result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    print('{}.ins file has been created...'.format(wt_file))

    return result['{}_ins'.format(col_name)]


def extract_month_avg(cha_file, channels, start_day, cal_day=None, end_day=None):
    """extract a simulated streamflow from the channel_day.txt file,
        store it in each channel file.

    Args:
        - cha_file (`str`): the path and name of the existing output file
        - channels (`list`): channel number in a list, e.g. [9, 60]
        - start_day ('str'): simulation start day after warmup period, e.g. '1/1/1993'
        - end_day ('str'): simulation end day e.g. '12/31/2000'

    Example:
        pest_utils.extract_month_str('path', [9, 60], '1/1/1993', '12/31/2000')
    """

    for i in channels:
        # Get only necessary simulated streamflow and convert monthly average streamflow
        os.chdir(cha_file)
        print(os.getcwd())
        df_stf = pd.read_csv(
                            "channel_day.txt",
                            delim_whitespace=True,
                            skiprows=3,
                            usecols=[6, 8],
                            names=['name', 'flo_out'],
                            header=None
                            )
        df_stf = df_stf.loc[df_stf['name'] == 'cha{:02d}'.format(i)]
        df_stf.index = pd.date_range(start_day, periods=len(df_stf.flo_out))
        mdf = df_stf.resample('M').mean()
        mdf.index.name = 'date'
        if cal_day is None:
            cal_day = start_day
        else:
            cal_day = cal_day
        if end_day is None:
            mdf = mdf[cal_day:]
        else:
            mdf = mdf[cal_day:end_day]
        mdf.to_csv('cha_mon_avg_{:03d}.txt'.format(i), sep='\t', float_format='%.7e')
        print('cha_{:03d}.txt file has been created...'.format(i))
        return mdf


def model_in_to_template_file(tpl_file=None):
    """write a template file for a SWAT parameter value file (model.in).

    Args:
        model_in_file (`str`): the path and name of the existing model.in file
        tpl_file (`str`, optional):  template file to write. If None, use
            `model_in_file` +".tpl". Default is None
    Note:
        Uses names in the first column in the pval file as par names.

    Example:
        pest_utils.model_in_to_template_file('path')

    Returns:
        **pandas.DataFrame**: a dataFrame with template file information
    """
    model_in_file = 'model.in'
    if tpl_file is None:
        tpl_file = model_in_file + ".tpl"
    mod_df = pd.read_csv(
                        model_in_file,
                        delim_whitespace=True,
                        header=None, skiprows=0,
                        names=["parnme", "parval1"])
    mod_df.index = mod_df.parnme
    mod_df.loc[:, "tpl"] = mod_df.parnme.apply(lambda x: " ~   {0:15s}   ~".format(x[3:-4]))
    # mod_df.loc[:, "tpl"] = mod_df.parnme.apply(lambda x: " ~   {0:15s}   ~".format(x[3:7]))
    with open(tpl_file, 'w') as f:
        f.write("ptf ~\n")
        # f.write("{0:10d} #NP\n".format(mod_df.shape[0]))
        SFMT_LONG = lambda x: "{0:<50s} ".format(str(x))
        f.write(mod_df.loc[:, ["parnme", "tpl"]].to_string(
                                                        col_space=0,
                                                        formatters=[SFMT_LONG, SFMT_LONG],
                                                        index=False,
                                                        header=False,
                                                        justify="left"))
    return mod_df


def riv_par_to_template_file(riv_par_file, tpl_file=None):
    """write a template file for a SWAT parameter value file (model.in).

    Args:
        model_in_file (`str`): the path and name of the existing model.in file
        tpl_file (`str`, optional):  template file to write. If None, use
            `model_in_file` +".tpl". Default is None
    Note:
        Uses names in the first column in the pval file as par names.

    Example:
        pest_utils.model_in_to_template_file('path')

    Returns:
        **pandas.DataFrame**: a dataFrame with template file information
    """

    if tpl_file is None:
        tpl_file = riv_par_file + ".tpl"
    mf_par_df = pd.read_csv(
                        riv_par_file,
                        delim_whitespace=True,
                        header=None, skiprows=2,
                        names=["parnme", "chg_type", "parval1"])
    mf_par_df.index = mf_par_df.parnme
    mf_par_df.loc[:, "tpl"] = mf_par_df.parnme.apply(lambda x: " ~   {0:15s}   ~".format(x))
    with open(tpl_file, 'w') as f:
        f.write("ptf ~\n# modflow_par template file.\n")
        f.write("NAME   CHG_TYPE    VAL\n")
        f.write(mf_par_df.loc[:, ["parnme", "chg_type", "tpl"]].to_string(
                                                        col_space=0,
                                                        formatters=[SFMT, SFMT, SFMT],
                                                        index=False,
                                                        header=False,
                                                        justify="left"))
    return mf_par_df
    # df_p.iloc[:, 0] = df_p.iloc[:, 0].map(lambda x: '{:<12s}'.format(x))
    # df_p.iloc[:, 2] = df_p.iloc[:, 2].map(lambda x: '{:<12.5e}'.format(x))


def fix_riv_pkg(wd, riv_file):
    """ Delete duplicate river cells in an existing MODFLOW river packgage.

    Args:
        wd ('str'): path of the working directory.
        riv_file ('str'): name of river package.
    """

    with open(os.path.join(wd, riv_file), "r") as fp:
        lines = fp.readlines()
        new_lines = []
        for line in lines:
            #- Strip white spaces
            line = line.strip()
            if line not in new_lines:
                new_lines.append(line)
            else:
                print(line)

    output_file = "{}_fixed".format(riv_file)
    with open(os.path.join(wd, output_file), "w") as fp:
        fp.write("\n".join(new_lines))    


def _remove_readonly(func, path, excinfo):
    """remove readonly dirs, apparently only a windows issue
    add to all rmtree calls: shutil.rmtree(**,onerror=remove_readonly), wk"""
    os.chmod(path, 128)  # stat.S_IWRITE==128==normal
    func(path)


# NOTE: Update description
def execute_beopest(
                master_dir, pst, num_workers=None, worker_root='..', port=4005, local=True,
                reuse_workers=None, copy_files=None, restart=None):
    """Execute BeoPEST and workers on the local machine

    Args:
        master_dir (str): 
        pst (str): [description]
        num_workers ([type], optional): [description]. Defaults to None.
        worker_root (str, optional): [description]. Defaults to '..'.
        port (int, optional): [description]. Defaults to 4005.
        local (bool, optional): [description]. Defaults to True.
        reuse_workers ([type], optional): [description]. Defaults to None.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
    """

    if not os.path.isdir(master_dir):
        raise Exception("master dir '{0}' not found".format(master_dir))
    if not os.path.isdir(worker_root):
        raise Exception("worker root dir not found")
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = int(num_workers)

    if local:
        hostname = "localhost"
    else:
        hostname = socket.gethostname()

    base_dir = os.getcwd()
    port = int(port)
    cwd = os.chdir(master_dir)
    if restart is None:
        os.system("start cmd /k beopest64 {0} /h :{1}".format(pst, port))
    else:
        os.system("start cmd /k beopest64 {0} /r /h :{1}".format(pst, port))
    time.sleep(1.5) # a few cycles to let the master get ready
    
    tcp_arg = "{0}:{1}".format(hostname,port)
    worker_dirs = []
    for i in range(num_workers):
        new_worker_dir = os.path.join(worker_root,"worker_{0}".format(i))
        if os.path.exists(new_worker_dir) and reuse_workers is None:
            try:
                shutil.rmtree(new_worker_dir, onerror=_remove_readonly)#, onerror=del_rw)
            except Exception as e:
                raise Exception("unable to remove existing worker dir:" + \
                                "{0}\n{1}".format(new_worker_dir,str(e)))
            try:
                shutil.copytree(master_dir,new_worker_dir)
            except Exception as e:
                raise Exception("unable to copy files from worker dir: " + \
                                "{0} to new worker dir: {1}\n{2}".format(master_dir,new_worker_dir,str(e)))
        elif os.path.exists(new_worker_dir) and reuse_workers is True:
            try:
                shutil.copyfile(pst, os.path.join(new_worker_dir, pst))
            except Exception as e:
                raise Exception("unable to copy *.pst from main worker: " + \
                                "{0} to new worker dir: {1}\n{2}".format(master_dir,new_worker_dir,str(e)))
        else:
            try:
                shutil.copytree(master_dir,new_worker_dir)
            except Exception as e:
                raise Exception("unable to copy files from worker dir: " + \
                                "{0} to new worker dir: {1}\n{2}".format(master_dir,new_worker_dir,str(e)))
        if copy_files is not None and reuse_workers is True:
            try:
                for f in copy_files:
                    shutil.copyfile(f, os.path.join(new_worker_dir, f))
            except Exception as e:
                raise Exception("unable to copy *.pst from main worker: " + \
                                "{0} to new worker dir: {1}\n{2}".format(master_dir,new_worker_dir,str(e)))
        cwd = new_worker_dir
        os.chdir(cwd)
        os.system("start cmd /k beopest64 {0} /h {1}".format(pst, tcp_arg))


# TODO: copy pst / option to use an existing worker
def execute_workers(
            worker_rep, pst, host, num_workers=None,
            start_id=None, worker_root='..', port=4005, reuse_workers=None, copy_files=None):
    """[summary]

    Args:
        worker_rep ([type]): [description]
        pst ([type]): [description]
        host ([type]): [description]
        num_workers ([type], optional): [description]. Defaults to None.
        start_id ([type], optional): [description]. Defaults to None.
        worker_root (str, optional): [description]. Defaults to '..'.
        port (int, optional): [description]. Defaults to 4005.

    Raises:
        Exception: [description]
        Exception: [description]
        Exception: [description]
        Exception: [description]
    """

    if not os.path.isdir(worker_rep):
        raise Exception("master dir '{0}' not found".format(worker_rep))
    if not os.path.isdir(worker_root):
        raise Exception("worker root dir not found")
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = int(num_workers)
    if start_id is None:
        start_id = 0
    else:
        start_id = start_id

    hostname = host
    base_dir = os.getcwd()
    port = int(port)
    cwd = os.chdir(worker_rep)
    tcp_arg = "{0}:{1}".format(hostname,port)

    for i in range(start_id, num_workers + start_id):
        new_worker_dir = os.path.join(worker_root,"worker_{0}".format(i))
        if os.path.exists(new_worker_dir) and reuse_workers is None:
            try:
                shutil.rmtree(new_worker_dir, onerror=_remove_readonly)#, onerror=del_rw)
            except Exception as e:
                raise Exception("unable to remove existing worker dir:" + \
                                "{0}\n{1}".format(new_worker_dir,str(e)))
            try:
                shutil.copytree(worker_rep,new_worker_dir)
            except Exception as e:
                raise Exception("unable to copy files from worker dir: " + \
                                "{0} to new worker dir: {1}\n{2}".format(worker_rep,new_worker_dir,str(e)))
        elif os.path.exists(new_worker_dir) and reuse_workers is True:
            try:
                shutil.copyfile(pst, os.path.join(new_worker_dir, pst))
            except Exception as e:
                raise Exception("unable to copy *.pst from main worker: " + \
                                "{0} to new worker dir: {1}\n{2}".format(worker_rep,new_worker_dir,str(e)))
        else:
            try:
                shutil.copytree(worker_rep,new_worker_dir)
            except Exception as e:
                raise Exception("unable to copy files from worker dir: " + \
                                "{0} to new worker dir: {1}\n{2}".format(worker_rep,new_worker_dir,str(e)))
        if copy_files is not None and reuse_workers is True:
            try:
                for f in copy_files:
                    shutil.copyfile(f, os.path.join(new_worker_dir, f))
            except Exception as e:
                raise Exception("unable to copy *.pst from main worker: " + \
                                "{0} to new worker dir: {1}\n{2}".format(worker_rep, new_worker_dir,str(e)))

        cwd = new_worker_dir
        os.chdir(cwd)
        os.system("start cmd /k beopest64 {0} /h {1}".format(pst, tcp_arg))


def cvt_stf_day_month_obd(obd_file):
    # create stf_mon.obd
    stf_obd = pd.read_csv(
                        obd_file,
                        sep='\t',
                        # usecols=['date', 'sub_37'],
                        index_col=0,
                        parse_dates=True,
                        na_values=[-999, '']
                        )
    stf_obd = stf_obd.resample('M').mean()
    stf_obd.to_csv(
        'stf_mon.obd.csv', index=True, index_label="date", na_rep=-999, float_format='%.7e')
    print("done ...")
    
if __name__ == '__main__':
    wd = "D:/Projects/Watersheds/MiddleBosque/Analysis/SWAT-MODFLOWs/optimization/main_opt"
    stdate = '1/1/1982'
    eddate = '12/31/1986'
    grid_ids = [5698, 5699]
    avg_stdate = '8/1/1985'
    avg_eddate = '8/31/1985'
    # print(os.path.abspath(swatmf.__file__))
    os.chdir(wd)
    mf_avg_obd_to_ins()
