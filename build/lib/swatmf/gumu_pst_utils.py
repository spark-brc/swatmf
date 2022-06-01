""" PEST support utilities for Gumu: 05/26/2022 created by Seonggyu Park
"""

import pandas as pd
import numpy as np

def extract_stf_dates(subs, sim_start, warmup, cal_start, cal_end, dates):
    rch_file = 'output.rch'
    sim_start =  sim_start[:-4] + str(int(sim_start[-4:])+ int(warmup))

    rch_file = 'output.rch'
    for i in subs:
        sim_stf = pd.read_csv(
                        rch_file,
                        delim_whitespace=True,
                        skiprows=9,
                        usecols=[1, 3, 6],
                        names=["date", "filter", "stf_sim"],
                        index_col=0)

        sim_stf_f = sim_stf.loc[i]
        sim_stf_f = sim_stf_f.drop(['filter'], axis=1)
        sim_stf_f.index = pd.date_range(sim_start, periods=len(sim_stf_f.stf_sim))
        sim_stf_f = sim_stf_f[cal_start:cal_end]
        sim_stf_f.to_csv('stf_{:03d}.txt'.format(i), sep='\t', encoding='utf-8', index=True, header=False, float_format='%.7e')
        print('stf_{:03d}.txt file has been created...'.format(i))
    print('Finished ...')


def extract_hg_wt_mean(hg_wt_subs, sim_start, warmup, cal_start, cal_end, dates):
    hg_rch_file = 'output-mercury.rch'
    sim_start =  sim_start[:-4] + str(int(sim_start[-4:])+ int(warmup))
    hg_df = pd.read_csv(hg_rch_file,
                        delim_whitespace=True,
                        skiprows=2,
                        usecols=["RCH", "Hg2PmgSto"],
                        index_col=0
                    )
    hg_df = hg_df.loc["REACH"]
    hg_wt_sims = pd.DataFrame()
    for i in hg_wt_subs:
        hg_dff = hg_df.loc[hg_df["RCH"] == int(i)]
        hg_dff.index = pd.date_range(sim_start, periods=len(hg_dff))
        hg_dfs = hg_dff[cal_start:cal_end]
        hg_dfs = hg_dfs.rename({'Hg2PmgSto': 'sub{:03d}'.format(i)}, axis=1)
        hg_wt_sims = pd.concat([hg_wt_sims, hg_dfs.loc[:, 'sub{:03d}'.format(i)]], axis=1)
    hg_wt_sims.index = pd.to_datetime(hg_wt_sims.index)
    hg_wt_simss = hg_wt_sims.loc[dates].mean()
    dff = pd.DataFrame(hg_wt_simss).T
    dff.index = pd.to_datetime([cal_end])
    for j in hg_wt_subs:
        dff['sub{:03d}'.format(j)].to_csv(
                'hg_wt_{:03d}.txt'.format(j),
                sep='\t', encoding='utf-8', index=True, header=False,
                float_format='%.7e')
        print('hg_wt_{:03d}.txt file has been created...'.format(j))
    print('Finished ...')

def extract_hg_sed_mean(hg_sed_subs, sim_start, warmup, cal_start, cal_end, dates):
    hg_rch_file = 'output-mercury.rch'
    sim_start =  sim_start[:-4] + str(int(sim_start[-4:])+ int(warmup))
    hg_df = pd.read_csv(hg_rch_file,
                        delim_whitespace=True,
                        skiprows=2,
                        usecols=["RCH", "SedTHgCppm"],
                        index_col=0
                    )
    hg_df = hg_df.loc["REACH"]
    hg_sed_sims = pd.DataFrame()
    for i in hg_sed_subs:
        hg_dff = hg_df.loc[hg_df["RCH"] == int(i)]
        hg_dff.index = pd.date_range(sim_start, periods=len(hg_dff))
        hg_dfs = hg_dff[cal_start:cal_end]
        hg_dfs = hg_dfs.rename({'SedTHgCppm': 'sub{:03d}'.format(i)}, axis=1)
        hg_sed_sims = pd.concat([hg_sed_sims, hg_dfs.loc[:, 'sub{:03d}'.format(i)]], axis=1)
    hg_sed_sims.index = pd.to_datetime(hg_sed_sims.index)
    hg_sed_sims_mon = hg_sed_sims.resample('M').mean()
    hg_sed_simss = hg_sed_sims_mon.loc[dates].mean()
    dff = pd.DataFrame(hg_sed_simss).T
    dff.index = pd.to_datetime([cal_end])
    for j in hg_sed_subs:
        dff['sub{:03d}'.format(j)].to_csv(
                'hg_sed_{:03d}.txt'.format(j),
                sep='\t', encoding='utf-8', index=True, header=False,
                float_format='%.7e')
        print('hg_sed_{:03d}.txt file has been created...'.format(j))
    print('Finished ...')


def hg_obd_to_ins(hg_sim_file, obd_file, col_name, cal_start, cal_end, time_step=None):
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
    #     stfobd_file = 'stf_day.obd'
    # if time_step == 'month':
    #     stfobd_file = 'stf_mon.obd'


    stf_obd = pd.read_csv(
                        obd_file,
                        sep='\t',
                        usecols=['date', col_name],
                        index_col=0,
                        parse_dates=True,
                        na_values=[-999, '']
                        )
    stf_obd = stf_obd[cal_start:cal_end]

    stf_sim = pd.read_csv(
                        hg_sim_file,
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

    with open(hg_sim_file+'.ins', "w", newline='') as f:
        f.write("pif ~" + "\n")
        result['{}_ins'.format(col_name)].to_csv(f, sep='\t', encoding='utf-8', index=False, header=False)
    print('{}.ins file has been created...'.format(hg_sim_file))
    return result['{}_ins'.format(col_name)]
