""" PEST support visualizations: 02/09/2021 created by Seonggyu Park
    last modified day: 02/21/2021 by Seonggyu Park
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from hydroeval import evaluator, nse, rmse, pbias
import numpy as np
import math


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


def str_df(start_date, sub_number, time_step=None):
    if time_step is None:
        time_step = "D"
        strobd_file = "swat_rch_day.obd"
    else:
        time_step = "M"
        strobd_file = "swat_rch_mon.obd."
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
    df = df.drop('filter', axis=1)
    df.index = pd.date_range(start_date, periods=len(df.str_sim), freq=time_step)
    
    return df



def apex_str_df(rch_file, start_date, rch_num, obd_nam, time_step=None):
    
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


def str_sim_obd(plot_df): # NOTE: temp for report

    fig, ax = plt.subplots(figsize=(10, 3))
    # ax.grid(True)
    ax.plot(plot_df.index, plot_df.iloc[:, 0], label='Simulated', color='green', marker='^', alpha=0.7)
    ax.scatter(
        plot_df.index, plot_df.iloc[:, 1], label='Observed',
        facecolors="None", edgecolors='red',
        lw=1.5,
        alpha=0.4,
        )
    ax.plot(plot_df.index, plot_df.iloc[:, 1], color='red', alpha=0.4, zorder=2,)
    
    ax.set_ylabel("Stream Discharge $(m^3/day)$",fontsize=14)

    # ax.margins(y=0.2)
    ax.tick_params(axis='both', labelsize=12)
    
    # add stats
    org_stat = plot_df.dropna()
    sim_org = org_stat.iloc[:, 0].to_numpy()
    obd_org = org_stat.iloc[:, 1].to_numpy()
    df_nse = evaluator(nse, sim_org, obd_org)
    df_rmse = evaluator(rmse, sim_org, obd_org)
    df_pibas = evaluator(pbias, sim_org, obd_org)
    r_squared = (
        ((sum((obd_org - obd_org.mean())*(sim_org-sim_org.mean())))**2)/
        ((sum((obd_org - obd_org.mean())**2)* (sum((sim_org-sim_org.mean())**2))))
        )    
    ax.text(
        0.95, -0.2,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(df_nse[0], df_rmse[0], df_pibas[0], r_squared),
        horizontalalignment='right',fontsize=10,
        bbox=dict(facecolor='green', alpha=0.5),
        transform=ax.transAxes
        )     
    fig.tight_layout()
    lines, labels = fig.axes[0].get_legend_handles_labels()
    ax.legend(
        lines, labels, loc = 'upper left', ncol=5,
        # bbox_to_anchor=(0, 0.202),
        fontsize=12)
    # plt.legend()
    plt.show()

def str_plot(plot_df, prep=None): # NOTE: with precipitation data

    colnams = plot_df.columns.tolist()
    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.grid(True)
    ax.plot(plot_df.index, plot_df.iloc[:, 0], label='Simulated', color='green', marker='^', alpha=0.7)
    ax.scatter(
        plot_df.index, plot_df.iloc[:, 1], label='Observed',
        # color='red',
        facecolors="None", edgecolors='red',
        lw=1.5,
        alpha=0.4,
        # zorder=2,
        )
    ax.plot(plot_df.index, plot_df.iloc[:, 1], color='red', alpha=0.4, zorder=2,)
    
    if prep:
        ax2=ax.twinx()
        ax2.bar(
            plot_df.index, plot_df.prep, label='Precipitation',
            width=20,
            color="blue", align='center', alpha=0.5, zorder=0)
        ax2.set_ylabel("Precipitation $(mm)$",color="blue",fontsize=14)
        ax2.invert_yaxis()
        ax2.set_ylim(plot_df.prep.max()*3, 0)
        ax.set_ylabel("Stream Discharge $(m^3/day)$",fontsize=14)
        ax2.tick_params(axis='y', labelsize=12)    
    ax.margins(y=0.2)
    ax.tick_params(axis='both', labelsize=12)
    
    # add stats
    org_stat = plot_df.dropna()
    sim_org = org_stat.iloc[:, 0].to_numpy()
    obd_org = org_stat.iloc[:, 1].to_numpy()
    df_nse = evaluator(nse, sim_org, obd_org)
    df_rmse = evaluator(rmse, sim_org, obd_org)
    df_pibas = evaluator(pbias, sim_org, obd_org)
    r_squared = (
        ((sum((obd_org - obd_org.mean())*(sim_org-sim_org.mean())))**2)/
        ((sum((obd_org - obd_org.mean())**2)* (sum((sim_org-sim_org.mean())**2))))
        )    
    ax.text(
        0.95, 0.05,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(df_nse[0], df_rmse[0], df_pibas[0], r_squared),
        horizontalalignment='right',fontsize=10,
        bbox=dict(facecolor='green', alpha=0.5),
        transform=ax.transAxes
        )     
    fig.tight_layout()
    lines, labels = fig.axes[0].get_legend_handles_labels()
    ax.legend(
        lines, labels, loc = 'lower left', ncol=5,
        # bbox_to_anchor=(0, 0.202),
        fontsize=12)
    # plt.legend()
    plt.show()


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


def str_plot_test(plot_df, cal_period=None, val_period=None):

    if cal_period:
        cal_df = plot_df[cal_period[0]:cal_period[1]]
    if val_period:
        val_df = plot_df[val_period[0]:val_period[1]]
    colnams = plot_df.columns.tolist()
    # plot
    fig, ax = plt.subplots(figsize=(16, 4))
    
    ax.grid(True)
    # cali
    ax.plot(cal_df.index, cal_df.iloc[:, 0], label='Calibrated', color='green', marker='^', alpha=0.7)
    ax.plot(val_df.index, val_df.iloc[:, 0], label='Validated', color='m', marker='x', alpha=0.7)
    ax.scatter(
        plot_df.index, plot_df.iloc[:, 1], label='Observed',
        # color='red',
        facecolors="None", edgecolors='red',
        lw=1.5,
        alpha=0.4,
        # zorder=2,
        )
    ax.plot(plot_df.index, plot_df.iloc[:, 1], color='red', alpha=0.4, zorder=2,)
    ax2=ax.twinx()
    ax2.bar(
        plot_df.index, plot_df.prep, label='Precipitation',
        width=20,
        color="blue", align='center', alpha=0.5, zorder=0)
    ax2.set_ylabel("Precipitation $(mm)$",color="blue",fontsize=14)
    ax.set_ylabel("Stream Discharge $(m^3/day)$",fontsize=14)
    ax2.invert_yaxis()
    ax2.set_ylim(plot_df.prep.max()*3, 0)
    ax.margins(y=0.2)
    ax.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)    
    # add stats cal
    cal_nse, cal_rmse, cal_pbias, cal_rsquared = get_stats(cal_df)
    ax.text(
        0.48, 0.05,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(cal_nse[0], cal_rmse[0], cal_pbias[0], cal_rsquared),
        horizontalalignment='right',fontsize=12,
        bbox=dict(facecolor='green', alpha=0.5),
        transform=ax.transAxes
        )
    # add stats val
    val_nse, val_rmse, val_pbias, val_rsquared = get_stats(val_df)
    ax.text(
        0.50, 0.05,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(val_nse[0], val_rmse[0], val_pbias[0], val_rsquared),
        horizontalalignment='left',fontsize=12,
        bbox=dict(facecolor='m', alpha=0.5),
        transform=ax.transAxes
        )
    fig.tight_layout()
    lines, labels = fig.axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    ax.legend(
        lines+h2, labels+l2, loc = 'upper right', ncol=4,
        bbox_to_anchor=(1, 1.13),
        fontsize=12)
    # plt.legend()
    plt.savefig('mb_wt.png', dpi=300, bbox_inches="tight")
    print(os.getcwd())
    
    plt.show()



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

def dtw_df(start_date, grid_id, obd_nam, time_step=None):
    #NOTE: for swatmf
    """create a dataframe containing extract simulated and observed depth to water 

    Args:
        start_date (str): datetime index format e.g., "1/1/1999"
        grid_id (int): grid id
        obd_nam (str): column name from modflow obd file
        time_step (str, optional): timestep for measured data. Defaults to "D (day)".

    Returns:
        dataframe: a dataframe containing extract simulated and observed depth to water
    """
    
    if time_step is None:
        time_step = "D"
        mfobd_file = "dtw_day.obd.csv"
    else:
        time_step = "M"
        mfobd_file = "dtw_mon.obd.csv"

    mf_obs = pd.read_csv(
                        "modflow.obs",
                        delim_whitespace=True,
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        mfobd_file,
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""])

    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    output_wt = pd.read_csv(
                        "swatmf_out_MF_obs",
                        delim_whitespace=True,
                        skiprows = 1,
                        names = grid_id_lst,)
    output_wt = output_wt[str(grid_id)] - float(mf_obs.loc[int(grid_id)])
    output_wt.index = pd.date_range(start_date, periods=len(output_wt))

    if time_step == 'M':
        output_wt = output_wt.resample('M').mean()
    # if prep_sub is not None:
    #     # Get precipitation data from *.DYL
    #     prep_file = 'sub{}.DLY'.format(prep_sub)
    #     with open(prep_file) as f:
    #         content = f.readlines()    
    #     year = content[0][:6].strip()
    #     mon = content[0][6:10].strip()
    #     day = content[0][10:14].strip()
    #     prep = [float(i[32:38].strip()) for i in content]
    #     prep_stdate = "/".join((mon,day,year))
    #     prep_df =  pd.DataFrame(prep, columns=['prep'])
    #     prep_df.index = pd.date_range(prep_stdate, periods=len(prep))
    #     prep_df = prep_df.replace(9999, np.nan)
    #     # if time_step == "M":
    #     prep_df = prep_df.resample('M').mean()
    #     output_wt = pd.concat([output_wt, mfobd_df[obd_nam], prep_df], axis=1)
    # else:
    #     output_wt = pd.concat([output_wt, mfobd_df[obd_nam]], axis=1)
    output_wt = pd.concat([output_wt, mfobd_df[obd_nam]], axis=1)
    output_wt = output_wt[output_wt[str(grid_id)].notna()]

    return output_wt       


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


def wt_plot(plot_df):

    colnams = plot_df.columns.tolist()
    # plot
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.grid(True)
    ax.plot(plot_df.index, plot_df.iloc[:, 0], label='Simulated', color='green', marker='^', alpha=0.7)
    ax.scatter(
        plot_df.index, plot_df.iloc[:, 1], label='Observed',
        # color='red',
        facecolors="None", edgecolors='red',
        lw=1.5,
        alpha=0.4,
        # zorder=2,
        )
    ax.plot(plot_df.index, plot_df.iloc[:, 1], color='red', alpha=0.4, zorder=2,)
    ax2=ax.twinx()
    ax2.bar(
        plot_df.index, plot_df.prep, label='Precipitation',
        width=20,
        color="blue", align='center', alpha=0.5, zorder=0)
    ax2.set_ylabel("Precipitation $(mm)$",color="blue",fontsize=14)
    ax.set_ylabel("Depth to Water $(m)$",fontsize=14)
    ax2.invert_yaxis()
    ax2.set_ylim(plot_df.prep.max()*3, 0)
    ax.margins(y=0.2)
    ax.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)    
    
    # add stats
    plot_df = plot_df.drop('prep', axis=1)
    org_stat = plot_df.dropna()

    sim_org = org_stat.iloc[:, 0].to_numpy()
    obd_org = org_stat.iloc[:, 1].to_numpy()
    df_nse = evaluator(nse, sim_org, obd_org)
    df_rmse = evaluator(rmse, sim_org, obd_org)
    df_pibas = evaluator(pbias, sim_org, obd_org)
    r_squared = (
        ((sum((obd_org - obd_org.mean())*(sim_org - sim_org.mean())))**2)/
        ((sum((obd_org - obd_org.mean())**2)* (sum((sim_org - sim_org.mean())**2))))
        )      
    ax.text(
        0.95, 0.05,
        'NSE: {:.2f} | RMSE: {:.2f} | PBIAS: {:.2f} | R-Squared: {:.2f}'.format(df_nse[0], df_rmse[0], df_pibas[0], r_squared),
        horizontalalignment='right',fontsize=10,
        bbox=dict(facecolor='green', alpha=0.5),
        transform=ax.transAxes
        )  
    ax.set_title(colnams[0], loc='center', fontsize=12)   
    fig.tight_layout()
    lines, labels = fig.axes[0].get_legend_handles_labels()
    ax.legend(
        lines, labels, loc = 'lower left', ncol=5,
        # bbox_to_anchor=(0, 0.202),
        fontsize=12)
    # plt.legend()
    plt.show()


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


def y_fmt(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9 ]
    suffix  = ["G", "M", "k", "" , "m" , "u", "n"  ]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >=d:
            val = y/float(d)
            signf = len(str(val).split(".")[1])
            if signf == 0:
                # return '{val:d} {suffix}'.format(val=int(val), suffix=suffix[i])
                return '{val:d}'.format(val=int(val), suffix=suffix[i])
            else:
                if signf == 1:
                    # print (val, signf)
                    if str(val).split(".")[1] == "0":
                    #    return '{val:d} {suffix}'.format(val=int(round(val)), suffix=suffix[i])
                        return '{val:d}'.format(val=int(round(val)), suffix=suffix[i]) 
                tx = "{"+"val:.{signf}f".format(signf = signf) +"} {suffix}"
                return tx.format(val=val, suffix=suffix[i])
                #return y
    return y


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
    seds = [float(i[124:134]) for i in content[9:]]
    latq = [float(i[184:194]) for i in content[9:]] 
    sub_df = pd.DataFrame(
        np.column_stack([subs, mons, preps, sws, latq, surqs, ets, percs, gwqs, seds]),
        columns=["subs","mons", "precip", "sw", "latq", "surq", "et", "perco", "gwq", "sed"])

    # conv_types = {'hru':str, 'sub':int, 'mon':float, 'area_km2':float, 'irr_mm':float}
    # hru_df = hru_df.astype(conv_types)
    sub_df = sub_df.loc[sub_df['mons'] < 13]
    sub_df['mons'] = sub_df['mons'].astype(int)
    sub_df['subs'] = sub_df['subs'].astype(int)
    return sub_df


def read_output_sed(wd):
    with open(os.path.join(wd, 'output.sed'), 'r') as f:
        content = f.readlines()
    subs = [int(i[5:10]) for i in content[1:]]
    mons = [float(i[19:25]) for i in content[1:]]
    seds = [float(i[49:61]) for i in content[1:]]

    sed_df = pd.DataFrame(
        np.column_stack([subs, mons, seds]),
        columns=["subs","mons", "sed"])

    # conv_types = {'hru':str, 'sub':int, 'mon':float, 'area_km2':float, 'irr_mm':float}
    # hru_df = hru_df.astype(conv_types)
    sed_df = sed_df.loc[sed_df['mons'] < 13]
    # sed_df['mons'] = sed_df['mons'].astype(int)
    sed_df['subs'] = sed_df['subs'].astype(int)
    return sed_df


def read_output_rsv(wd):
    with open(os.path.join(wd, 'output.rsv'), 'r') as f:
        content = f.readlines()
    subs = [int(i[5:14]) for i in content[9:]]
    mons = [float(i[14:19]) for i in content[9:]]
    flow = [float(i[43:55]) for i in content[9:]]
    seds = [float(i[103:115]) for i in content[9:]]

    sed_df = pd.DataFrame(
        np.column_stack([subs, mons, flow, seds]),
        columns=["subs","mons", "flow", "sed"])

    # conv_types = {'hru':str, 'sub':int, 'mon':float, 'area_km2':float, 'irr_mm':float}
    # hru_df = hru_df.astype(conv_types)
    sed_df = sed_df.loc[sed_df['mons'] < 13]
    # sed_df['mons'] = sed_df['mons'].astype(int)
    sed_df['subs'] = sed_df['subs'].astype(int)
    return sed_df


def read_output_rch(wd):
    with open(os.path.join(wd, 'output.rch'), 'r') as f:
        content = f.readlines()
    subs = [int(i[5:10]) for i in content[9:]]
    mons = [float(i[19:25]) for i in content[9:]]
    flow = [float(i[49:61]) for i in content[9:]]
    seds = [float(i[97:109]) for i in content[9:]]

    sed_df = pd.DataFrame(
        np.column_stack([subs, mons, flow, seds]),
        columns=["subs","mons", "flow", "sed"])

    # conv_types = {'hru':str, 'sub':int, 'mon':float, 'area_km2':float, 'irr_mm':float}
    # hru_df = hru_df.astype(conv_types)
    sed_df = sed_df.loc[sed_df['mons'] < 13]
    # sed_df['mons'] = sed_df['mons'].astype(int)
    sed_df['subs'] = sed_df['subs'].astype(int)
    return sed_df


def phi_progress_plot(filename):
    rec_file = filename[:-3] + 'rec' 
    with open(rec_file, "r") as f:
        model_calls = []
        phis = []
        for line in f.readlines():
            if line.strip().startswith("Model calls so far"):
                model_calls.append(int(line.replace('\n', '').split()[5]))
            if line.strip().startswith("Starting phi for this iteration"):
                phis.append(float(line.replace('\n', '').split()[6]))
    
    df = pd.DataFrame({'Model Runs': model_calls, 'Phi': phis})
    df = df.set_index('Model Runs')
    df.plot(figsize=(5,5), grid=True)

def plot_tseries_ensembles(
                    pst, pr_oe, pt_oe, width=10, height=4, dot=True,
#                     onames=["hds","sfr"]
                    ):
    # pst.try_parse_name_metadata()
    # get the observation data from the control file and select 
    obs = pst.observation_data.copy()
    obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-6:])
    obs['time'] = time_col
#     # onames provided in oname argument
#     obs = obs.loc[obs.oname.apply(lambda x: x in onames)]
    # only non-zero observations
#     obs = obs.loc[obs.obgnme.apply(lambda x: x in pst.nnz_obs_groups),:]
    # make a plot
    ogs = obs.obgnme.unique()
    fig,axes = plt.subplots(len(ogs),1,figsize=(width,height*len(ogs)))
    ogs.sort()
    # for each observation group (i.e. timeseries)
    for ax,og in zip(axes,ogs):
        # get values for x axis
        oobs = obs.loc[obs.obgnme==og,:].copy()
        oobs.loc[:,"time"] = oobs.loc[:,"time"].astype(str)
#         oobs.sort_values(by="time",inplace=True)
        tvals = oobs.time.values
        onames = oobs.obsnme.values
        if dot is True:
            # plot prior
            [ax.scatter(tvals,pr_oe.loc[i,onames].values,color="gray",s=30, alpha=0.5) for i in pr_oe.index]
            # plot posterior
            [ax.scatter(tvals,pt_oe.loc[i,onames].values,color='b',s=30,alpha=0.2) for i in pt_oe.index]
            # plot measured+noise 
            oobs = oobs.loc[oobs.weight>0,:]
            tvals = oobs.time.values
            onames = oobs.obsnme.values
            ax.scatter(oobs.time,oobs.obsval,color='red',s=30).set_facecolor("none")
        if dot is False:
            # plot prior
            [ax.plot(tvals,pr_oe.loc[i,onames].values,"0.5",lw=0.5,alpha=0.5) for i in pr_oe.index]
            # plot posterior
            [ax.plot(tvals,pt_oe.loc[i,onames].values,"b",lw=0.5,alpha=0.5) for i in pt_oe.index]
            # plot measured+noise 
            oobs = oobs.loc[oobs.weight>0,:]
            tvals = oobs.time.values
            onames = oobs.obsnme.values
            ax.plot(oobs.time,oobs.obsval,"r-",lw=2)
        ax.tick_params(axis='x', labelrotation=90)
        ax.margins(x=0.01)
        ax.set_title(og,loc="left")
    # fig.tight_layout()
    plt.show()


def plot_prior_posterior_par_hist(prior_df, post_df, sel_pars, width=7, height=5, ncols=3):
    nrows = math.ceil(len(sel_pars)/ncols)
    fig, axes = plt.subplots(figsize=(width, height), nrows=nrows, ncols=ncols)
    ax1 = fig.add_subplot(111, frameon=False)
    ax1 = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, ax in enumerate(axes.flat):
        if i<len(sel_pars):
            colnam = sel_pars['parnme'].tolist()[i]
            ax.hist(prior_df.loc[:, colnam].values,
                    bins=np.linspace(
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parlbnd'].values[0], 
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parubnd'].values[0], 20),
                    color = "gray", alpha=0.5, density=True,
                    label="Prior"
            )
            y, x, _ = ax.hist(post_df.loc[:, colnam].values,
                    bins=np.linspace(
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parlbnd'].values[0], 
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parubnd'].values[0], 20), 
                     alpha=0.5, density=True, label="Posterior"
            )
            ax.set_ylabel(colnam)
            ax.set_yticks([])
    plt.xlabel("Parameter range")
    plt.show()
