
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from hydroeval import evaluator, nse, rmse, pbias
import numpy as np
import math
import matplotlib.dates as mdates

from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from swatmf import handler, objfns
import pyemu


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
                    sep=r'\s+',
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
                    sep=r'\s+',
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
                sep=r'\s+',
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
                        sep=r'\s+',
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
                        sep=r'\s+',
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
                        sep=r'\s+',
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        "MODFLOW/" + mfobd_file,
                        sep=r'\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")

    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    output_wt = pd.read_csv(
                        "MODFLOW/apexmf_out_MF_obs",
                        sep=r'\s+',
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




def get_rels_objs(wd, pst_file, iter_idx=None, opt_idx=None):
    pst = pyemu.Pst(os.path.join(wd, pst_file))
    if iter_idx is None:
        iter_idx = pst.control_data.noptmax
    if opt_idx is None:
        opt_idx = -1
    # load observation data
    obs = pst.observation_data.copy()
    pst_nam = pst_file[:-4]
    # load posterior simulation
    pt_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"{0}.{1}.obs.csv".format(pst_nam, iter_idx)))

    pt_ut = pt_oe.loc[opt_idx].T
    opt_df = pd.DataFrame()
    opt_df = pd.concat([pt_ut, obs], axis=1)
    sims = opt_df.iloc[:, 0].tolist()
    obds = opt_df.iloc[:, 2].tolist()
    pbias = objfns.pbias(obds, sims)
    ns = objfns.nashsutcliffe(obds, sims)
    rsq = objfns.rsquared(obds, sims)
    rmse = objfns.rmse(obds, sims)
    mse = objfns.mse(obds, sims)
    pcc = objfns.correlationcoefficient(obds, sims)
    return ns, pbias, rsq, rmse, mse, pcc


def get_rels_objs_new(df, obgnme=None):
    if obgnme is not None:
        df = df.loc[df["obgnme"]==obgnme]
    sims = df.loc[:, "best_rel"].tolist()
    obds = df.loc[:, "obd"].tolist()
    pbias = objfns.pbias(obds, sims)
    ns = objfns.nashsutcliffe(obds, sims)
    rsq = objfns.rsquared(obds, sims)
    rmse = objfns.rmse(obds, sims)
    mse = objfns.mse(obds, sims)
    pcc = objfns.correlationcoefficient(obds, sims)
    return ns, pbias, rsq, rmse, mse, pcc





def get_rels_cal_val_objs(wd, pst_file, iter_idx=None, opt_idx=None, calval=None):
    pst = pyemu.Pst(os.path.join(wd, pst_file))
    if iter_idx is None:
        iter_idx = pst.control_data.noptmax
    if opt_idx is None:
        opt_idx = -1
    # load observation data
    obs = pst.observation_data.copy()
    pst_nam = pst_file[:-4]
    # load posterior simulation
    pt_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"{0}.{1}.obs.csv".format(pst_nam, iter_idx)))

    pt_ut = pt_oe.loc[opt_idx].T
    opt_df = pd.DataFrame()
    opt_df = pd.concat([pt_ut, obs], axis=1)

    if calval is None:
        calval = "cal"
    opt_df = opt_df.loc[opt_df["obgnme"]==calval]
    sims = opt_df.iloc[:, 0].tolist()
    obds = opt_df.iloc[:, 2].tolist()
    pbias = objfns.pbias(obds, sims)
    ns = objfns.nashsutcliffe(obds, sims)
    rsq = objfns.rsquared(obds, sims)
    rmse = objfns.rmse(obds, sims)
    mse = objfns.mse(obds, sims)
    return ns, pbias, rsq, rmse

def get_p_factor(pst, pt_oe, perc_obd_nz=None, cal_val=False):
    """calculate p-factor

    :param pst: pst object
    :type pst: class
    :param pt_oe: posterior ensamble
    :type pt_oe: dataframe
    :param perc_obd_nz: percentage of observation noise, defaults to None
    :type perc_obd_nz: real, optional
    :param cal_val: option to separate calibration and validation, defaults to False
    :type cal_val: bool, optional
    :return: p-factor value
    :rtype: real
    """
    obs = pst.observation_data.copy()
    if perc_obd_nz is None:
        perc_obd_nz=10
    perc = perc_obd_nz*0.01
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-8:])
    obs['time'] = time_col
    obs['time'] = pd.to_datetime(obs['time'])    
    df = pd.DataFrame(
        {'date':obs['time'],
        'obd':obs["obsval"],
        'weight':obs["weight"],
        'obgnme':obs["obgnme"],
        'pt_min': pt_oe.min(),
        'pt_max': pt_oe.max(),
        }
        )
    if cal_val is True:
        pfactors = []
        for i in ["cal", "val"]:
            cvdf = df.loc[df["obgnme"]==i]
            conditions = [
                ((cvdf.obd+(cvdf.obd*perc)) > cvdf.pt_min) & 
                ((cvdf.obd-(cvdf.obd*perc)) < cvdf.pt_max)
                    ]
            cvdf['pfactor'] = np.select(
                conditions, [1], default=0
                )
            pfactor = cvdf.loc[:, 'pfactor'].value_counts()[1] / len(cvdf.loc[:, 'pfactor'])
            pfactors.append(pfactor)
        print(pfactors)
        return pfactors
    else:
        conditions = [
            ((df.obd+(df.obd*perc)) > df.pt_min) & 
            ((df.obd-(df.obd*perc)) < df.pt_max)
                ]
        df['pfactor'] = np.select(
            conditions, [1], default=0
            )
        pfactor = df.loc[:, 'pfactor'].value_counts()[1] / len(df.loc[:, 'pfactor'])
        print(pfactor)
        df.to_csv('testpfactor.csv')
        return pfactor
    

def get_d_factor(pst, pt_oe, cal_val=False):
    obs = pst.observation_data.copy()
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-8:])
    obs['time'] = time_col
    obs['time'] = pd.to_datetime(obs['time'])    
    df = pd.DataFrame(
        {'date':obs['time'],
        'obd':obs["obsval"],
        'weight':obs["weight"],
        'obgnme':obs["obgnme"],
        'pt_min': pt_oe.min(),
        'pt_max': pt_oe.max(),
        }
        )
    if cal_val is True:
        dfactors = []
        for i in ["cal", "val"]:
            cvdf = df.loc[df["obgnme"]==i]
            std_obd = np.std(cvdf['obd'])
            dist_pts = (cvdf['pt_max'] - cvdf['pt_min']).mean()
            dfactor = dist_pts/std_obd
            dfactors.append(dfactor)
        print(dfactors)
        return dfactors
    else:
        std_obd = np.std(df['obd'])
        dist_pts = (df['pt_max'] - df['pt_min']).mean()
        dfactor = dist_pts/std_obd
        print(dfactor)
        return dfactor


def create_rels_objs(wd, pst_file, iter_idx):
    pst = pyemu.Pst(os.path.join(wd, pst_file))
    # load observation data
    # obs = pst.observation_data.copy()
    pst_nam = pst_file[:-4]
    # load posterior simulation
    pt_oe = pyemu.ObservationEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"{0}.{1}.obs.csv".format(pst_nam, iter_idx)))
    pt_par = pyemu.ParameterEnsemble.from_csv(
        pst=pst,filename=os.path.join(wd,"{0}.{1}.par.csv".format(pst_nam, iter_idx)))
    pt_oe_df = pd.DataFrame(pt_oe, index=pt_oe.index, columns=pt_oe.columns)
    pt_par_df = pd.DataFrame(pt_par, index=pt_par.index, columns=pt_par.columns)
    nss = []
    pbiass = []
    rsqs = []
    rmses = []
    mses = []
    pccs = []
    # for i in range(np.shape(pt_oe)[0]):
    for i in pt_oe.index:
        ns, pbias, rsq, rmse, mse, pcc = get_rels_objs(wd, pst_file, iter_idx=iter_idx, opt_idx=i)
        nss.append(ns)
        pbiass.append(pbias)
        rsqs.append(rsq)
        rmses.append(rmse)
        mses.append(mse)
        pccs.append(pcc)
    objs_df = pd.DataFrame(
        {
            "ns": nss, "pbias": pbiass, "rsq": rsqs, "rmse": rmses,
            "mse": mses, "pcc":pccs
            },
        index=pt_oe.index)
    pt_oe_df = pd.concat([pt_oe_df, objs_df], axis=1)
    pt_par_df = pd.concat([pt_par_df, objs_df], axis=1)
    pt_oe_df.to_csv(os.path.join(wd, "{0}.{1}.obs.objs.csv".format(pst_nam, iter_idx)))
    pt_par_df.to_csv(os.path.join(wd, "{0}.{1}.par.objs.csv".format(pst_nam, iter_idx)))

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
                        sep=r'\s+',
                        skiprows = 2,
                        usecols = [3, 4],
                        index_col = 0,
                        names = ["grid_id", "mf_elev"],)
    mfobd_df = pd.read_csv(
                        "MODFLOW/" + mfobd_file,
                        sep=r'\s+',
                        index_col=0,
                        header=0,
                        parse_dates=True,
                        na_values=[-999, ""],
                        delimiter="\t")
    grid_id_lst = mf_obs.index.astype(str).values.tolist()
    # read simulated water elevation
    output_wt = pd.read_csv(
                        "MODFLOW/apexmf_out_MF_obs",
                        sep=r'\s+',
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

    
def get_par_offset(pst):
    pars = pst.parameter_data.copy()
    pars = pars.loc[:, ["parnme", "offset"]]
    return pars

def plot_prior_posterior_par_hist(
        wd,
        pst, prior_df, post_df, sel_pars, 
        width=7, height=5, ncols=3, bestcand=None, parobj_file=None):
    nrows = math.ceil(len(sel_pars)/ncols)
    pars_info = get_par_offset(pst)
    fig, axes = plt.subplots(figsize=(width, height), nrows=nrows, ncols=ncols)
    ax1 = fig.add_subplot(111, frameon=False)
    ax1 = plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i, ax in enumerate(axes.flat):
        if i<len(sel_pars):
            colnam = sel_pars['parnme'].tolist()[i]
            offset = pars_info.loc[colnam, "offset"]
            
            ax.hist(prior_df.loc[:, colnam].values + offset,
                    bins=np.linspace(
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parlbnd'].values[0]+ offset, 
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parubnd'].values[0]+ offset, 20),
                    color = "gray", alpha=0.5, density=False,
                    label="Prior"
            )
            y, x, _ = ax.hist(post_df.loc[:, colnam].values + offset,
                    bins=np.linspace(
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parlbnd'].values[0]+ offset, 
                        sel_pars.loc[sel_pars["parnme"]==colnam, 'parubnd'].values[0]+ offset, 20), 
                        alpha=0.5, density=False, label="Posterior"
            )
            ax.set_title(colnam, fontsize=9, ha='left', x=0.07, y=0.93, backgroundcolor='white')
            # ax.set_yticks([])
            if parobj_file is not None:
                po_df = pd.read_csv(os.path.join(wd, parobj_file))
                x = po_df.loc[po_df["real_name"]==bestcand, colnam].values + offset
                ax.axvline(x=x, color='r', linestyle="--", alpha=0.5)
        else:
            ax.axis('off')
            ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # ax.set_xticks(ax.get_xticks()[::1])
        
        ax.tick_params(axis='x', labelsize=8)       
    plt.ylabel(r"Frequency", fontsize=10)
    plt.xlabel(r"Parameter relative change (%)", fontsize=10)
    plt.tight_layout()
    plt.savefig('par_hist.png', bbox_inches='tight', dpi=300)
    plt.show()
    print(os.getcwd())

# scratches for QSWATMOD
# data comes from hanlder module and SWATMFout class

def plot_observed_data(ax, df3, obd_col):
    size = 10
    ax.plot(
        df3.index.values, df3[obd_col].values, c='m', lw=1.5, alpha=0.5,
        label="Observed", zorder=3
    )
    # ax.scatter(
    #     df3.index.values, df3[obd_col].values, c='m', lw=1, alpha=0.5, s=size, marker='x',
    #     label="Observed", zorder=3
    # )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%Y'))
    if len(df3[obd_col]) > 1:
        calculate_metrics(ax, df3, obd_col)
    else:
        display_no_data_message(ax)

def plot_stf_sim_obd(ax, stf_obd_df, obd_col):
    ax.plot(stf_obd_df.index.values, stf_obd_df.stf_sim.values, c='limegreen', lw=1, label="Simulated")
    plot_observed_data(ax, stf_obd_df, obd_col)
    # except Exception as e:
    #     handle_exception(ax, str(e))

def plot_stf_sim(ax, stf_df):
    try:
        ax.plot(stf_df.index.values, stf_df.stf_sim.values, c='limegreen', lw=1, label="Simulated")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%Y'))
    except Exception as e:
        handle_exception(ax, str(e)) 


# gw
def plot_gw_sim(ax, df, grid_id):
    ax.plot(df.index.values, df[str(grid_id)].values, c='skyblue', lw=1, label="Simulated")


def plot_gw_sim_obd(ax, sim_df, grid_id, obd_df, obd_col):
    df =  pd.concat([sim_df.loc[:, str(grid_id)], obd_df.loc[:, obd_col]], axis=1).dropna()
    ax.plot(df.index.values, df[str(grid_id)].values, c='skyblue', lw=1, label="Simulated")
    ax.plot(
        df.index.values, df[obd_col].values, c='m', lw=1.5, alpha=0.5,
        label="Observed", zorder=3
    )
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d\n%Y'))
    if len(df[obd_col]) > 1:
        calculate_metrics_gw(ax, df, grid_id, obd_col)
    else:
        display_no_data_message(ax)  


# NOTE: metrics =======================================================================================
def calculate_metrics(ax, df3, obd_col):
    r_squared = ((sum((df3[obd_col] - df3[obd_col].mean()) * (df3.stf_sim - df3.stf_sim.mean())))**2) / (
            (sum((df3[obd_col] - df3[obd_col].mean())**2) * (sum((df3.stf_sim - df3.stf_sim.mean())**2)))
    )
    dNS = 1 - (sum((df3.stf_sim - df3[obd_col])**2) / sum((df3[obd_col] - (df3[obd_col]).mean())**2))
    PBIAS = 100 * (sum(df3[obd_col] - df3.stf_sim) / sum(df3[obd_col]))
    display_metrics(ax, dNS, r_squared, PBIAS)

def calculate_metrics_gw(ax, df3, grid_id, obd_col):
    r_squared = (
            ((sum((df3.loc[:, obd_col] - df3.loc[:, obd_col].mean()) * 
            (df3.loc[:, grid_id]  - df3.loc[:, grid_id].mean())))**2) / 
            ((sum((df3.loc[:, obd_col] - df3.loc[:, obd_col].mean())**2) * 
            (sum((df3.loc[:, grid_id]  - df3.loc[:, grid_id].mean())**2)))
            ))
    dNS = 1 - (sum((df3.loc[:, grid_id]  - df3.loc[:, obd_col] )**2) / sum((df3.loc[:, obd_col] - (df3.loc[:, obd_col]).mean())**2))
    PBIAS = 100 * (sum(df3.loc[:, obd_col] - df3.loc[:, grid_id] ) / sum(df3.loc[:, obd_col] ))
    display_metrics(ax, dNS, r_squared, PBIAS)

def display_metrics(ax, dNS, r_squared, PBIAS):
    ax.text(
        .01, 0.90, f'Nash-Sutcliffe: {dNS:.4f}',
        fontsize=8, horizontalalignment='left', color='limegreen', transform=ax.transAxes
    )
    ax.text(
        .01, 0.80, f'$R^2$: {r_squared:.4f}',
        fontsize=8, horizontalalignment='left', color='limegreen', transform=ax.transAxes
    )
    ax.text(
        .99, 0.90, f'PBIAS: {PBIAS:.4f}',
        fontsize=8, horizontalalignment='right', color='limegreen', transform=ax.transAxes
    )

def display_no_data_message(ax):
    ax.text(
        .01, .95, 'Nash-Sutcliffe: ---',
        fontsize=8, horizontalalignment='left', transform=ax.transAxes
    )
    ax.text(
        .01, 0.90, '$R^2$: ---',
        fontsize=8, horizontalalignment='left', color='limegreen', transform=ax.transAxes
    )
    ax.text(
        .99, 0.95, 'PBIAS: ---',
        fontsize=8, horizontalalignment='right', color='limegreen', transform=ax.transAxes
    )

def handle_exception(ax, exception_message):
    ax.text(
        .5, .5, exception_message,
        fontsize=12, horizontalalignment='center', weight='extra bold', color='y', transform=ax.transAxes
    )

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)


# std plot
def std_plot(axes, dff, viz_ts, widthExg=1, cutcolor='k'):
    # fig, axes = plt.subplots(
    #     nrows=4, figsize=(14, 7), sharex=True,
    #     gridspec_kw={
    #                 'height_ratios': [0.2, 0.2, 0.4, 0.2],
    #                 'hspace': 0.1
    #                 })
    # plt.subplots_adjust(left=0.06, right=0.98, top=0.83, bottom=0.05)
    width = -20
    dff = dff.resample("M").mean()
    # == Precipitation ============================================================
    axes[0].bar(
        dff.index, dff.prec, width * widthExg,
        # edgecolor = 'w',
        align='edge',
        # linewidth = 0.1,
        color='slateblue')
    axes[0].set_ylim((dff.prec.max() + dff.prec.max() * 0.1), 0)
    axes[0].xaxis.tick_top()
    axes[0].spines['bottom'].set_visible(False)
    axes[0].tick_params(axis='both', labelsize=8)

    # if viz_ts == "MA":
    #     axes[0].set_title('Water Balance - Monthly Average [mm]', fontsize=12, fontweight='semibold')
    #     dff = dff.resample("M").mean()
    # elif viz_ts == "AA":
    #     axes[0].set_title('Water Balance - Annual Average [mm]', fontsize=12, fontweight='semibold')
    # ttl = axes[0].title
    # ttl.set_position([0.5, 1.8])
    # == Soil Water ===============================================================
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].get_xaxis().set_visible(False)
    axes[1].bar(
        dff.index, dff.sw, width * widthExg,
        bottom=dff.gwq + dff.latq + dff.surq,
        # edgecolor = 'w',
        align='edge',
        # linewidth = 0.3,
        color='lightgreen')
    axes[1].set_ylim(
        (dff.gwq + dff.latq + dff.surq).max(),
        (dff.gwq + dff.latq + dff.surq + dff.sw).max()
        )
    axes[1].tick_params(axis='both', labelsize=8)
    # == Interaction ============================================================
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['bottom'].set_visible(False)
    axes[2].get_xaxis().set_visible(False)
    # gwq -> Groundwater discharge to stream
    axes[2].bar(
        dff.index, dff.gwq, width * widthExg,
        # edgecolor = 'w',
        align='edge',
        # linewidth = 0.3,
        color='darkgreen')
    # latq -> lateral flow to stream
    axes[2].bar(
        dff.index, dff.latq, width * widthExg,
        bottom=dff.gwq,
        # edgecolor='w',
        align='edge',
        # linewidth=0.3,
        color='forestgreen')
    # surq -> surface runoff to stream
    axes[2].bar(
        dff.index, dff.surq, width * widthExg,
        bottom=dff.latq + dff.gwq,
        # edgecolor='w',
        align='edge',
        # linewidth=0.3,
        color='limegreen')
    # Soil water
    axes[2].bar(
        dff.index, dff.sw, width * widthExg,
        bottom=dff.gwq + dff.latq + dff.surq,
        # edgecolor='w',
        align='edge',
        # linewidth=0.3,
        color='lightgreen')
    axes[2].axhline(y=0, xmin=0, xmax=1, lw=0.3, ls='--', c='grey')
    # swgw -> seepage to aquifer
    axes[2].bar(
        dff.index, dff.swgw*-1, width * widthExg,
        # bottom = df.latq,
        # edgecolor='w',
        align='edge',
        # linewidth=0.8,
        color='b')
    # perco -> recharge to aquifer
    axes[2].bar(
        dff.index, dff.perco*-1, width * widthExg,
        bottom=dff.swgw*-1,
        # edgecolor='w',
        align='edge',
        # linewidth=0.8,
        color='dodgerblue')
    # gw -> groundwater volume
    axes[2].bar(
        dff.index, dff.gw*-1, width * widthExg,
        bottom=(dff.perco*-1) + (dff.swgw*-1),
        # edgecolor='w',
        color=['skyblue'],
        align='edge',
        # linewidth=0.8
        )
    axes[2].set_ylim(
        -1*(dff.swgw + dff.perco).max(),
        (dff.gwq + dff.latq + dff.surq).max())
    axes[2].tick_params(axis='both', labelsize=8)
    axes[2].set_yticklabels([float(abs(x)) for x in axes[2].get_yticks()])
    # ===
    axes[3].bar(
        dff.index, dff.gw, width * widthExg,
        bottom=(dff.perco) + (dff.swgw),
        # edgecolor='w',
        color=['skyblue'],
        align='edge',
        # linewidth=0.8
        )
    # axes[3].set_yticklabels([int(abs(x)) for x in axes[3].get_yticks()])
    axes[3].set_ylim(
        ((dff.gw + dff.perco + dff.swgw).max()),
        ((dff.gw + dff.perco + dff.swgw).min())
        )
    axes[3].spines['top'].set_visible(False)
    axes[3].tick_params(axis='both', labelsize=8)
    # this is for a broken y-axis  ##################################
    d = .003  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    # if self.dlg.checkBox_darktheme.isChecked():
    #     cutcolor = 'w'
    # else:
    #     cutcolor = 'k'
    cutcolor = 'k'
    kwargs = dict(transform=axes[1].transAxes, color=cutcolor, clip_on=False)
    axes[1].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    axes[1].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=axes[2].transAxes)  # switch to the bottom axes
    axes[2].plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    axes[2].plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    axes[2].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    axes[2].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    kwargs.update(transform=axes[3].transAxes)  # switch to the bottom axes
    axes[3].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    axes[3].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    # if self.dlg.checkBox_std_legend.isChecked():
    names = (
        'Precipitation', 'Soil Water', 'Surface Runoff', 'Lateral Flow',
        'Groundwater Flow to Stream',
        'Seepage from Stream to Aquifer',
        'Deep Percolation to Aquifer',
        'Groundwater Volume')
    colors = (
        'slateblue', 'lightgreen', 'limegreen', 'forestgreen', 'darkgreen',
        'b',
        'dodgerblue',
        'skyblue')
    ps = []
    for c in colors:
        ps.append(
            Rectangle(
                (0, 0), 0.1, 0.1, fc=c,
                # ec = 'k',
                alpha=1))
    legend = axes[0].legend(
        ps, names,
        loc='upper left',
        # title="EXPLANATION",
        # ,handlelength = 3, handleheight = 1.5,
        edgecolor='none',
        fontsize=8,
        bbox_to_anchor=(-0.02, 1.8),
        # labelspacing = 1.5,
        ncol=4)
    legend._legend_box.align = "left"
    # legend text centered
    for t in legend.texts:
        t.set_multialignment('left')
    # plt.setp(legend.get_title(), fontweight='bold', fontsize=10)
    # plt.show()


def plot_sen_morris(df):
    df = df.loc[df.sen_mean_abs>1e-6,:]
    # df.loc[:,["sen_mean_abs","sen_std_dev"]].plot(kind="bar", figsize=(9,3), fontsize=12)
    #ax = plt.gca()
    #ax.set_ylim(1,ax.get_ylim()[1]*1.1)
    # plt.yscale('log');
    fig,ax = plt.subplots(1,1,figsize=(6,5))
    swat_df = df.loc[df["pargp"]=="swat"]
    hk_df = df.loc[df["pargp"]=="hk"]
    ss_df = df.loc[df["pargp"]=="ss"]
    sy_df = df.loc[df["pargp"]=="sy"]
    
    ax.scatter(swat_df.sen_mean_abs,swat_df.sen_std_dev,marker="^",s=80,c="r", alpha=0.5, label="swat")
    ax.scatter(hk_df.sen_mean_abs,hk_df.sen_std_dev,marker="s",s=80,c="b", alpha=0.5, label="hy")
    ax.scatter(ss_df.sen_mean_abs,ss_df.sen_std_dev,marker="o",s=80,c="g", alpha=0.5, label="ss")
    ax.scatter(sy_df.sen_mean_abs,sy_df.sen_std_dev,marker="x",s=80,c="k", alpha=0.5, label="sy")

    # tmp_df = tmp_df.iloc[:8]
    for x,y,n in zip(df.sen_mean_abs,df.sen_std_dev,df.index):
        if x > 1000000:
            ax.text(x,y,n, fontsize=12)
    mx = max(ax.get_xlim()[1],ax.get_ylim()[1])
    mn = min(ax.get_xlim()[0],ax.get_ylim()[0])
    ax.plot([mn,mx],[mn,mx],"k--", alpha=0.3)
    ax.set_ylim(mn,mx)
    ax.set_xlim(mn,mx)
    ax.grid()
    ax.set_ylabel(r"$\sigma$", fontsize=12)
    ax.set_xlabel(r"$\mu^*$", fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12, loc="lower right")
    plt.tight_layout()  
    plt.show()  

def get_pr_pt_df(pst, pr_oe, pt_oe, bestrel_idx=None):
    obs = pst.observation_data.copy()
    time_col = []
    for i in range(len(obs)):
        time_col.append(obs.iloc[i, 0][-8:])
    obs['time'] = time_col
    obs['time'] = pd.to_datetime(obs['time'])
    # print(pt_oe.loc["4"])
    if bestrel_idx is not None:
        df = pd.DataFrame(
            {'date':obs['time'],
            'obd':obs["obsval"],
            'pr_min': pr_oe.min(),
            'pr_max': pr_oe.max(),
            'pt_min': pt_oe.min(),
            'pt_max': pt_oe.max(),
            'best_rel': pt_oe.loc[str(bestrel_idx)],
            'obgnme': obs['obgnme'],
            }
            )
    else:
        df = pd.DataFrame(
            {'date':obs['time'],
            'obd':obs["obsval"],
            'pr_min': pr_oe.min(),
            'pr_max': pr_oe.max(),
            'pt_min': pt_oe.min(),
            'pt_max': pt_oe.max(),
            'obgnme': obs['obgnme'],
            }
            )
    df.set_index('date', inplace=True)
    return df


def plot_fill_between_ensembles(
        df, 
        width=12, height=4,
        caldates=None,
        valdates=None,
        size=None,
        pcp_df=None,
        bestrel_idx="best_rel",
        ):
    """plot time series of prior/posterior predictive uncertainties

    :param df: dataframe of prior/posterior created by get_pr_pt_df function
    :type df: dataframe
    :param width: plot width, defaults to 12
    :type width: int, optional
    :param height: plot height, defaults to 4
    :type height: int, optional
    :param caldates: calibration start and end dates, defaults to None
    :type caldates: list, optional
    :param valdates: validation start and end dates, defaults to None
    :type valdates: list, optional
    :param size: symbol size, defaults to None
    :type size: int, optional
    :param pcp_df: dataframe of precipitation, defaults to None
    :type pcp_df: dataframe, optional
    :param bestrel_idx: realization index, defaults to None
    :type bestrel_idx: string, optional
    """
    if size is None:
        size = 30
    fig, ax = plt.subplots(figsize=(width,height))
    # x_values = df.loc[:, "newtime"].values
    x_values = df.index.values
    if caldates is not None:
        caldf = df[caldates[0]:caldates[1]]
        valdf = df[valdates[0]:valdates[1]]
        ax.fill_between(
            df.index.values, df.loc[:, 'pr_min'].values, df.loc[:, 'pr_max'].values, 
            facecolor="0.5", alpha=0.4, label="Prior")
        ax.fill_between(
            caldf.index.values, caldf.loc[:, 'pt_min'].values, caldf.loc[:, 'pt_max'].values, 
            facecolor="g", alpha=0.4, label="Posterior")
        if bestrel_idx is not None:
            ax.plot(caldf.index.values, caldf.loc[:, bestrel_idx].values, c='g', lw=1, label="calibrated")
        ax.fill_between(
            valdf.index.values, valdf.loc[:, 'pt_min'].values, valdf.loc[:, 'pt_max'].values, 
            facecolor="m", alpha=0.4, label="Forecast")        
        ax.scatter(
            df.index.values, df.loc[:, 'obd'].values, 
            color='red',s=size, zorder=10, label="Observed").set_facecolor("none")
        if bestrel_idx is not None:
            ax.plot(valdf.index.values, valdf.loc[:, bestrel_idx].values, c='m', lw=1, label="validated")
    else:
        ax.fill_between(
            x_values, df.loc[:, 'pr_min'].values, df.loc[:, 'pr_max'].values, 
            facecolor="0.5", alpha=0.4, label="Prior")
        ax.fill_between(
            x_values, df.loc[:, 'pt_min'].values, df.loc[:, 'pt_max'].values, 
            facecolor="g", alpha=0.4, label="Posterior")
        if bestrel_idx is not None:
            ax.plot(
                x_values, df.loc[:, bestrel_idx].values, 
                c='b', lw=1, label="calibrated", zorder=3)
        ax.scatter(
            x_values, df.loc[:, 'obd'].values, 
            color='red',s=size, zorder=5, label="Observed", alpha=0.5).set_facecolor("none")
    if pcp_df is not None:
        # pcp_df.index.freq = None
        ax2=ax.twinx()
        ax2.bar(
            pcp_df.index, pcp_df.loc[:, "pcpmm"].values, label='Precipitation',
            width=20 ,
            color="blue", 
            align='center', 
            alpha=0.5
            )
        ax2.set_ylabel("Precipitation $(mm/month)$",fontsize=12)
        ax2.invert_yaxis()
        ax2.set_ylim(pcp_df.loc[:, "pcpmm"].max()*3, 0)
        # ax.set_ylabel("Stream Discharge $(m^3/day)$",fontsize=14)
        ax2.tick_params(axis='y', labelsize=12)
    ax.set_ylabel(r"Daily streamflow $(m^3/s)$", fontsize=12)
    ax.margins(0.01)
    ax.tick_params(axis='both', labelsize=12)
    # ax.set_ylim(0, df.max().max()*1.5)
    # ax.set_ylim(0, 800)
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    order = [0,1,2,3]
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    # fig.legend(
    #     [tlines[idx] for idx in order],[tlables[idx] for idx in order],
    #     fontsize=10,
    #     loc = 'lower center',
    #     bbox_to_anchor=(0.5, -0.08),
    #     ncols=7)

    tlables = labels
    tlines = lines

    fig.legend(
        [tlines[idx] for idx in order],[tlables[idx] for idx in order],
        fontsize=10,
        loc = 'lower center',
        bbox_to_anchor=(0.5, -0.08),
        ncols=7)


    # fig.legend(fontsize=12, loc="lower left")
    years = mdates.YearLocator()
    # print(years)
    yearsFmt = mdates.DateFormatter('%Y')  # add some space for the year label
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b') 
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.set_minor_formatter(monthsFmt)
    plt.setp(ax.xaxis.get_minorticklabels(), fontsize = 8, rotation=90)
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    # ax.set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep"]*7)
    ax.tick_params(axis='both', labelsize=8, rotation=0)
    ax.tick_params(axis = 'x', pad=20)

    plt.tight_layout()
    plt.savefig('cal_val.png', bbox_inches='tight', dpi=300)
    plt.show()


def single_plot_fdc_added(
                    df,
                    width=10, height=8, dot=True,
                    size=None, bstc=False,
                    orgsim=None
                    ):
    """plot flow exceedence

    :param df: dataframe created by get_pr_pt_df function
    :type df: dataframe
    :param width: figure width, defaults to 10
    :type width: int, optional
    :param height: figure hight, defaults to 8
    :type height: int, optional
    :param dot: scatter or line, defaults to True
    :type dot: bool, optional
    :param size: maker size, defaults to None
    :type size: int, optional
    :param bstcs: best candiates, defaults to None
    :type bstcs: list, optional
    :param orgsim: _description_, defaults to None
    :type orgsim: _type_, optional
    """
    if size is None:
        size = 30
    obs_d, obd_exd = convert_fdc_data(df.obd.values)
    pr_min_d, pr_min_exd = convert_fdc_data(df.pr_min.values)
    pr_max_d, pr_max_exd = convert_fdc_data(df.pr_max.values)
    pt_min_d, pt_min_exd = convert_fdc_data(df.pt_min.values)
    pt_max_d, pt_max_exd = convert_fdc_data(df.pt_max.values)

    fig, ax = plt.subplots(figsize=(width,height))
    ax.fill_between(pr_min_exd*100, pr_min_d, pr_max_d, interpolate=False, facecolor="0.5", alpha=0.4)
    ax.fill_between(pt_min_exd*100, pt_min_d, pt_max_d, interpolate=False, facecolor="b", alpha=0.4)
    ax.scatter(obd_exd*100, obs_d, color='red',s=size, zorder=10, label="Observed").set_facecolor("none")
    if orgsim is not None:
        orgsim = orgsim
        org_d, org_exd = convert_fdc_data(orgsim.iloc[:, 0].values)
        ax.plot(org_exd*100, org_d, c='limegreen', lw=2, label="Original")
    if bstc is True:
        # for bstc in bstcs:
        #     dd, eexd = convert_fdc_data(df.best_rel.values)
        #     ax.plot(eexd*100, dd, lw=2, label=bstc)
        # for bstc in bstcs:
        dd, eexd = convert_fdc_data(df.best_rel.values)
        ax.plot(eexd*100, dd, lw=2, label="best_rel")



    ax.set_yscale('log')
    ax.set_xlabel(r"Exceedence [%]", fontsize=12)
    ax.set_ylabel(r"Flow rate $[m^3/s]$", fontsize=12)
    ax.margins(0.01)
    ax.tick_params(axis='both', labelsize=12)
    plt.legend(fontsize=12, loc="lower left")
    plt.tight_layout()
    plt.savefig('fdc.png', bbox_inches='tight', dpi=300)
    plt.show()
    print(os.getcwd())  

    # return pr_oe_min


def convert_fdc_data(data):
    data = np.sort(data)[::-1]
    exd = np.arange(1.,len(data)+1) / len(data)
    return data, exd


def plot_each_obg(df, rel_idx):
    # df = handler.filter_candidates2()
    obgs = df.obgnme.unique()
    df = df[["obsval", "obgnme", "time", rel_idx]]

    fig, axes = plt.subplots(figsize=(5.5, 12), nrows=len(obgs))
    for obg, ax in zip(obgs, axes):
        dff = df.loc[df["obgnme"]==obg]
        obds = dff.obsval.values
        sims = dff.loc[:, rel_idx].values
        pbias = objfns.pbias(obds, sims)
        ns = objfns.nashsutcliffe(obds, sims)
        rsq = objfns.rsquared(obds, sims)
        rmse = objfns.rmse(obds, sims)
        mse = objfns.mse(obds, sims)
        pcc = objfns.correlationcoefficient(obds, sims)
        ax.text(
            0, 1,
            f'obg: {obg} - ns: {ns:.2f}, rmse: {rmse:.2f}, pbias: {pbias:.2f}, rsq: {rsq:.2f}',
            horizontalalignment='left',fontsize=10,
            bbox=dict(facecolor='m', alpha=0.5),
            transform=ax.transAxes
            )
        ax.scatter(dff.time, dff.obsval, c="r", s=3, alpha=0.2, zorder=3)
        ax.plot(dff.time, dff.loc[:, rel_idx])
    # plt.title(f"rel_name: {rel_idx}")
    fig.suptitle(f'rel{rel_idx}', fontsize=10, y=1)
    fig.tight_layout()
    plt.savefig(f'rel{rel_idx}.png', bbox_inches='tight', dpi=300)
    # plt.show()

def fdc(
        df, rel_nams, ncols=5,
        width=7, height=5, obgnme="sub03"
        ):
    nrows = math.ceil(len(rel_nams)/ncols)
    fig, axes = plt.subplots(
        figsize=(width, height), nrows=nrows, ncols=ncols,
        sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        if i<len(rel_nams):
            rel_idx = rel_nams[i]
            dff = df[["obsval", "obgnme", "time", rel_idx]]
            dff = dff.loc[dff["obgnme"]==obgnme]
            odd, oeexd = convert_fdc_data(dff.obsval.values)
            sdd, seexd = convert_fdc_data(dff.loc[:, rel_idx].values)
            ax.plot(seexd*100, sdd, lw=2, label="sim_sub03")
            ax.plot(oeexd*100, odd, lw=2, label="obd")
            ax.set_yscale('log')
            # ax.set_xlabel(r"Exceedence [%]", fontsize=12)
            # ax.set_ylabel(r"Flow rate $[m^3/s]$", fontsize=12)
            ax.margins(0.01)
            # ax.tick_params(axis='both', labelsize=12)
            # plt.legend(fontsize=12, loc="lower left")
            ax.text(
                1, 0.8, f'rel{rel_idx}', fontsize=10,
                horizontalalignment='right',
                transform=ax.transAxes)
        else:
            ax.axis('off')
            ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.savefig(f'fdc_{obgnme}.png', bbox_inches='tight', dpi=300)
    plt.show()



def single_fdc(df):
    # from get_stf_sim_obd
    fig, ax = plt.subplots()
    odd, oeexd = convert_fdc_data(df.iloc[:, 1].values)
    sdd, seexd = convert_fdc_data(df.iloc[:, 0].values)
    ax.plot(seexd*100, sdd, lw=2, label="sim")
    ax.plot(oeexd*100, odd, lw=2, label="obd")
    ax.set_yscale('log')
    # ax.set_xlabel(r"Exceedence [%]", fontsize=12)
    # ax.set_ylabel(r"Flow rate $[m^3/s]$", fontsize=12)
    ax.margins(0.01)
    # ax.tick_params(axis='both', labelsize=12)
    # plt.legend(fontsize=12, loc="lower left")
    # ax.text(
    #     1, 0.8, f'rel{rel_idx}', fontsize=10,
    #     horizontalalignment='right',
        # transform=ax.transAxes)
    plt.tight_layout()
    # plt.savefig(f'fdc_{obgnme}.png', bbox_inches='tight', dpi=300)
    plt.show()

