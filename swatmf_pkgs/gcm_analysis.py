# %%
import pandas as pd
import numpy as np
import os
import glob
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def get_weather_folder_lists(wd):
    os.chdir(wd)
    wt_fds = [name for name in os.listdir(".") if os.path.isdir(name)]
    full_paths = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]
    return wt_fds, full_paths


def cvt_gcm_pcp(wd):
    wt_fds, full_paths = get_weather_folder_lists(wd)
    for p in full_paths:
        os.chdir(p)
        print("Folder changed to {}".format(p))
        inf = [f for f in glob.glob("*.csv")][0]
        print("{} file found ...".format(inf))
        df =  pd.read_csv(os.path.join(p, inf), index_col=0, parse_dates=True)
        tot_pts = len(df.columns)
        print("  Converting {} to 'pcp1.pcp' file ... processing".format(inf))

        for i in tqdm(range(tot_pts)):
            df.iloc[:, i] = df.iloc[:, i].map(lambda x: '{:05.1f}'.format(x))
        df.insert(loc=0, column='year', value=df.index.year)
        df.insert(loc=1, column='jday', value=df.index.dayofyear.map(lambda x: '{:03d}'.format(x)))
        dff = df.T.astype(str).apply(''.join)
        with open(os.path.join(p, 'pcp1.pcp'), 'w') as f:
            for i in range(4):
                f.write('\n')
            dff.to_csv(f, line_terminator='\n', index=False, header=None)
        print("  Converting GCM format file to 'pcp1.pcp' file ... passed")


def read_gcm(file_path, subs):
    df = pd.read_csv(
        file_path, index_col=0, parse_dates=True)
    df.columns = ['sub_{}'.format(i+1) for i in range(len(df.columns))]
    cols = ['sub_{}'.format(i) for i in subs]
    df = df[cols]
    return df


def read_pcp(pcp_path, loc_num):
    with open(pcp_path, 'r') as f:
        content = f.readlines()
    doy = [int(i[:7]) for i in content[4:]]
    date = pd.to_datetime(doy, format='%Y%j')
    df = pd.DataFrame(index=date)
    start_num = 7
    for i in tqdm(range(loc_num)):
        pp = [float(i[start_num:start_num+5]) for i in content[4:]]
        start_num += 5
        df["sub_{}".format(i+1)] = pp
    return df

# %%
working_path = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\weather_inputs"
wt_fds, full_paths = get_weather_folder_lists(working_path)

#%% Create GCMs in a dataframe
sub = 137
dff =  pd.DataFrame()
for i, name in zip(full_paths, wt_fds):
    os.chdir(i)
    inf = [f for f in glob.glob("*.csv")][0]
    df = read_gcm(inf, [sub])
    df.rename(columns = {'sub_137':'{}'.format(name)}, inplace = True)
    df = df['1/1/2000':'12/31/2050']
    dff = pd.concat([dff, df], axis=1)

#%% Create BASE in a dataframe
base = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\models\\base\\pcp1.pcp"
base_df = read_pcp(base, 257)
base_dff = base_df['sub_{}'.format(sub)]
base_dff = base_dff.rename('base')
base_dff = base_dff['1/1/2000':'12/31/2019']
base_dff

#%% Get Total dataframe
tot_df = pd.concat([base_dff, dff], axis=1)
tot_df = tot_df.resample('M').sum()
tot_df

#%%
abase_df = base_dff.resample('A').sum()
adf =dff.resample('A').sum()
atot_df = pd.concat([abase_df, adf], axis=1)
atot_df

#%% get month plot 
mon_df = tot_df.groupby(tot_df.index.month).mean()
mon_df

#%% get min max
gcsms_mon = mon_df.iloc[:, 1:]
s245_min = gcsms_mon.iloc[:, :4].min(axis = 1)
s245_max = gcsms_mon.iloc[:, :4].max(axis = 1)
s585_min = gcsms_mon.iloc[:, 4:].min(axis = 1)
s585_max = gcsms_mon.iloc[:, 4:].max(axis = 1)



#%% plot month
# 
marker = itertools.cycle((',', '+', '.', 'o', '*', 'v', '^', '<', '>',)) 

f, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax1 = f.add_subplot(111, frameon=False)
ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axes[1].plot(mon_df.index, mon_df['base'], label='base')
# ax.fill_between(base_dff.index, 0, base_dff.sub_137, alpha=0.3)
axes[1].fill_between(mon_df.index, s245_min, s245_max, alpha=0.3, label='ssp245 ')
axes[1].fill_between(mon_df.index, s585_min, s585_max, alpha=0.3, label='ssp585')

# axes[0].plot(mon_df.index, mon_df, marker=[',', '+', '.', 'o', '*'])
for i in range(len(mon_df.columns)):
    axes[0].plot(mon_df.index, mon_df.iloc[:, i], marker = next(marker))
axes[0].legend(mon_df.columns.tolist())
axes[1].legend()
# month_names = ['Jan','Feb','Mar','Apr','May','Jun',
#                'Jul','Aug','Sep','Oct','Nov','Dec']
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
for ax in axes:
    ax.set_xticklabels(month_names, rotation=90)
    ax.set_xticks(mon_df.index[::1])
ax1.set_ylabel('Monthly Rainfall Intensity $[mm/month]$')
plt.tight_layout()
plt.savefig(os.path.join(working_path, 'okvg_gcms_mon.png'), dpi=300, bbox_inches="tight")
plt.show()

#%%
# Boxplot
f, axes = plt.subplots(3, 4, figsize=(12,8), sharex=True)
ax1 = f.add_subplot(111, frameon=False)
ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']


# plot. Set color of marker edge
flierprops = dict(
                marker='o', markerfacecolor='#1f77b4', markersize=7,
                # linestyle='None',
                # markeredgecolor='none',
                alpha=0.3)
for i, ax in enumerate(axes.flat):
    df_m = tot_df.loc[tot_df.index.month==i+1]
    ax.boxplot(df_m.values, flierprops=flierprops)
    ax.set_xticks([i+1 for i in range(9)])
    ax.set_xticklabels(df_m.keys(), rotation=90)
    # ax.set_xticks(df_m.columns[::1])
    ax.set_title(
        month_names[i],
        horizontalalignment='left',
        x=0.02,
        y=0.85,
        )
ax1.set_ylabel('Monthly Rainfall Intensity $[mm/month]$')
plt.tight_layout()
plt.savefig(os.path.join(working_path, 'okvg_gcms_mon2.png'), dpi=300, bbox_inches="tight")
plt.show()

#%% get min max
agcm = atot_df.iloc[:, 1:]
s245_min = agcm.iloc[:, :4].min(axis = 1)
s245_max = agcm.iloc[:, :4].max(axis = 1)
s585_min = agcm.iloc[:, 4:].min(axis = 1)
s585_max = agcm.iloc[:, 4:].max(axis = 1)


ctot_df = atot_df['1/1/2000':'12/31/2019']

#%% plot month
# 
marker = itertools.cycle((',', '+', '.', 'o', '*', 'v', '^', '<', '>',)) 

f, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
ax1 = f.add_subplot(111, frameon=False)
ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
axes[1].plot(atot_df.index, atot_df['base'], label='base')
# ax.fill_between(base_dff.index, 0, base_dff.sub_137, alpha=0.3)
axes[1].fill_between(atot_df.index, s245_min, s245_max, alpha=0.3, label='ssp245')
axes[1].fill_between(atot_df.index, s585_min, s585_max, alpha=0.3, label='ssp585')

# axes[0].plot(mon_df.index, mon_df, marker=[',', '+', '.', 'o', '*'])
for i in range(len(ctot_df.columns)):
    axes[0].plot(ctot_df.index, ctot_df.iloc[:, i], marker = next(marker))
axes[0].legend(atot_df.columns.tolist(), ncol=2)
axes[1].legend()

month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
axes[0].margins( y=0.2)
ax1.set_ylabel('Annual Rainfall Intensity $[mm/year]$', labelpad=10)
plt.tight_layout()
plt.savefig(os.path.join(working_path, 'okvg_gcms_a.png'), dpi=300, bbox_inches="tight")
plt.show()



#%%
if __name__ == '__main__':
    wd = input("Where is folder?   ")
    print("location of folder is: "+ wd)
    print("We are going through all and converting them!")
    # gcm_df(f, [1,2,257])
    cvt_gcm_pcp(wd=wd)
    print('Converted!')
    # pcp_df(path_pcp)

# %%
