# %%
import pandas as pd
import numpy as np
import os
import glob
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
    df = df['1/1/2000':'12/31/2019']
    dff = pd.concat([dff, df], axis=1)

#%% Create BASE in a dataframe
base = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\models\\base\\pcp1.pcp"
base_df = read_pcp(base, 257)
base_dff = base_df['sub_{}'.format(sub)]
base_dff = base_dff['1/1/2000':'12/31/2019']

#%%
tot_df = pd.concat([base_dff, dff], axis=1)
tot_df

#%% get month plot 
mon_df = tot_df.groupby(tot_df.index.month).mean()

#%%
gcsms_mon = mon_df.iloc[:, 1:]
gcsms_min =  gcsms_mon.min(axis = 1)
gcsms_max = gcsms_mon.max(axis = 1)

#%%
f, ax = plt.subplots()
ax.plot(mon_df.index, mon_df['sub_137'])
# ax.fill_between(base_dff.index, 0, base_dff.sub_137, alpha=0.3)
ax.fill_between(mon_df.index, gcsms_min, gcsms_max, alpha=0.3)
ax.legend(mon_df.columns.tolist())
plt.show()

#%%
f, ax = plt.subplots()
ax.plot(mon_df.index, mon_df)

ax.legend(mon_df.columns.tolist())
plt.show()

#%%
df_jan = tot_df.loc[tot_df.index.month==1]
f, ax = plt.subplots()
df_jan.boxplot(column=df_jan.columns.tolist())
plt.xticks(rotation=90)
plt.show()



#%%
dff_jan = dff.loc[dff.index.month==1]
f, ax = plt.subplots()
dff_jan.boxplot(column=dff_jan.columns.tolist())
plt.xticks(rotation=90)
plt.show()

#%%
dff_jan.columns


#%%
base_dff = base_df.sub_137.groupby(base_df.index.month).mean()

#%%
f, ax = plt.subplots()
ax.plot(dff.index, dff, lw=3)
ax.plot(base_dff.index, base_dff)
# ax.fill_between(base_dff.index, 0, base_dff.sub_137, alpha=0.3)
# ax.fill_between(base_dff.index, 0, base_dff.sub_240, alpha=0.3)
plt.show()



#%%
mir_df = read_gcm(file_path, [137, 240])
mir_df
base = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\models\\base\\pcp1.pcp"
base_df = read_pcp(base, 257)

#%%
base_df = base_df[['sub_137', 'sub_240']]
base_dff  = base_df.groupby(base_df.index.month).mean()
f, ax = plt.subplots()
ax.plot(base_dff.index, base_dff, lw=3)
ax.fill_between(base_dff.index, 0, base_dff.sub_137, alpha=0.3)
ax.fill_between(base_dff.index, 0, base_dff.sub_240, alpha=0.3)
plt.show()

#%%

f, ax = plt.subplots()
ax.boxplot(
    [base_df['sub_137'].loc[base_df.index.month==1],
    base_df['sub_240'].loc[base_df.index.month==1]],
    )
plt.show()



#%%
f, ax = plt.subplots()
# ax.set(yscale="log")
sns.kdeplot(base_df['sub_137'], shade=True, label='b')
sns.kdeplot(base_df['sub_240'], shade=True, label='a')
ax.legend()
plt.show()

#%%
f, ax = plt.subplots()
sort = np.sort(base_df['sub_137'])[::-1]
exceedence = np.arange(1.,len(sort)+1) / len(sort)
ax.plot(exceedence*100, sort)
ax.set(yscale="log")
plt.show()
# %%
# file_path = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\weather_inputs\\ssp245-miroc6\\pcp1.pcp"
# # subs = [1, 2, 257]
# # subs_idx = [0] + subs
# # df = pd.read_csv(
# #     file_path, index_col=0, usecols=subs_idx,
# #     names=['idx'] + ['sub_{}'.format(i) for i in subs], parse_dates=True)
# with open(file_path, 'r') as f:
#     content = f.readlines()
# doy = [int(i[:7]) for i in content[4:]]
# date = pd.to_datetime(doy, format='%Y%j')

# loc_num = 257

# df = pd.DataFrame(index=date)
# start_num = 7
# for i in tqdm(range(loc_num)):
#     pp = [float(i[start_num:start_num+5]) for i in content[4:]]
#     start_num += 5
#     df["sub_{}".format(i+1)] = pp
# print(df)
# df.to_csv(os.path.join("D:\\", 'test.txt'))


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
