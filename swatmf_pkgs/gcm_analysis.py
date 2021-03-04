# %%
import pandas as pd
import numpy as np
import os
import glob
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import shutil


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



# def read_gcm_avg(file)
def read_pcp(pcp_path, nloc):
    with open(pcp_path, 'r') as f:
        content = f.readlines()
    doy = [int(i[:7]) for i in content[4:]]
    date = pd.to_datetime(doy, format='%Y%j')
    df = pd.DataFrame(index=date)
    start_num = 7
    for i in tqdm(range(nloc)):
        pp = [float(i[start_num:start_num+5]) for i in content[4:]]
        start_num += 5
        df["sub_{}".format(i+1)] = pp
    return df


def read_tmp(tmp_path, nloc):
    with open(tmp_path, 'r') as f:
        content = f.readlines()
    doy = [int(i[:7]) for i in content[4:]]
    date = pd.to_datetime(doy, format='%Y%j')
    df_max = pd.DataFrame(index=date)
    df_min = pd.DataFrame(index=date)
    max_start_num = 7
    min_start_num = 13
    for i in tqdm(range(nloc)):
        max_tmp = [float(i[max_start_num:max_start_num+5]) for i in content[4:]]
        max_start_num += 10
        df_max["max_sub_{}".format(i+1)] = max_tmp

        min_tmp = [float(i[min_start_num:min_start_num+5]) for i in content[4:]]
        min_start_num += 10
        df_min["min_sub_{}".format(i+1)] = min_tmp
    return df_max, df_min


def modify_tmp(weather_wd, weather_modified_wd, copy_files_fr_org=None):
    os.chdir(weather_wd)
    model_nams = [name for name in os.listdir(".") if os.path.isdir(name)]
    model_paths = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]

    for i, mp in tqdm(zip(model_nams, model_paths), total=len(model_nams)):
        new_weather_dir = os.path.join(weather_modified_wd, "{}".format(i))
        if not os.path.exists(new_weather_dir):
            os.mkdir(new_weather_dir)
        with open(os.path.join(mp, 'Tmp1.Tmp'), "r") as f:
            lines = f.readlines()
        with open(os.path.join(new_weather_dir, 'Tmp1.Tmp'), "w") as f:
            count = 0
            for line in lines[:4]:
                f.write(line.strip()+'\n')
            for line in lines[4:]:
                if int(line[:4]) < 2020:
                    f.write((line[:7]+('-99.0-99.0')*154).strip()+'\n')
                else:
                    f.write(line.strip()+'\n')
        if copy_files_fr_org is not None:
            try:
                os.chdir(mp)
                for f in copy_files_fr_org:
                    shutil.copyfile(f, os.path.join(new_weather_dir,f))
            except Exception as e:
                raise Exception("unable to copy {0} from model dir: " + \
                                "{1} to new model dir: {2}\n{3}".format(f, mp, new_weather_dir,str(e)))               
    print('Done!')




#%%
if __name__ == '__main__':
    working_path = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\weather_inputs"
    working_path2 = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\weather_inputs_modified"
    modify_tmp(working_path, working_path2, copy_files_fr_org=['pcp1.pcp'])


