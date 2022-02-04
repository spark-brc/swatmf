
import pandas as pd
import numpy as np
import os
import glob
import csv
from tqdm import tqdm

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

if __name__ == '__main__':
    wd = input("Where is folder?   ")
    print("location of folder is: "+ wd)
    print("We are going through all and converting them!")

    cvt_gcm_pcp(wd=wd)
    print('Converted!')
