""" SWAT-MODFLOW scenario support utilities: 02/10/2021 created by Seonggyu Park
    last modified day: 02/10/2021 by Seonggyu Park
"""

import pandas as pd
import numpy as np
import time
import os
import shutil
import socket
import multiprocessing as mp
import csv
import glob
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
    print("  Done!")



def _remove_readonly(func, path, excinfo):
    """remove readonly dirs, apparently only a windows issue
    add to all rmtree calls: shutil.rmtree(**,onerror=remove_readonly), wk"""
    os.chmod(path, 128)  # stat.S_IWRITE==128==normal
    func(path)


def execute_scenarios(
            models_wd, weathers_wd,
            scn_models_wd=None, model_lists=None, weather_lists=None, copy_files=None):
    
    if model_lists is None:
        os.chdir(models_wd)
        model_nams = [name for name in os.listdir(".") if os.path.isdir(name)]
        model_paths = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]
    if weather_lists is None:
        os.chdir(weathers_wd)
        weather_nams = [name for name in os.listdir(".") if os.path.isdir(name)]
        weather_paths = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]
    if scn_models_wd is None:
        scn_models_wd = '..'

    for i, mp in zip(model_nams, model_paths):
        for j, wp in zip(weather_nams, weather_paths):
            new_model_dir = os.path.join(scn_models_wd, "{}_{}".format(i, j))
            # try:
            #     shutil.rmtree(new_model_dir, onerror=_remove_readonly)#, onerror=del_rw)
            # except Exception as e:
            #     raise Exception("unable to remove existing worker dir:" + \
            #                     "{0}\n{1}".format(new_model_dir,str(e)))
            print("  Creating new model: {}_{} ... processing".format(i, j))
            try:
                shutil.copytree(mp, new_model_dir)
                os.chdir(wp)
                if copy_files is None:
                    onlyfiles = [f for f in os.listdir(wp) if os.path.isfile(os.path.join(wp, f))]
                for f in onlyfiles:
                    shutil.copyfile(f, os.path.join(new_model_dir,f))
            except Exception as e:
                raise Exception("unable to copy files from worker dir: " + \
                                "{0} to new worker dir: {1}\n{2}".format(mp, new_model_dir,str(e)))

            cwd = new_model_dir
            os.chdir(cwd)
            os.system("start cmd /k SWAT-MODFLOW3_fp.exe")
            print("  Creating new model: {}_{} ... passed".format(i, j))  

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


if __name__ == '__main__':
    mwd = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\models"
    wwd = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\weather_inputs"
    gwd = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\gcm_performances"
    execute_scenarios(mwd, wwd, scn_models_wd=gwd)
