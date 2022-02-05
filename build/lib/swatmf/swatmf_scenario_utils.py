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


def create_base_scn_pcp(
        nloc, base_file, scenario_dir, base_start_date, 
        base_end_date, scn_start_date, scn_end_date, output_dir
        ):
    base_df = read_pcp(base_file, nloc)
    base_df = base_df[base_start_date:base_end_date]
    wt_fds, full_paths = get_weather_folder_lists(scenario_dir)

    for n, p in zip(wt_fds, full_paths):
        os.chdir(p)
        scn_df = read_pcp('pcp1.pcp', nloc)
        scn_df = scn_df[scn_start_date:scn_end_date]
        df = pd.concat([base_df, scn_df], axis=0)

        tot_pts = len(df.columns)
        print("  Combining base pcp and scenario pcp data in {} ... processing".format(n))
        for i in tqdm(range(tot_pts)):
            df.iloc[:, i] = df.iloc[:, i].map(lambda x: '{:05.1f}'.format(x))
        df.insert(loc=0, column='year', value=df.index.year)
        df.insert(loc=1, column='jday', value=df.index.dayofyear.map(lambda x: '{:03d}'.format(x)))
        dff = df.T.astype(str).apply(''.join)

        new_pcp_dir = os.path.join(output_dir, n)
        if not os.path.exists(new_pcp_dir):
            os.mkdir(new_pcp_dir)
        try:
            with open(os.path.join(new_pcp_dir, 'pcp1.pcp'), 'w') as f:
                for i in range(4):
                    f.write('\n')
                dff.to_csv(f, line_terminator='\n', index=False, header=None)
            print("  Combining base pcp and scenario pcp data in {} ... passed".format(n))
        except Exception as e:
                raise Exception("unable to copy {} from model: " + \
                                "to new worker dir: {1}\n{2}".format(f, new_pcp_dir, str(e)))  

    print("   Done!")


def _remove_readonly(func, path, excinfo):
    """remove readonly dirs, apparently only a windows issue
    add to all rmtree calls: shutil.rmtree(**,onerror=remove_readonly), wk"""
    os.chmod(path, 128)  # stat.S_IWRITE==128==normal
    func(path)


def execute_scenarios(
            models_wd, weathers_wd,
            scn_models_wd=None, model_lists=None, weather_lists=None,
            copy_files_fr_model=None,
            copy_files_fr_weather=None,
            reuse_models=None,
            ):
    
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
            if os.path.exists(new_model_dir) and reuse_models is None:
                try:
                    shutil.rmtree(new_model_dir, onerror=_remove_readonly)#, onerror=del_rw)
                except Exception as e:
                    raise Exception("unable to remove existing worker dir:" + \
                                    "{0}\n{1}".format(new_model_dir,str(e)))
                print("  Creating new model: {}_{} ... processing".format(i, j))
                try:
                    shutil.copytree(mp, new_model_dir)
                except Exception as e:
                    raise Exception("unable to copy files from worker dir: " + \
                                    "{0} to new worker dir: {1}\n{2}".format(mp, new_model_dir,str(e)))
                print("  Creating new model: {}_{} ... passed".format(i, j)) 
            elif os.path.exists(new_model_dir) and reuse_models is True:
                try:
                    os.chdir(mp)
                    shutil.copyfile('file.cio', os.path.join(new_model_dir, 'file.cio'))
                except Exception as e:
                    raise Exception("unable to copy *.pst from main worker: " + \
                                    "{0} to new worker dir: {1}\n{2}".format(mp, new_model_dir,str(e)))
                print("  Using existing model: {}_{} ... passed".format(i, j)) 
            else:
                try:
                    shutil.copytree(mp, new_model_dir)
                except Exception as e:
                    raise Exception("unable to copy files from worker dir: " + \
                                    "{0} to new worker dir: {1}\n{2}".format(mp, new_model_dir,str(e)))
                print("  Creating new model: {}_{} ... passed".format(i, j)) 
            if copy_files_fr_model is not None and reuse_models is True:
                try:
                    os.chdir(mp)
                    for f in copy_files_fr_model:
                        shutil.copyfile(f, os.path.join(new_model_dir,f))
                except Exception as e:
                    raise Exception("unable to copy {0} from model dir: " + \
                                    "{1} to new model dir: {2}\n{3}".format(f, mp, new_model_dir,str(e)))      
            if copy_files_fr_weather is not None and reuse_models is True:
                try:
                    os.chdir(wp)
                    for f in copy_files_fr_weather:
                        shutil.copyfile(f, os.path.join(new_model_dir,f))
                except Exception as e:
                    raise Exception("unable to copy {0} from model dir: " + \
                                    "{1} to new model dir: {2}\n{3}".format(f, mp, new_model_dir,str(e)))               
            cwd = new_model_dir
            os.chdir(cwd)
            os.system("start cmd /k swatmf3-fp.exe")


def extract_scenario_results(
                    models_wd, result_files, model_results_wd=None, model_list=None, suffix=None,     
                    ):
    os.chdir(models_wd)
    if suffix is None:
        suffix = '_results'
    if model_list is None:
        model_list = [name for name in os.listdir(".") if os.path.isdir(name)]
        model_paths = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]
    else:
        model_paths = [os.path.abspath(name) for name in model_list if os.path.isdir(name)]
    if model_results_wd is None:
        model_results_wd = '..'
    
    for i , mp in tqdm(zip(model_list, model_paths)):
        model_result_dir = os.path.join(model_results_wd, "{}{}".format(i, suffix))
        if not os.path.exists(model_result_dir):
            os.mkdir(model_result_dir)
        try:
            print("  Coyping output files from {} to {} ... processing".format(i, mp))  
            os.chdir(mp)
            for f in result_files:
                shutil.copyfile(f, os.path.join(model_result_dir, f))
            print("  Coyping output files from {} to {} ... passed".format(i, mp))  
        except Exception as e:
                raise Exception("unable to copy {} from model: " + \
                                "to new worker dir: {1}\n{2}".format(f, mp, str(e)))            


def read_input_std(wd_base):
    with open(os.path.join(wd_base, "input.std"), "r") as inf:
        content = inf.readlines()
        for i, l in enumerate(inf, 1):
            if l.strip().startswith("HRU Input Summary Table 1:"):
                stnum = i
            if l.strip().startswith("HRU CN Input Summary Table:"):
                ednum = i

    hrus = []
    areas = []
    for i in content[stnum+1:ednum-2]:
        hrus.append(i.split()[1])
        areas.append(i.split()[2])

    hru_df = pd.DataFrame({'hru':hrus, 'area(ha)':areas})
    hru_df = hru_df.astype({'hru':int, 'area(ha)':float})
    hru_df['area(m2)'] = hru_df.loc[:, 'area(ha)']*10000
    return hru_df


# if __name__ == '__main__':
#     base_file = 'D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\models\\base\\pcp1.pcp'
#     scenario_dir = 'D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\weather_fgoals'
#     base_st = '1/1/1980'
#     base_ed = '12/31/2019'
#     scn_st = '1/1/2020'
#     scn_ed = '12/31/2099'
#     output_dir = 'D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\combined_base_fgoals_pcp'



#     create_base_scn_pcp(257, base_file, scenario_dir, base_st, base_ed, scn_st, scn_ed, output_dir)
#     # print('hi')



if __name__ == '__main__':
    models_wd = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\models"
    weather_wd = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\combined_base_scn_pcp"
    scn_wd = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\\scn_models"
    mrwd = "D:\\Projects\\Watersheds\\Okavango\\scenarios\\okvg_swatmf_scn_climates\\scn_bau_model_results_2020-2050"
    result_files = [
        'output.rch',    
        'output.sub',
        'output.rsv',
        'output.std',
        'swatmf_out_MF_gwsw_monthly',
        'swatmf_out_MF_head_monthly',
        'swatmf_out_MF_recharge_monthly',
        'swatmf_out_SWAT_gwsw_monthly',
        'swatmf_out_SWAT_recharge_monthly'
        ]
    scn_nams = ['ssp245', 'ssp585']
    cc_nams = ['fgoals-g3', 'inm-cm5-0', 'miroc6', 'noresm2-lm' , 'ukesm1-0-ll']
    model_list = []
    for i in scn_nams:
        for j in cc_nams:
            model_list.append('bau_{}-{}'.format(i, j))
    extract_scenario_results(scn_wd, result_files, model_results_wd=mrwd, model_list=model_list)
    # execute_scenarios(
    #         models_wd, weather_wd, scn_models_wd=scn_wd,
    #         reuse_models=True,
    #         copy_files_fr_model=['swatmf3-fp.exe']
    #         # copy_files_fr_weather=['pcp1.pcp', 'Tmp1.Tmp']
    #         )
    print('hi')