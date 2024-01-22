import os
from datetime import datetime
import pyemu
import pandas as pd
import sys
import subprocess

# path = "D:/spark/gits/swatmf"
# sys.path.insert(1, path)

from swatmf import swatmf_pst_par, utils
from swatmf import swatmf_pst_utils

wd = os.getcwd()
os.chdir(wd)

def time_stamp(des):
    time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
    print('\n' + 35*'+ ')
    print(time + ' |  {} ...'.format(des))
    print(35*'+ ' + '\n')

def modify_riv_pars():
    des = "updating river parameters"
    time_stamp(des)
    swatmf_pst_par.riv_par(wd)

def modify_hk_sy_pars_pp(pp_included):
    des = "modifying MODFLOW HK, VHK, and SY parameters"
    time_stamp(des)
    data_fac = pp_included
    for i in data_fac:
        outfile = i + '.ref'
        pyemu.utils.geostats.fac2real(i, factors_file=i+'.fac', out_file=outfile)

def execute_swat_edit():
    des = "modifying SWAT parameters"
    # time_stamp(des)
    # pyemu.os_utils.run('Swat_Edit.exe', cwd='.')
    p = subprocess.Popen('Swat_Edit.exe' , cwd = '.')
    p.wait()

def execute_swatmf():
    des = "running model"
    time_stamp(des)
    # pyemu.os_utils.run('APEX-MODFLOW3.exe >_s+m.stdout', cwd='.')
    pyemu.os_utils.run('swatmf_rel230922.exe', cwd='.')

def extract_stf_results(subs, sim_start, warmup, cal_start, cal_end):
    if time_step == 'day':
        des = "simulation successfully completed | extracting daily simulated streamflow"
        time_stamp(des)
        swatmf_pst_utils.extract_day_stf(subs, sim_start, warmup, cal_start, cal_end)
    elif time_step == 'month':
        des = "simulation successfully completed | extracting monthly simulated streamflow"
        time_stamp(des)
        swatmf_pst_utils.extract_month_stf(subs, sim_start, warmup, cal_start, cal_end)

def extract_gw_level_results(grids, sim_start, cal_end):
    des = "simulation successfully completed | extracting depth to water values"
    time_stamp(des)
    swatmf_pst_utils.extract_depth_to_water(grids, sim_start, cal_end)
    
def extract_avg_depth_to_water(
                avg_grids, start_day, 
                avg_stdate, avg_eddate,
                ):
    des = "simulation successfully completed | extracting average depth to water values"
    time_stamp(des)
    swatmf_pst_utils.extract_avg_depth_to_water(
                                avg_grids, start_day, 
                                avg_stdate, avg_eddate,
                                time_step="day")

def extract_baseflow_results(subs, sim_start, cal_start, cal_end):
    des = "simulation successfully completed | calculating baseflow ratio"
    time_stamp(des)
    swatmf_pst_utils.extract_month_baseflow(subs, sim_start, cal_start, cal_end)

if __name__ == '__main__':
    os.chdir(wd)
    swatmf_con = pd.read_csv('swatmf.con', sep='\t', names=['names', 'vals'], index_col=0, comment="#")
    # get default vals
    # wd = swatmf_con.loc['wd', 'vals']
    sim_start = swatmf_con.loc['sim_start', 'vals']
    warmup = swatmf_con.loc['warm-up', 'vals']
    cal_start = swatmf_con.loc['cal_start', 'vals']
    cal_end = swatmf_con.loc['cal_end', 'vals']
    cha_act = swatmf_con.loc['subs','vals']
    grid_act = swatmf_con.loc['grids','vals']
    riv_parm = swatmf_con.loc['riv_parm', 'vals']
    baseflow_act = swatmf_con.loc['baseflow', 'vals']
    time_step = swatmf_con.loc['time_step','vals']
    pp_act = swatmf_con.loc['pp_included', 'vals']

    
    # modifying river pars
    if swatmf_con.loc['riv_parm', 'vals'] != 'n':
        modify_riv_pars()
    if swatmf_con.loc['pp_included', 'vals'] != 'n':
        pp_included = swatmf_con.loc['pp_included','vals'].strip('][').split(', ')
        pp_included = [i.replace("'", "").strip() for i in pp_included]  
        modify_hk_sy_pars_pp(pp_included)
    # execute model
    execute_swat_edit()
    execute_swatmf()
    # extract sims
    # if swatmf_con.loc['cha_file', 'vals'] != 'n' and swatmf_con.loc['fdc', 'vals'] != 'n':
    if swatmf_con.loc['subs', 'vals'] != 'n':
        subs = swatmf_con.loc['subs','vals'].strip('][').split(', ')
        subs = [int(i) for i in subs]
        extract_stf_results(subs, sim_start, warmup, cal_start, cal_end)
    if swatmf_con.loc['grids', 'vals'] != 'n':
        grids = swatmf_con.loc['grids','vals'].strip('][').split(', ')
        grids = [int(i) for i in grids]        
        extract_gw_level_results(grids, sim_start, cal_end)
    # NOTE: this is a temporary function
    if swatmf_con.loc['avg_grids', 'vals'] != 'n':
        avg_grids = swatmf_con.loc['avg_grids','vals'].strip('][').split(', ')
        avg_grids = [int(i) for i in avg_grids]    

        avg_stdate = swatmf_con.loc['avg_dtw_stdate', 'vals']
        avg_eddate = swatmf_con.loc['avg_dtw_eddate', 'vals']
        extract_avg_depth_to_water(avg_grids, sim_start, avg_stdate, avg_eddate)

    print(wd)





