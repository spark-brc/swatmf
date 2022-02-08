import os
from datetime import datetime
import pyemu
import pandas as pd
from swatmf import swatmf_pst_par
from swatmf import swatmf_pst_utils

def forward_run(
                subs, grids, sim_start, cal_start, cal_end, 
                time_step, riv_parm, depth_to_water, baseflow):
    wd = os.getcwd()
    os.chdir(wd)
    print(wd)

    if riv_parm == 'y':
        time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
        print('\n' + 30*'+ ')
        print(time + ' |  updating river parameters...')
        print(30*'+ ' + '\n')
        swatmf_pst_par.riv_par(wd)

    # if time_step == 'day':

    time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
    print('\n' + 30*'+ ')
    print(time + ' |  modifying SWAT parameters...')
    print(30*'+ ' + '\n')
    pyemu.os_utils.run('Swat_Edit.exe', cwd='.')

    time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
    print('\n' + 30*'+ ')
    print(time + ' |  running model...')
    print(30*'+ ' + '\n')
    # pyemu.os_utils.run('SWAT-MODFLOW3.exe >_s+m.stdout', cwd='.')
    pyemu.os_utils.run('SWAT-MODFLOW3', cwd='.')
    time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')

    print('\n' + 35*'+ ')
    print(time + ' | simulation successfully completed | extracting simulated values...')
    print(35*'+ ' + '\n')
    if time_step == 'day':
        swatmf_pst_utils.extract_day_stf(subs, sim_start, cal_start, cal_end)
    elif time_step == 'month':
        swatmf_pst_utils.extract_month_str(subs, sim_start, cal_start, cal_end)

    if depth_to_water == 'y':
        print('\n' + 35*'+ ')
        print(time + ' | simulation successfully completed | extracting depth to water values...')
        print(35*'+ ' + '\n')        
        swatmf_pst_utils.extract_depth_to_water(grids, sim_start, cal_end)
    
    if baseflow == 'y':
        print('\n' + 35*'+ ')
        print(time + ' | simulation successfully completed | calculating baseflow ratio...')
        print(35*'+ ' + '\n')
        swatmf_pst_utils.extract_month_baseflow(subs, sim_start, cal_start, cal_end)


    # extract_watertable_sim([5699, 5832], '1/1/1980', '12/31/2005')

if __name__ == '__main__':
    cwd = os.getcwd()
    os.chdir(cwd)
    swatmf_con = pd.read_csv('swatmf.con', sep='\t', names=['names', 'vals'], index_col=0, comment="#")
    wd = swatmf_con.loc['wd', 'vals']
    subs = swatmf_con.loc['subs','vals'].strip('][').split(', ')
    subs = [int(i) for i in subs]
    grids = swatmf_con.loc['grids','vals'].strip('][').split(', ')
    grids = [int(i) for i in grids]
    sim_start = swatmf_con.loc['sim_start','vals']
    cal_start = swatmf_con.loc['cal_start','vals']
    cal_end = swatmf_con.loc['cal_end','vals']
    time_step = swatmf_con.loc['time_step','vals']
    riv_parm = swatmf_con.loc['riv_parm','vals']
    depth_to_water = swatmf_con.loc['depth_to_water','vals']
    baseflow = swatmf_con.loc['baseflow','vals']


    forward_run(
                subs, grids, sim_start, cal_start, cal_end, 
                time_step, riv_parm, depth_to_water, baseflow
                )

