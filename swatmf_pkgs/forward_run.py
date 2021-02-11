import os
from datetime import datetime
import pyemu
from sm_pst_par import riv_par
from sm_pst_utils import extract_month_str, extract_watertable_sim, extract_month_baseflow


wd = os.getcwd()
os.chdir(wd)
print(wd)

# file path
rch_file = 'output.rch'
# reach numbers that are used for calibration
subs = [225, 240]
bfrs = [66, 68, 147]

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  modifying SWAT parameters...')
print(30*'+ ' + '\n')
riv_par(wd)
pyemu.os_utils.run('Swat_Edit.exe', cwd='.')

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'+ ')
print(time + ' |  running model...')
print(30*'+ ' + '\n')
# pyemu.os_utils.run('SWAT-MODFLOW3.exe >_s+m.stdout', cwd='.')
pyemu.os_utils.run('SWAT-MODFLOW3_fp_091120', cwd='.')
time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')

print('\n' + 35*'+ ')
print(time + ' | simulation successfully completed | extracting simulated values...')
print(35*'+ ' + '\n')
extract_month_str(rch_file, subs, '1/1/2003', '1/1/2003', '12/31/2007')

print('\n' + 35*'+ ')
print(time + ' | simulation successfully completed | calculating baseflow ratio...')
print(35*'+ ' + '\n')
extract_month_baseflow('output.sub', bfrs, '1/1/2003', '1/1/2003', '12/31/2007')


# extract_watertable_sim([5699, 5832], '1/1/1980', '12/31/2005')


