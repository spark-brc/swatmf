import os
from datetime import datetime
import pyemu
from sm_pst_par import riv_par
from sm_pst_utils import extract_month_str, extract_watertable_sim

wd = os.getcwd()
os.chdir(wd)
print(wd)

# file path
rch_file = 'output.rch'
# reach numbers that are used for calibration
subs = [124, 7, 92, 147, 56, 168, 66, 138, 228, 68, 210, 79, 63]

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'* ')
print(time + ' |  modifying SWAT parameters...')
print(30*'* ' + '\n')
# riv_par(wd)
pyemu.os_utils.run('Swat_Edit.exe', cwd='.')

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'* ')
print(time + ' |  running model...')
print(30*'* ' + '\n')
# pyemu.os_utils.run('SWAT-MODFLOW34.exe >_s+m.stdout', cwd='.')
pyemu.os_utils.run('SWAT-MODFLOW3_fp.exe', cwd='.')

time = datetime.now().strftime('[%m/%d/%y %H:%M:%S]')
print('\n' + 30*'* ')
print(time + ' | simulation successfully completed | extracting simulated values...')
print(30*'* ' + '\n')
extract_month_str(rch_file, subs, '1/1/1963', '1/1/1963', '12/31/1970')
# extract_watertable_sim([5699, 5832], '1/1/1980', '12/31/2005')

