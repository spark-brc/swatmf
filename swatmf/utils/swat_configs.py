"""
SWAT utilities for changing text input files and executing the model
"""
import os
import platform
import re
import subprocess as sp
import pandas as pd
import sys
import datetime as dt
from tqdm import tqdm


class SwatEdit:
    """ create object with SWAT settings for running a model
    """
    def __init__(self, model_dir):
        # self.output_dir = '/tmp/output_swat'
        # self.temp_dir = '/tmp/swat_runs'
        
        self.model_dir = model_dir
        os.chdir(model_dir)
        self.back_dir = os.path.join(self.model_dir, "backup") # should be backup
        self.subbasins = []
        self.hrus = []
        self.param = []
        self.out_reach = []
        self.fig_file = ''
        self.conc_output = []
        self.sim_ts = {}
        self.run_output_dir = ''
        self.verbose = True
        self.verbose_swat = False
        self.executed = False
        self.saveconc_files = []
        self.drainmod = False
        self.ops_table = []
        self.mgt_table = []
        self.subbasins_ops = []
        self.hrus_ops = []
        self.subbasins_mgt = []
        self.hrus_mgt = []

    def get_subbasins(self):
        out_reaches = self.out_reach
        fig_file = self.fig_file
        if len(out_reaches) == 0:
            subbasins_list = []
        else:
            subbasin_list = []
            route_list = []
            add_list = []
            if len(fig_file) == 0:
                print(
                    'Warning: You must provide the full path to the .fig file to get the subbasins for the watershed. '
                    'Changes (if any) will be applied to all subbasins...')
                subbasins_list = []
            else:
                with open(fig_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        temp = line.split()
                        if temp[0] == "subbasin":
                            subbasin_list.append(temp)
                        elif temp[0] == "route":
                            route_list.append(temp)
                        elif temp[0] == "add":
                            add_list.append(temp)
                    subbasin_list = pd.DataFrame(subbasin_list)
                    route_list = pd.DataFrame(route_list)
                    add_list = pd.DataFrame(add_list)
                subbasins_list = []
                for out_reach in out_reaches:
                    subbasins = []
                    # find inflow element to out_reach in route_list
                    inflow = int(route_list.loc[pd.to_numeric(route_list[3]) == out_reach].to_numpy()[:, -1:])
                    # get list of subbasins draining to out_reach
                    subbasins = subbasin_root(inflow, subbasin_list, route_list, add_list, subbasins)
                    # append to general list
                    subbasins_list.append(subbasins)
        self.subbasins = subbasins_list
        # return subbasin_list

    def read_swat_pars_db(self):
        swat_cal_pars = pd.read_csv("swat_pars.db.csv", comment="#")
        cal_pars = swat_cal_pars.loc[swat_cal_pars["flag"]==1]
        return cal_pars

    def create_swat_pars_cal(self):
        df = self.read_swat_pars_db()
        # df['chg_type'] = df['chg_type'].astype(str)
        # df['chg_type'].astype(str, inplace=True)
        # df["chg_type"].fillna("multiply", inplace=True)
        df.fillna({"chg_type":"multiply"}, inplace=True)
        df["chg_val"] = 0
        SFMT_SHORT = lambda x: "{0:<10s} ".format(str(x))
        SFMT_LONG = lambda x: "{0:<20s} ".format(str(x))
        formatters = [SFMT_LONG] + [SFMT_SHORT]*2 + [SFMT_LONG] + [SFMT_SHORT]*2 
        
        # create model.in
        with open("swat_pars.cal", 'w') as f:
            # f.write("{0:10d} #NP\n".format(mod_df.shape[0]))
            f.write("# this is test...\n")
            f.write(
                df.loc[:, ["parnam", "obj_type", "chg_type", "chg_val", "lb", "ub"
                           ]].to_string(
                                        col_space=2,
                                        formatters=formatters,
                                        index=False,
                                        header=True,
                                        justify="left"))
        df["chg_val"] = df.parnam.apply(lambda x: " ~   {0:15s}   ~".format(x))
        with open("swat_pars.cal.tpl", 'w') as tplf:
            tplf.write("ptf ~\n")
            tplf.write("# this is test...\n")
            tplf.write(
                df.loc[:, ["parnam", "obj_type", "chg_type", "chg_val", "lb", "ub"
                           ]].to_string(
                                        col_space=2,
                                        formatters=formatters,
                                        index=False,
                                        header=True,
                                        justify="left"))

    def read_new_parms(self):
        df = pd.read_csv("swat_pars.cal", sep=r"\s+", comment="#")
        df = df[["parnam", "chg_val", "chg_type","obj_type"]]
        dic  = df.set_index('parnam').T.to_dict('list')
        return dic



    def update_swat_parms(self):
        # get attributes from SWAT configuration object
        param = self.param
        subbasins = self.subbasins
        hrus = self.hrus
        drainmod = self.drainmod
        mgt_table = self.mgt_table
        subbasins_mgt = self.subbasins_mgt
        hrus_mgt = self.hrus_mgt
        ops_table = self.ops_table
        subbasins_ops = self.subbasins_ops
        hrus_ops = self.hrus_ops
        
        # create output directory if it does not exist

        # try:
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        # except OSError:
        #     print('Tried to create directory {:s} that already exists...'.format(output_dir))

        # try:
        #     if not os.path.exists(temp_dir):
        #         os.makedirs(temp_dir)
        # except OSError:
        #     print('Tried to create directory {:s} that already exists...'.format(temp_dir))

        # # unzip SWAT model in temp directory
        # input_dir_file = os.path.abspath(model_file)
        backup_dir = self.back_dir # this might be backup folder?
        model_dir = self.model_dir # this might be backup folder?

        # if not os.path.exists(model_dir_file):
        #     os.makedirs(model_dir_file)

        # sp.run(['unzip', input_dir_file, '-d', model_dir_file], stdout=sp.DEVNULL, stderr=sp.DEVNULL, check=True)
        
        # if drainmod:
        #     update_txt(model_dir_file)

        # # copy/update SWAT executable to directory containing text input files
        # swat_dir_file = os.path.abspath(swat_dir + '/' + swat_exec_name)

        # if platform.system() == 'Linux':
        #     sp.run(['cp', swat_dir_file, model_dir_file, '--update'], check=True)
        # elif platform.system() == 'Darwin':
        #     sp.run(['cp', '-f', swat_dir_file, model_dir_file], check=True)

        # prepare SWAT input text files
        # sp.run(['cp', '-r', backup_dir, model_dir], check=True)
        # output_dir = os.path.abspath(output_dir + '/' + new_model_name)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # write SWAT input text files        
        write_new_files(param, subbasins, hrus, backup_dir, model_dir)        
        write_mgt_tables(mgt_table, 'mgt', subbasins_mgt, hrus_mgt, model_dir)
        write_mgt_tables(ops_table, 'ops', subbasins_ops, hrus_ops, model_dir)

        # # remove SWAT model files from temporal directory
        # sp.run(['rm', '-rf', model_dir_file], check=True)

        # self.run_output_dir = output_dir

        # return output_dir

def write_new_files(param_all, subs, hrus, input_dir, model_dir):
    """Write new SWAT text input files based on a list of parameters to be changed.
    input(s):
        param_all = dictionary containing a set of n parameters to be modified. The format is as follows:
            {'key_name_1':[list_1], ..., 'key_name_n':[list_n]}
            where:
                'keyname_i' is the ith string with the SWAT name of the ith parameter to be modified
                [list_i] is the ith list of three elements defining the inputs 'value','method', and 'ext' of the function 'replaceLine'
        input_dir = directory for the SWAT TxtInOut folder to be modified
        output_dir = output directory where the new text files will be written 
    """
    if type(param_all) == dict:
        param_all = [param_all]

    if (len(param_all) < len(subs)) and (len(param_all) != 0 and len(param_all) != 1):
        sys.exit('List of parameters must have the same length as subbasins list or be empty')
    dir_list = os.listdir(input_dir)

    # get list of subbasins and hrus ready to write files
    subs, hrus = prepare_subs_hrus(dir_list, subs, hrus)
    param_list = []
    if len(param_all) < len(subs):
        if len(param_all) == 1:
            param_list = [param_all[0] for _ in subs]
        elif len(param_all) == 0:
            param_list = [[] for _ in subs]
    elif (len(param_all) > 1) and (len(param_all) > len(subs)):
        param_list = [param_all[i] for i, _ in enumerate(subs)]
    else:
        param_list = list(param_all)

    # sort param, subs and hrus according to size of subs
    subs, hrus, param_list = sort_subs_hrus(subs, hrus, param_list)
    for i, sub in enumerate(subs):
        hru = hrus[i]
        param = param_list[i]
        param_df = pd.DataFrame.from_dict(
                                        param,
                                        orient='index',
                                        columns=['value', 'method', 'ext']
                                        )
        exts = param_df.ext.unique().tolist()
        # exts = 
        write_ext_files(param_df, dir_list, sub, hru, exts, input_dir, model_dir)


def write_mgt_tables(mgt_tables, ext, subs, hrus, output_dir):
    if type(mgt_tables) is not list:
        mgt_tables = [mgt_tables]
    if (len(mgt_tables) < len(subs)) and (len(mgt_tables) != 0 and len(mgt_tables) != 1):
        sys.exit('List of management tables must have the same length as subbasins list or be empty')
    dir_list = os.listdir(output_dir)

    # get list of subbasins and hrus ready to write files
    subs, hrus = prepare_subs_hrus(dir_list, subs, hrus)

    if len(mgt_tables) < len(subs):
        if len(mgt_tables) == 1:
            mgt_list = [mgt_tables[0] for _ in subs]
        elif len(mgt_tables) == 0:
            mgt_list = [[] for _ in subs]
    elif (len(mgt_tables) > 1) and (len(mgt_tables) > len(subs)):
        mgt_list = [mgt_tables[i] for i, _ in enumerate(subs)]
    else:
        mgt_list = list(mgt_tables)

    # sort param, subs and hrus according to size of subs
    subs, hrus, mgt_list = sort_subs_hrus(subs, hrus, mgt_list)

    for i, sub in enumerate(subs):
        hru = hrus[i]
        mgt_table = mgt_list[i]
        if type(mgt_table) == list and len(mgt_table) == 0:
            break
        else:
            write_mgt_files(mgt_table, ext, dir_list, sub, hru, output_dir)


def write_mgt_files(mgt_table, ext, dir_list, subbasins, hrus, output_dir):
    
    # build reference lists of hru codes
    hru_ref = ['{:05d}{:04d}'.format(x, y) for i, x in enumerate(subbasins) for y in hrus[i]]
    
    # create list of files to change
    files = ['{part1}.{part2}'.format(part1=x, part2=ext) for x in hru_ref]
    
    if ext == 'mgt':
        table = mgt_table.op_sched
        ind = 30
    elif ext == 'ops':
        table = mgt_table.schedule
        ind = 1
    
    table = [x + '\n' for x in table]
    
    for file in files:        
        with open(os.path.abspath(output_dir + '/' + file), 'r', encoding='ISO-8859-1') as f:
            data = f.readlines()
            del data[ind:]
            data = data + table

        with open(os.path.abspath(output_dir + '/' + file), "w") as f:
            f.writelines(data)


def write_ext_files(param_df, dir_list, subbasins, hrus, exts, input_dir, output_dir):
    # build reference lists of subbasin and hru codes
    sub_ref = ['{:05d}0000'.format(x) for x in subbasins]
    hru_ref = ['{:05d}{:04d}'.format(x, y) for i, x in enumerate(subbasins) for y in hrus[i]]
    for ext in exts:
        # get files in input directory with extension '.ext'
        files_all = [x for x in dir_list if (x.endswith('.{}'.format(ext)) and not x.startswith('output'))]
        param = param_df.loc[(param_df.ext == ext)].to_dict(orient='index')
        n_line = []
        txtformat = []
        print(f"  modifying '{ext}' obj ...")
        if ext == 'sol':
            var_list = ['SNAM', 'HYDGRP', 'SOL_ZMX', 'ANION_EXCL', 'SOL_CRK', 'TEXTURE',
                       'SOL_Z', 'SOL_BD', 'SOL_AWC', 'SOL_K', 'SOL_CBN', 'SOL_CLAY', 'SOL_SILT',
                       'SOL_SAND', 'SOL_ROCK', 'SOL_ALB', 'USLE_K', 'SOL_EC', 'SOL_CAL', 'SOL_PH']
            n_line = [x for x in range(2, len(var_list) + 2)]
            txtformat = ['s', 's', '12.2f', '6.3f', '6.3f', 's',
                         '12.2f', '12.2f', '12.2f', '12.2f', '12.2f', '12.2f', '12.2f',
                         '12.2f', '12.2f', '12.2f', '12.2f', '12.2f', '12.2f', '12.2f']
        elif ext == 'chm':
            var_list = ['SOL_NO3', 'SOL_ORGN', 'SOL_SOLP', 'SOL_ORGP', 'PPERCO_SUB']
            n_line = [x for x in range(4, len(var_list) + 4)]
            txtformat = ['12.2f', '12.2f', '12.2f', '12.2f', '12.2f']
        else:
            var_list = []

        # create list of files to change
        files_all.sort()
        if len(files_all) > 1:
            crit = int(files_all[0][8])
            if crit == 0:
                files = ['{part1}.{part2}'.format(part1=x, part2=ext) for x in sub_ref]
            else:
                files = ['{part1}.{part2}'.format(part1=x, part2=ext) for x in hru_ref]
        else:
            files = files_all  # this is the case of .bsn and .wwq
        
        # modify list of files
        # for file in tqdm(files):
        for file in tqdm(files):
            with open(os.path.abspath(input_dir + '/' + file), 'r', encoding='ISO-8859-1') as f:
                data = f.readlines()
                param_names = list(param.keys())
                for param_name in param_names:
                    c = 0
                    if var_list:
                        ind = [i for i, x in enumerate(var_list) if param_name in x][0]
                        c = n_line[ind] - 1
                        num_format = txtformat[ind]
                    else:
                        if (ext == 'rte') & (param_name == 'CH_ERODMO'):
                            c = 23
                            num_format = '6.2f'
                        elif (ext == 'rte') & (param_name == 'HRU_SALT'):
                            c = 28
                            num_format = '6.2f'
                        else:
                            for line in data:
                                if re.search(param_name, line):
                                    num_format = ''
                                    break
                                else:
                                    c = c + 1
                    new_line = replace_line(
                                            data[c],
                                            param[param_name]['value'],
                                            param[param_name]['method'],
                                            param[param_name]['ext'],
                                            num_format
                                            )
                    data[c] = new_line
            with open(os.path.abspath(output_dir + '/' + file), "w") as f:
                f.writelines(data)


def replace_line(line, value, method, ext, num_format):
    """Replace old values in a line by new values.
    input(s):
        line = string containing values to be changed. Generally, it consists of two parts separated by either ':' or '|'
        value = number or string used to determine the new values in the given line
        method = indicates how the new values are going to be determined. So far, there are four options:
            'replace', the new value is the input 'value';
            'multiply', the new value is obtained by changing the original value in the line by a fraction given by the input 'value';
            'factor', the new value is the old value multiplied by the input 'value'
            'add', the new value is the old value plus the input 'value'
        ext = SWAT input file extension
        num_format = format of the number to be replaced
    output(s):
        new_line = string containing the original line with values modified according to the inputs 'value' and 'method'
    """

    if (ext == 'sol') | (ext == 'chm'):  # especial case for soil properties
        parts = line.split(':')
        num = parts[1].strip()
        new_value = []
        if is_float(num) | is_int(num):  # change single numeric values
            if method == 'replace':
                new_value = value
            elif method == 'multiply':
                new_value = (1 + value) * float(num)
            elif method == 'factor':
                new_value = value * float(num)
            elif method == 'add':
                new_value = value + float(num)
            part1 = '{:{}}'.format(new_value, num_format)

        else:  # change array of values
            if not num_format == 's':
                # get the number of positions from num_format
                n = int(num_format.split('.')[0])
                # split string of numbers based on N and convert to float
                # nums = [float(num[i:i + n]) for i in range(0, len(num), n)]
                nums = num.split()
                nums = [float(num_) for num_ in nums]
                if method == 'replace':
                    new_value = [value for _ in nums]
                elif method == 'multiply':
                    new_value = [(1.0 + value) * x for x in nums]
                elif method == 'factor':
                    new_value = [value * x for x in nums]
                elif method == 'add':
                    new_value = [(value + x) for x in nums]
                part1 = ''.join(['{:{}}'.format(x, num_format) for x in new_value])
            else:  # change strings
                part1 = ' {:13s}'.format(value)
        new_line = '{part1}:{part2}\n'.format(part1=parts[0], part2=part1)

    elif ext == 'rte':  # especial case for routing (some parameters can be an array of values)
        parts = line.split('|')
        num = parts[0].strip()
        new_value = []

        if is_float(num) | is_int(num):  # change single numeric values
            if method == 'replace':
                new_value = value
            elif method == 'multiply':
                new_value = (1 + value) * float(num)
            elif method == 'factor':
                new_value = value * float(num)
            elif method == 'add':
                new_value = value + float(num)

            if is_int(num): # determine number format since num_format is not provided
                part0 = '{:14d}'.format(int(new_value))
            elif is_float(num):  
                nd = 6 #abs(decimal.Decimal(num).as_tuple().exponent)
                part0 = '{:14.{}f}'.format(new_value, nd)
            else:
                part0 = '{:14s}'.format(value)
            new_line = '{part1}    |{part2}'.format(part1=part0, part2=parts[1])

        else:  # change array of values (format is provided)
            n = int(num_format.split('.')[0])  # get the number of positions from num_format
            nums = [float(num[i:i + n]) for i in
                    range(0, len(num), n)]  # split string of numbers based on N and convert to float

            if method == 'replace':
                new_value = [value for _ in nums]
            elif method == 'multiply':
                new_value = [(1.0 + value) * x for x in nums]
            elif method == 'factor':
                new_value = [value * x for x in nums]
            elif method == 'add':
                new_value = [(value + x) for x in nums]
            part0 = ''.join(['{:{}}'.format(x, num_format) for x in new_value])
            if len(parts) < 2:
                parts.append('\n')
            new_line = '{part1}|{part2}'.format(part1=part0, part2=parts[1])

    else:  # generic case (only single values; array of values requires an especial case)
        parts = line.split('|')
        num = parts[0].strip()
        new_value = []

        if method == 'replace':
            new_value = value
        elif method == 'multiply':
            new_value = (1 + value) * float(num)
        elif method == 'factor':
            new_value = value * float(num)
        elif method == 'add':
            new_value = value + float(num)
        
        if is_int(num):
            part0 = '{:16d}'.format(int(new_value))
        elif is_float(num):
            nd = 6 #abs(decimal.Decimal(num).as_tuple().exponent)
            part0 = '{:16.{}f}'.format(new_value, nd)
        else:
            part0 = '{:13s}   '.format(value)

        new_line = '{part1}    |{part2}'.format(part1=part0, part2=parts[1])

    return new_line

def sort_subs_hrus(subs, hrus, params):
    ind = [i for i, _ in sorted(enumerate(subs), key=lambda x: len(x[1]), reverse=True)]
    subs2 = [subs[i] for i in ind]
    hrus2 = [hrus[i] for i in ind]
    params2 = [params[i] for i in ind]
    return subs2, hrus2, params2


def prepare_subs_hrus(dir_list, subs, hrus):
    # get all subbasins and hrus    
    sub_list, hru_list = get_all_subs_hrus(dir_list)
    # complete subs and hrus lists if necessary
    if len(subs) == 0:
        subs = [list(sub_list)]
        hrus = [list(hru_list)]
    else:        
        subs = [sub_i if len(sub_i) > 0 else list(sub_list) for sub_i in subs]        
        if len(hrus) == 0:
            hrus = list(hrus)
            for sub_i in subs:
                hru_i = []
                for element in sub_i:
                    ind = [i for i, x in enumerate(sub_list) if x == element]
                    for i in ind:
                        hru_i.append(hru_list[i])
                hrus.append(hru_i)
        else:
            for i, sub_i in enumerate(subs):
                hru_i = hrus[i]
                if len(hru_i) == 0:
                    for element in sub_i:
                        ind = [w for w, x in enumerate(sub_list) if x == element]
                        for w in ind:
                            hru_i.append(hru_list[w])
                    hrus[i] = hru_i

                    # subs, hrus = simplify_subs(subs,hrus) # uncomment if you want to remove duplicated subbasins and respective hrus
    return subs, hrus


def get_all_subs_hrus(dir_list):
    hru_codes = [
        x.split('.')[0] for x in dir_list if (x.endswith('.{}'.format('hru')) and not x.startswith('output'))
        ]
    sub_aux = [x[0:5] for x in hru_codes]
    hru_aux = [x[-4:] for x in hru_codes]
    sub_list = sorted(list(set(sub_aux)))
    hru_list = []
    for sub_i in sub_list:
        ind = [i for i, x in enumerate(sub_aux) if x == sub_i]
        temp = [hru_aux[i] for i in ind]
        hru_list.append(sorted(temp))
    sub_list = [int(x) for x in sub_list]
    hru_list = [[int(x) for x in y] for y in hru_list]
    return sub_list, hru_list

def subbasin_root(inflow, subbasin_list, route_list, add_list, subbasins):
    type_inflow = type_element(inflow, subbasin_list, route_list, add_list)
    if type_inflow == 'add':
        inflow_elements = add_list.loc[pd.to_numeric(add_list[2]) == inflow].to_numpy()[:, -2:].reshape(-1, 1)
        for element in inflow_elements:
            subbasins = subbasin_root(int(element), subbasin_list, route_list, add_list, subbasins)
    elif type_inflow == 'route':
        element = route_list.loc[pd.to_numeric(route_list[2]) == inflow].to_numpy()[:, -1:]
        subbasins = subbasin_root(int(element), subbasin_list, route_list, add_list, subbasins)
    elif type_inflow == 'subbasin':
        subbasins.append(inflow)
    subbasins.sort()
    return subbasins


def type_element(inflow, subbasin_list, route_list, add_list):
    temp = subbasin_list
    if is_empty_list(inflow, temp):
        temp = route_list
        if is_empty_list(inflow, temp):
            temp = add_list
            if is_empty_list(inflow, temp):
                sys.exit('Verify your .fig file since an element was not found...')
            else:
                type_out = 'add'
        else:
            type_out = 'route'
    else:
        type_out = 'subbasin'
    return type_out

# set of definitions for tracking back draining subbasins using .fig files
def is_empty_list(inflow, ref):
    aux = ref.loc[pd.to_numeric(ref[2]) == inflow]
    if len(aux) > 0:
        iflag = bool(0)
    else:
        iflag = bool(1)
    return iflag


def is_float(s):
    """Determine whether a string can be converted to a float number.
    input(s):
        s = string
    output(s):
        True/False
    """
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_int(s):
    """Determine whether a string can be converted to an integer number.
    input(s):
        s = string
    output(s):
        True/False
    """
    try:
        int(s)
        return True
    except ValueError:
        return False



# def plot_tot():
if __name__ == '__main__':
    # wd = "/Users/seonggyu.park/Documents/projects/kokshila/swatmf_results"
    model_dir = "/Users/seonggyu.park/Documents/projects/tools/test/Honeyoy_Model_manual"
    model_dir = "D:\\tmp\\swatmf_dir"
    m1 = SwatEdit(model_dir)
    subbasins_filename = 'D:\\Projects\\Tools\\swat-pytools\\resources\\csv_files\\subbasins.csv'
    subbasins = [i for i in range(1, 198)]

    # Step 2: Parameters to change



    # pars = m1.read_new_parms()
    # print(pars)

    new_parms = m1.read_new_parms()
    # print(new_parms)


    m1.param = [new_parms]
    m1.subbasins = [subbasins]
    m1.update_swat_parms()
    # # swat_model.run_swat()





