"""
MODFLOW utilities for changing input variables
"""
import os
import pandas as pd
from pyemu.pst.pst_utils import SFMT,IFMT,FFMT
import glob
import shutil

class mfEdit:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        os.chdir(self.model_dir)
        self.riv_parms = []

    def create_backup_riv(self):
        riv_files = [f for f in glob.glob("*.riv")]
        if len(riv_files) == 1:
            riv_f = os.path.basename(riv_files[0])
            # duplicate original riv file
            if not os.path.exists('riv_package.org'):
                shutil.copy(riv_f, 'riv_package.org')
                print('The original river package "{}" has been backed up...'.format(riv_f))
            else:
                print('The "riv_package.org" file already exists...')        

    def create_riv_par(self, chns, chg_type=None, rivcd=None, rivbot=None, val=None):
        if rivcd is None:
            rivcd =  ['rivcd_{}'.format(x) for x in chns]
        if rivbot is None:
            rivbot =  ['rivbot_{}'.format(x) for x in chns]
        if chg_type is None:
            chg_type = 'unfchg'
        if val is None:
            val = 0.0000001
        riv_f = rivcd + rivbot
        df = pd.DataFrame()
        df['parnme'] = riv_f
        df['chg_type'] = chg_type
        df['val'] = val
        for i in range(len(df)):
            if (df.iloc[i, 0][:5]) == 'rivcd':
                df.iloc[i, 1] = 'pctchg'
        df.index = df.parnme
        with open('mf_riv.par', 'w') as f:
            f.write("# modflow_par file.\n")
            f.write("NAME   CHG_TYPE    VAL\n")
            f.write(
                df.loc[:, ["parnme", "chg_type", "val"]].to_string(
                                                            col_space=0,
                                                            formatters=[SFMT, SFMT, SFMT],
                                                            index=False,
                                                            header=False,
                                                            justify="left")
                )
        print("'mf_riv.par' file has been exported to the SWAT-MODFLOW working directory!")
        return df
    
    def read_modflow_par(self):
        # read mf_riv_par.par
        riv_pars = pd.read_csv('mf_riv.par', sep=r'\s+', comment='#', index_col=0)
        # get parameter types and channel number
        riv_pars['par_type'] = [x.split('_')[0] for x in riv_pars.index]
        riv_pars['chn_no'] = [x.split('_')[1] for x in riv_pars.index]
        return riv_pars
    
    def riv_par_to_template_file(self):
        """write a template file for a SWAT parameter value file (model.in).

        Args:
            model_in_file (`str`): the path and name of the existing model.in file
            tpl_file (`str`, optional):  template file to write. If None, use
                `model_in_file` +".tpl". Default is None
        Note:
            Uses names in the first column in the pval file as par names.

        Example:
            pest_utils.model_in_to_template_file('path')

        Returns:
            **pandas.DataFrame**: a dataFrame with template file information
        """
        riv_par_file = "mf_riv.par"
        tpl_file = riv_par_file + ".tpl"
        mf_par_df = pd.read_csv(
                            riv_par_file,
                            sep=r'\s+',
                            header=None, skiprows=2,
                            names=["parnme", "chg_type", "parval1"])
        mf_par_df.index = mf_par_df.parnme
        mf_par_df.loc[:, "tpl"] = mf_par_df.parnme.apply(lambda x: " ~   {0:12s}   ~".format(x))
        with open(tpl_file, 'w') as f:
            f.write("ptf ~\n# modflow_par template file.\n")
            f.write("NAME   CHG_TYPE    VAL\n")
            f.write(mf_par_df.loc[:, ["parnme", "chg_type", "tpl"]].to_string(
                                                            col_space=0,
                                                            formatters=[SFMT, SFMT, SFMT],
                                                            index=False,
                                                            header=False,
                                                            justify="left"))
        return mf_par_df

    def update_riv_parms(self):
        riv_parms = self.riv_parms
        write_new_riv(riv_parms)


def write_new_riv():
    riv_files = [f for f in glob.glob("*.riv")]
    if len(riv_files) == 1:
        riv_f = os.path.basename(riv_files[0])
    elif len(riv_files) > 1:
        print(
                "You have more than one River Package file!("+str(len(riv_files))+")"+"\n"
                + str(riv_files)+"\n"
                + "Solution: Keep only one file!")
    else:
        print("File Not Found! - We couldn't find your *.riv file.")
    parms_df = pd.read_csv("mf_riv.par", sep=r'\s+', comment="#")
    with open("riv_package.org", "r") as f:
        data = f.readlines()
        for parnam in parms_df["NAME"]:
            parn = parnam.split("_")[1]
            par_type = parnam.split("_")[0]
            chg_type = parms_df.loc[parms_df["NAME"]==parnam, "CHG_TYPE"].values[0]
            value = parms_df.loc[parms_df["NAME"]==parnam, "VAL"].values[0]
            c = 0
            for line in data:
                if line.strip().endswith(parn):
                    new_line = replace_line(line, value, chg_type, par_type)
                    data[c] = new_line
                c += 1
    with open(riv_f, "w") as wf:
        wf.writelines(data)
    print(f" {'>'*3} {os.path.basename(riv_f)}" + " file is overwritten successfully!")

def replace_line(line, value, method, par_type):
    parts = line.split()
    rivcd = parts[4].strip()
    rivbot = parts[5].strip()
    rivstage = parts[3].strip()
    ly = parts[0].strip()
    row = parts[1].strip()
    col = parts[2].strip()
    # print(c)
    if len(parts) > 6:
        extra = " ".join(str(x) for x in parts[6:])
    else:
        extra = " "
    if par_type == "rivcd":    
        if method == "pctchg":
            newcond = float(rivcd) + (float(rivcd) * float(value)/100)
        elif method == "unfchg":
            newcond = float(rivcd) + float(value)
        new_line = (
                    f'{int(ly):5d}{int(row):5d}{int(col):5d}'+
                    f'{float(rivstage):14.6e}{float(newcond):14.6e}{float(rivbot):14.6e}'
                    + "  "
                    f'{extra}\n'
                    )
    if par_type == "rivbot":    
        if method == "pctchg":
            newbot = float(rivbot) + (float(rivbot) * float(value)/100)
        elif method == "unfchg":
            newbot = float(rivbot) + float(value)
        new_line = (
                    f'{int(ly):5d}{int(row):5d}{int(col):5d}'+
                    f'{float(rivstage):14.6e}{float(rivcd):14.6e}{float(newbot):14.6e}'
                    + "  "
                    f'{extra}\n'
                    )
    return new_line










