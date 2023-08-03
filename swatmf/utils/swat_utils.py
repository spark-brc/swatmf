import os
import glob
import shutil
import pandas as pd
# from .. import gcm_analysis
from tqdm import tqdm


class WeatherData(object):
    
    def __init__(self, wd):
        """weather dataset

        Args:
            wd (path): weather dataset path
        """
        
        os.chdir(wd)
    
    def get_weather_folder_lists(self):
        """return weather folder names and full paths

        Returns:
            str: return weather folder names and full paths
        """
        wt_fds = [name for name in os.listdir(".") if os.path.isdir(name)]
        full_paths = [os.path.abspath(name) for name in os.listdir(".") if os.path.isdir(name)]
        return wt_fds, full_paths
        
    @property
    def cvt_gcm_pcp(self):
        full_paths = self.fullpaths
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

    def read_gcm(self, file_path, subs):
        df = pd.read_csv(
            file_path, index_col=0, parse_dates=True)
        df.columns = ['sub_{}'.format(i+1) for i in range(len(df.columns))]
        cols = ['sub_{}'.format(i) for i in subs]
        df = df[cols]
        return df

    # def read_gcm_avg(file)
    def read_pcp(self, pcp_path, nloc):
        with open(pcp_path, 'r') as f:
            content = f.readlines()
        doy = [int(i[:7]) for i in content[4:]]
        date = pd.to_datetime(doy, format='%Y%j')
        df = pd.DataFrame(index=date)
        start_num = 7
        for i in tqdm(range(nloc)):
            pp = [float(i[start_num:start_num+5]) for i in content[4:]]
            start_num += 5
            new_columns = pd.DataFrame({f"sub_{i+1}":pp}, index=date)
            df = pd.concat([df, new_columns], axis=1)
        return df

    def read_tmp(self, tmp_path, nloc):
        with open(tmp_path, 'r') as f:
            content = f.readlines()
        doy = [int(i[:7]) for i in content[4:]]
        date = pd.to_datetime(doy, format='%Y%j')
        df_max = pd.DataFrame(index=date)
        df_min = pd.DataFrame(index=date)
        max_start_num = 7
        min_start_num = 12
        for i in tqdm(range(nloc)):
            max_tmp = [float(i[max_start_num:max_start_num+5]) for i in content[4:]]
            max_start_num += 10
            new_max = pd.DataFrame({f"max_sub{i+1}":max_tmp}, index=date)
            df_max = pd.concat([df_max, new_max], axis=1)
            min_tmp = [float(i[min_start_num:min_start_num+5]) for i in content[4:]]
            min_start_num += 10
            new_min = pd.DataFrame({f"min_sub{i+1}":min_tmp}, index=date)
            df_min = pd.concat([df_min, new_min], axis=1)
        return df_max, df_min

    def modify_tmp(self, weather_wd, weather_modified_wd, copy_files_fr_org=None):
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

    def cvt_pcp_each(self, output_wd, pcp_nloc, pcp_file=None):
        if pcp_file is None:
            pcp_file = 'pcp1.pcp'
        nms, fps = self.get_weather_folder_lists()
        for nm, fp in zip(nms, fps):
            print(f'reading model {fp} ... processing')
            df = self.read_pcp(os.path.join(fp, pcp_file), pcp_nloc)
            stdate = [df.index[0].strftime('%Y%m%d')]
            if os.path.exists(os.path.join(output_wd, nm, 'PCP')):
                shutil.rmtree(os.path.join(output_wd, nm, 'PCP'), ignore_errors=True)
            print(f'writing model {nm} to indivisual file ... processing')      
            for i in tqdm(range(len(df.columns))):
                wd_df = stdate + df.iloc[:, i].map(lambda x: '{:.3f}'.format(x)).tolist()
                os.makedirs(os.path.join(output_wd, nm, 'PCP'), exist_ok=True)
                os.chdir(os.path.join(output_wd, nm, 'PCP'))
                with open(f'PCP_{i+1:03d}.txt', 'w') as fp:
                    for l in wd_df:
                        fp.write(l+"\n")

    def cvt_tmp_each(self, output_wd, tmp_nloc, tmp_file=None):
        if tmp_file is None:
            tmp_file = 'Tmp1.Tmp'
        nms, fps = self.get_weather_folder_lists()
        for nm, fp in zip(nms, fps):
            print(f'reading model {fp} ... processing')
            max_df, min_df = self.read_tmp(os.path.join(fp, tmp_file), tmp_nloc)
            stdate = [max_df.index[0].strftime('%Y%m%d')]
            if os.path.exists(os.path.join(output_wd, nm, 'TMP')):
                shutil.rmtree(os.path.join(output_wd, nm, 'TMP'), ignore_errors=True)
            print(f'writing model {nm} to indivisual file ... processing')      
            for i in tqdm(range(len(max_df.columns))):
                maxs_ = max_df.iloc[:, i].map(lambda x: '{:.2f}'.format(x)).tolist()
                mins_ = min_df.iloc[:, i].map(lambda x: '{:.2f}'.format(x)).tolist()
                ab = [f'{mx},{mi}' for mx, mi in zip(maxs_, mins_)]
                abf = stdate + ab

                os.makedirs(os.path.join(output_wd, nm, 'TMP'), exist_ok=True)
                os.chdir(os.path.join(output_wd, nm, 'TMP'))
                with open(f'TMP_{i+1:03d}.txt', 'w') as fp:
                    for l in abf:
                        fp.write(l+"\n")



    def cvt_tmp_each2(self, output_wd, tmp_nloc, tmp_file=None):
        if tmp_file is None:
            tmp_file = 'Tmp1.Tmp'
        nms, fps = self.get_weather_folder_lists()
        for nm, fp in zip(nms[1:], fps[1:]):
            print(f'reading model {fp} ... processing')
            max_df, min_df = self.read_tmp(os.path.join(fp, tmp_file), tmp_nloc)
            
            max_mean = max_df.mean(axis=1)
            min_mean = min_df.mean(axis=1)
            print(f'inserting additional reaches in dataframe ... processing')   
            for j in tqdm(range(155, 258)):
                max_mean.name = f'max_sub{j}'
                min_mean.name = f'min_sub{j}'
                max_df = pd.concat([max_df, max_mean], axis=1)
                min_df = pd.concat([min_df, min_mean], axis=1)
            stdate = [max_df.index[0].strftime('%Y%m%d')]
            if os.path.exists(os.path.join(output_wd, nm, 'TMP')):
                shutil.rmtree(os.path.join(output_wd, nm, 'TMP'), ignore_errors=True)

            print(f'writing model {nm} to indivisual file ... processing')      
            
            for i in tqdm(range(len(max_df.columns))):
                maxs_ = max_df.iloc[:, i].map(lambda x: '{:.2f}'.format(x)).tolist()
                mins_ = min_df.iloc[:, i].map(lambda x: '{:.2f}'.format(x)).tolist()
                ab = [f'{mx},{mi}' for mx, mi in zip(maxs_, mins_)]
                abf = stdate + ab

                os.makedirs(os.path.join(output_wd, nm, 'TMP'), exist_ok=True)
                os.chdir(os.path.join(output_wd, nm, 'TMP'))
                with open(f'TMP_{i+1:03d}.txt', 'w') as fp:
                    for l in abf:
                        fp.write(l+"\n")
