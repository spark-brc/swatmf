import os
from tqdm import tqdm

def rename_files(src_dir, dst_dir):
    # find all files
    src_list = os.listdir(src_dir)
    for src in tqdm(src_list):
        os.rename(
            os.path.join(src_dir, src), 
            os.path.join(dst_dir, f'ppwb03_{int(src[5:-4]):03d}.jpg'))

if __name__ == '__main__':
    src_dir = "E:/Study/PEST/2022_The Maths Behind PEST and PEST++/day03/uncertainty_analysis_spark"
    dst_dir = "E:/Study/PEST/2022_The Maths Behind PEST and PEST++/day03/temp"
    rename_files(src_dir, dst_dir)

