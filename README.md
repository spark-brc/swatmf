
[<img src="https://img.shields.io/badge/LABEL-MESSAGE-COLOR.svg?logo=LOGO">](<LINK>)
[<img src="https://img.shields.io/pypi/v/swatmf?color=blue"](<LINK>)



[![](https://zenodo.org/badge/304147230.svg)](https://zenodo.org/badge/latestdoi/304147230)
[![PyPI - License](https://img.shields.io/pypi/l/swatmf)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI Version](https://img.shields.io/pypi/v/swatmf?color=blue)](https://pypi.python.org/pypi/swatmf)


# Uncertainty Analysis for APEX model

## Get data and jupyter notebooks
You essentially have 2 options:

#### - Easy way
- [Download the data zip file](https://github.com/spark-brc/apex-ua/archive/refs/heads/main.zip)
- Unzip `apex-ua-main.zip` to a prefered location.

#### - Hard way (Dev mode)  
- You will need to install Git if you don’t have it installed already. Downloads are available at [the link](https://git-scm.com/download). On windows, be sure to select the option that installs command-line tools  
- For Git, you will need to set up SSH keys to work with Github. To do so:
    - Go to GitHub.com and set up an account
    - On Windows, open Git Bash (on Mac/Linux, just open a terminal) and set up ssh keys if you haven’t already. To do this, simply type ssh-keygen in git bash/terminal and accept all defaults (important note - when prompted for an optional passphrase, just hit return.)  
- Follow the [instructions](https://help.github.com/articles/adding-a-new-ssh-key-to-your-github-account/) to set up the SSH keys with your GitHub account.
- Clone the materials from GitHub.
    - Open a git bash shell from the start menu (or, on a Mac/Linux, open a terminal)
    - Navigate to the folder you made to put the course materials
    - Clone the materials by executing the following in the git bash or terminal window:    

    ```bash
    git clone https://github.com/spark-brc/apex-ua.git
    ```  
        
## Installation
To execute jupyter notebook, we need the Miniconda environment.

#### 1. Miniconda Python:
- If you don't already have conda installed, please download Miniconda for your operating system from https://conda.io/en/latest/miniconda.html (choose the latest version for your operating system, 64-bit). You should not need elevated rights to install this.
- Run the installer and select "only my user" when prompted. This will allow you to work with your python installation directly.

#### 2. Set Environment and install libraries:
- After installation, go to the START menu and select "Miniconda Prompt" to open a DOS box.
- Type the following command:
```bash
conda install -c conda-forge mamba
```  

- Using the [cd](https://www.computerhope.com/issues/chusedos.htm) command in the Miniconda DOS box, navigate to the location where you have `environment.yml` the file and type: 
```bash
mamba env create -f environment.yml
``` 
and hit ENTER.

After your virtual environment setup is complete, change the environment to `apex-ua`:  
```bash
conda activate apex-ua
```  
- Launch jupyter notebook 
```bash
jupyter notebook
```

A browser window with a Jupyter notebook instance should open. Yay!
