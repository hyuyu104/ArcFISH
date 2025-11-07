# Workflow

This page describes the workflow of developing the package on the server. I assume the current working directory is `/proj/yunligrp/users/hongyuyu/AxisWiseTest`. This is also the place where I put all SnapFISH2-related files.

To prepare your own directory, please copy the followings to your working directory:
```sh
cd /YOUR/WKDIR
cp -R /proj/yunligrp/users/hongyuyu/AxisWiseTest/snapfish2 .
cp /proj/yunligrp/users/hongyuyu/AxisWiseTest/pyproject.toml .
cp /proj/yunligrp/users/hongyuyu/AxisWiseTest/README.md .
cp /proj/yunligrp/users/hongyuyu/AxisWiseTest/LICENSE .
mkdir data
cp /proj/yunligrp/users/hongyuyu/AxisWiseTest/data/data.json data/data.json
cp /proj/yunligrp/users/hongyuyu/AxisWiseTest/data/zenodo.json data/zenodo.json
mkdir docs
cp -R /proj/yunligrp/users/hongyuyu/AxisWiseTest/docs/source docs/source
cp /proj/yunligrp/users/hongyuyu/AxisWiseTest/docs/Makefile docs/Makefile
cp /proj/yunligrp/users/hongyuyu/AxisWiseTest/docs/make.bat docs/make.bat
cp -R /proj/yunligrp/users/hongyuyu/AxisWiseTest/figures .
```
and follow the steps below.

## Local installation on longleaf

On the longleaf server, load python environment and install `snapfish2` locally:
```sh
$ module load anaconda
# /PYTHON/ENV/PATH is where the python environment will be created
# Change this to your own directory to create your environment
$ conda create --prefix /PYTHON/ENV/PATH python==3.12.2
$ conda activate /PYTHON/ENV/PATH

# cd to the directory with pyproject.toml file
$ ls
> ... pyproject.toml ...
# Install snapfish2 locally
$ python -m pip install -e ".[docs]"
```
After these steps, the package should be installed in the environment `/PYTHON/ENV/PATH`. Note this is an editable installation, meaning that one does not need to re-install the package everytime the source code is changed.

## Data download

After this step, all the required data should be downloaded and stored at `data`. The multiplexed FISH data are from the following studies:

1. Takei, Y. et al. Single-cell nuclear architecture across cell types in the mouse brain. Science 374, 586–594 (2021).

2. Takei, Y. et al. Integrated spatial genomics reveals global architecture of single nuclei. Nature 590, 344–350 (2021).

3. Su, J.-H., Zheng, P., Kinrot, S. S., Bintu, B. & Zhuang, X. Genome-Scale Imaging of the 3D Organization and Transcriptional Activity of Chromatin. Cell 182, 1641-1659.e26 (2020).

I also used some ChIP-seq and ChIA-PET data from [ENCODE](https://www.encodeproject.org) to benchmark different methods. The data are summarized in two files: `data/data.json` and `data/zenodo.json`. To download all data in these two files, first create a new access key for 4DN following [this instruction](https://data.4dnucleome.org/help/user-guide/downloading-files) and save the key at `~/keypairs.json`. Then submit a SLURM job by `sbatch` with the following lines:
```sh
$ module load anaconda
$ conda activate /PYTHON/ENV/PATH
$ python figures/utils/data.py -d data
```
All data files should be ready after the job finishes running.

## Running jupyter notebooks on server with VSCode

The jupyter notebooks to generate figures are in the `figures` directory. To run the notebooks on server, need to request an interactive node and initialize jupyter with the python environment created. The following shows how to do this in VSCode.
1. Setup the interactive node:
```sh
$ srun -N 1 -n 1 -t 8:00:00 --mem=32g -p interact --pty /bin/bash
# Wait until the job has been allocated resources
$ hostname # print the hostname at this interactive node
> c1131.ll.unc.edu
# Open a jupyter notebook
$ module load anaconda
$ conda activate /PYTHON/ENV/PATH
$ python -m pip install jupyter
# Later in ipynb, the default directory is pwd
# So open jupyter in a place you want
$ jupyter notebook

>    To access the server, open this file in a browser:
>        file:///nas/longleaf/home/hongyuyu/.local/share/jupyter/runtime/jpserver-951736-open.html
>    Or copy and paste one of these URLs:
>        http://localhost:8888/tree?token=2bd5214506dc5db706e490da659d4a64f3591779803b6ae9
>        http://127.0.0.1:8888/tree?token=2bd5214506dc5db706e490da659d4a64f3591779803b6ae9
```
Install jupyter extension in VSCode: in extensions (Ctrl+Shift+X), search and install "Jupyter". Sometimes it shows installed but it is installed locally. Remember to open the server first in VSCode and go to extension, press **install in longleaf.unc.edu**.

**Note**: do not close the terminal. Otherwise the interactive node will be closed as well.

**Note**: the directory where `jupyter notebook` is executed will be the working directory of opened `.ipynb` files.

2. Setup the ssh command in vscode: Shift+Cmd+P, "Remote-SSH: Connet to Host", "Configure SSH Hosts", "/Users/redfishhh/.ssh/config", add the following to the `config` file:
```sh
Host b* c* g* t*
  HostName %h
  User hongyuyu
  ProxyJump longleaf.unc.edu
```
This ensures when `ssh` to the node, it is first `ssh` to the longleaf server, then run `ssh` to the node **on the server**. Otherwise VSCode will try to connect directly to `c1131.ll.unc.edu`.

3. ssh to the node: Shift+Cmd+P, "Remote-SSH: Connet to Host", connect to `c1131.ll.unc.edu`. Now we are in the interactive node. Open any `.ipynb` file, press "select kernel" on the upper right corner, "existing jupyter server", paste any of the last two urls in step 1.

**Note**: do not directly select the python environment when selecting kernel; otherwise the program is not using the node, it is running in the shell.

After setting up the jupyter kernel, the `.ipynb` file can be run on the server.

## Changing source code and documentation pages

The source code in the `snapfish2` folder can be edited directly. Some remarks on `snapfish2/__init__.py`:
```python
sys.modules.update({
    f"{__name__}.{m}": globals()[m]
    for m in ["pl", "tl", "loop", "domain"]
})
```
This explicitly adds the module names `snapfish2.pl`, `snapfish2.tl`, etc. Sphinx will not compile properly without explicitly adding the module names.
```python
__version__ = "2.0.0"
```
This is the version name. `pyproject.toml` will also read the version from this variable.

For the documentation pages, add or modify `md` or `rst` files in `docs/source`. To generate the sphinx documentation:
```sh
$ cd docs
# Remove previous rst files generated by autodoc
$ rm -r source/api/generated
$ make clean && make html
```
Note `autodoc` is run simultaneously so no need to add `sphinx-apidoc -o source/generated ../snapfish2`.