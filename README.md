SnapFISH2: An integrative analysis tool for multiplexed FISH
============================================================

> To view the full html documentation, download `docs/build` and open `docs/build/html/index.html` in any browser.

## Local installation on server

On the longleaf server, load python environment and install `snapfish2` locally:
```sh
module load anaconda
# /PYTHON/ENV/PATH is where the python environment will be created
# Change this to your own directory to create your environment
conda create --prefix /PYTHON/ENV/PATH python==3.12.2
conda activate /PYTHON/ENV/PATH

# cd to the directory with pyproject.toml file
ls
> ... pyproject.toml ...
# Install snapfish2 locally
python -m pip install -e .
```
After these steps, the package should be installed in the environment `/PYTHON/ENV/PATH`.

## Data download

If you plan to use the longleaf server to do analysis, all the data can be found at `/proj/yunligrp/users/hongyuyu/AxisWiseTest/data`. You can create a soft link as well so that the `data` folder is visible in your working directory: `ln -s /proj/yunligrp/users/hongyuyu/AxisWiseTest/data /proj/yunligrp/users/YOUR/WKDIR/`.

If you plan to download all the data, the file `data/data.json` lists the ID for each data from ENCODE or 4DN data portal. To automatically download all data, first need to create a new access key for 4DN.

> From [https://data.4dnucleome.org/help/user-guide/downloading-files](https://data.4dnucleome.org/help/user-guide/downloading-files): to create a new access key, first log in to the data portal, then click on your account in the upper right and click on Profile from the dropdown menu. There will be a button near the bottom of the page to add an access key. Save the key and secret; typically this is done by creating a file in your home directory called keypairs.json with the following contents/format (replacing the x’s with the appropriate key and secret, of course). Run `vim keypairs.json` in terminal to open the keypairs file. Enter
```json
{
    "default": {
        "key": "XXXXXXXX",
        "secret": "XXXXXXXXXX",
        "server": "https://data.4dnucleome.org"
    }
}
```
Then activate the python environment and download all data from ENCODE and 4DN (remember to submit a job if on the server, see `/proj/yunligrp/users/hongyuyu/AxisWiseTest/run.sh` for an example script):
```sh
module load anaconda
conda activate /PYTHON/ENV/PATH
python /proj/yunligrp/users/hongyuyu/AxisWiseTest/figures/utils/data.py -d /WHERE/TO/PUT/DATA
```

## Workflow

The jupyter notebooks to generate figures are in `/proj/yunligrp/users/hongyuyu/AxisWiseTest/figures`. To run the notebooks on server, need to request an interactive node and initialize jupyter with the python environment created. The following shows how to do this in VSCode.

1. Setup the interactive node:
```sh
srun -N 1 -n 1 -t 8:00:00 --mem=32g -p interact --pty /bin/bash
# Wait until the job has been allocated resources
hostname # print the hostname at this interactive node
> c1131.ll.unc.edu
# Open a jupyter notebook
module load anaconda
conda activate /PYTHON/ENV/PATH
python -m pip install jupyter
# Later in ipynb, the default directory is pwd
# So open jupyter in a place you want
jupyter notebook

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
This ensures when ssh to the node, it is first ssh to the longleaf server, then run ssh to the node **on the server**. Otherwise VSCode will try to connect directly to `c1131.ll.unc.edu`.

3. ssh to the node: Shift+Cmd+P, "Remote-SSH: Connet to Host", connect to `c1131.ll.unc.edu`. Now we are in the interactive node. Open any `.ipynb` file, press "select kernel" on the upper right corner, "existing jupyter server", paste any of the last two urls in step 1.

**Note**: do not directly select the python environment when selecting kernel; otherwise the program is not using the node, it is running in the shell.

After setting up the jupyter kernel, the `.ipynb` file can be run on the server.