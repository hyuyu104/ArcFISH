# SnapFISH2: An integrative analysis tool for multiplexed FISH

> This is the index page. To view the full html documentation, download `docs/build` and open `docs/build/html/index.html` in any browser.

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