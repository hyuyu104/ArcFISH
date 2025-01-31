Installation
============

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