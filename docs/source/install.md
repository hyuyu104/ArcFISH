# Installation

## Create an environment and install SnapFISH2

It is recommended to first create a new Python environment and then install `SnapFISH2`. This can be accomplished either by `conda` or Python's `venv`:
::::{tabs}
:::{group-tab} conda
```sh
conda create --name snapfish2_env python==3.12.2
conda activate snapfish2_env
python -m pip install snapfish2
```
:::

:::{group-tab} venv
```sh
python -m venv snapfish2_env
source .snapfish2_env/bin/activate
python -m pip install snapfish2
```
:::
::::

## Local installation for development

To install the latest version on github and the dependencies to build the sphinx documentation:
::::{tabs}
:::{group-tab} conda
```sh
conda create --name snapfish2_env python==3.12.2
conda activate snapfish2_env
git clone https://github.com/hyuyu104/snapfish2
cd snapfish2
python -m pip install -e ".[docs]"
```
:::

:::{group-tab} venv
```sh
python -m venv snapfish2_env
source .snapfish2_env/bin/activate
git clone https://github.com/hyuyu104/snapfish2
cd snapfish2
python -m pip install -e ".[docs]"
```
:::
::::
This will create an editable install, meaning that all modifications made on the source code will be reflected immediately without requiring re-installing the package. To compile the sphinx documentation, `cd docs` and then run `make clean && make html`.