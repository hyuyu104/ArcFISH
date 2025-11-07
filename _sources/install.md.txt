# Installation

## Create an environment and install ArcFISH

It is recommended to first create a new Python environment and then install `ArcFISH`. This can be accomplished either by `conda` or Python's `venv`:
::::{tabs}
:::{group-tab} conda
```sh
conda create --name arcfish_env python==3.12.2
conda activate arcfish_env
python -m pip install arcfish
```
:::

:::{group-tab} venv
```sh
python -m venv arcfish_env
source .arcfish_env/bin/activate
python -m pip install arcfish
```
:::
::::

## Local installation for development

To install the latest version on github and the dependencies to build the sphinx documentation:
::::{tabs}
:::{group-tab} conda
```sh
conda create --name arcfish_env python==3.12.2
conda activate arcfish_env
git clone https://github.com/hyuyu104/arcfish
cd arcfish
python -m pip install -e ".[docs]"
```
:::

:::{group-tab} venv
```sh
python -m venv arcfish_env
source .arcfish_env/bin/activate
git clone https://github.com/hyuyu104/arcfish
cd arcfish
python -m pip install -e ".[docs]"
```
:::
::::
This will create an editable install, meaning that all modifications made on the source code will be reflected immediately without requiring re-installing the package. To compile the sphinx documentation, `cd docs` and then run `make clean && make html`.