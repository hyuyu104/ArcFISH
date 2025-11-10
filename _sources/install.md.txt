# Installation

## Python Environment Management

Before installing ArcFISH, it's highly recommended to create an isolated Python environment. This prevents conflicts between different packages and ensures reproducible installations. You can use either **conda** or **venv** for this purpose:

### Conda
**Conda** is a popular package and environment manager that comes with Anaconda or Miniconda. It can manage packages from multiple languages (not just Python) and handles complex dependencies efficiently.

**Installation:**
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight) or [Anaconda](https://www.anaconda.com/products/distribution) (full distribution)
- Conda provides pre-compiled packages, which can be faster to install and more reliable across different systems

### Python venv
**venv** is Python's built-in virtual environment tool. It's lightweight and comes standard with Python 3.3+, requiring no additional installation.

**Features:**
- Creates isolated Python environments using only pip for package management
- Smaller footprint and faster environment creation
- Uses the same Python version as your system installation

## Create an environment and install ArcFISH

Choose one of the following methods to create a new Python environment and install `ArcFISH`:
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