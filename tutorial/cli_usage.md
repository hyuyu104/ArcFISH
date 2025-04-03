# SnapFISH2 as a command line tool

Here we demonstrate how to use SnapFISH2 as a command line tool. Make sure `snapfish2` is successfully installed and activate the environment with SnapFISH2 in shell:
::::{tabs}
:::{group-tab} conda
```sh
>>> conda activate snapfish2_env
>>> python -m pip show snapfish2
Name: snapfish2
Version: ...
```
:::

:::{group-tab} venv
```sh
>>> source .snapfish2_env/bin/activate
>>> python -m pip show snapfish2
Name: snapfish2
Version: ...
```
:::
::::

## Loop/domain/compartment calling

Here we use the DNA seqFISH+ data from [Takei et al., 2021](https://www.science.org/doi/10.1126/science.abj1966) to call chromatin loops, TADs, and A/B compartments. The data can be downloaded from the 4DN data portal with ID: 4DNFIW4S8M6J (biological replicate 1), 4DNFI4LI6NNV (biological replicate 2), and 4DNFIDUJQDNO (biological replicate 3).

Download the data and place them like the following:
```sh
..
├── data
│   ├── takei_science_2021
│   │   ├── 4DNFIW4S8M6J.csv
│   │   ├── 4DNFI4LI6NNV.csv
│   │   └── 4DNFIDUJQDNO.csv
├── output
```

### Preprocessing

First run the following to preprocess the data and store them as `AnnData` objects:
```sh
snapfish2 preprocess \
    -i "rep1::data/takei_science_2021/4DNFIW4S8M6J.csv,\
        rep2::data/takei_science_2021/4DNFI4LI6NNV.csv,\
        rep3::data/takei_science_2021/4DNFIDUJQDNO.csv" \
    -o pp_chr_adata \
    -v X::103,Y::103,Z::250
```
The `-v` argument specifies how to convert the voxel coordinates to coordinates in nm. For Takei et al data, this ratio is 103 for X and Y axes and 250 for Z axis. `-v` can be omitted if the data in already in nm.

This step creates a directory `pp_chr_adata` with a `.h5ad` file for each chromosome. The processed data will be used in all following steps.

### Loop calling

Once the processed data are stored in `pp_chr_adata`, loop calling can be done by
```sh
snapfish2 loop -i pp_chr_adata -o output/loop_res.bedpe
```
which stores the loop calling result to `output/loop_res.bedpe`.

If desired, loop calling parameters can be changed by adding additional arguments:

1. `fdr`: FDR cutoff to define loop candidates, by default 0.1.
2. `pval`: p-value cutoff to filter loop summits, by default 1e-5.
3. `lo`: minimum loop length to be considered, by default 100Kb.
4. `up`: maximum loop length to be considered, by default 1Mb.
5. `gap`: candidates within `gap` away are treated as from the same summit, by default 50Kb.
6. `outer`: local background size, by default 50Kb.

For example, to test loops with range from 10Kb to 2Mb, change the command to
```sh
snapfish2 loop -i pp_chr_adata -o output/loop_res.bedpe -lo 10000 -up 1000000
```

### Domain calling

To call TADs, run the following
```sh
snapfish2 domain -i pp_chr_adata -o output/domain_res.bedpe
```
The domain calling result is in `output/domain_res.bedpe`.

Additional optional arguments for TAD calling are:

1. `fdr`: FDR cutoff to define TAD boundaries, by default 0.1.
2. `window`: region size to consider intra/inter-domain contacts, by default 100Kb.
3. `tree`: whether to return hierarchical TADs, by default True.
4. `min`: minimum TAD size, by default 0.

### Compartment calling

To call A/B compartments, run the following
```sh
snapfish2 cpmt -i pp_chr_adata -o output/cpmt_res.bedpe
```
An optional `-min` argument can be passed in to specify the minimum A/B compartment size, by default 0.

```{note}
The compartment calling result here is not very meaningful as only a 1.5Mb (60 loci at 25Kb resolution) region is imaged on each chromosome. This is mainly to demonstrate the compartment calling workflow.
```

## Differential loop/domain/compartment