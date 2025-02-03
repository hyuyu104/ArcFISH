# SnapFISH2: An integrative analysis tool for multiplexed FISH

::::{grid} 2
:::{grid-item}
:columns: 9
SnapFISH2 is a computational tool to identify chromatin loops, TADs, and A/B compartments from multiplexed DNA-FISH, where the chromatin is in-situ hybridized by fluorescence probes and imaged using high-resolution microscopes (please see the [4DN data portal](https://data.4dnucleome.org/resources/data-collections/chromatin-tracing-datasets) for a collection of available datasets). SnapFISH2 can also be used to call differential loops, TADs, or A/B compartments from two experiment conditions.

DNA-FISH data are stored as [dask arrays](https://docs.dask.org/en/stable/) in SnapFISH2 and thus adapts well with computing clusters and supports larger-than-memory storage. Algorithmically, by processing each imaging axis separately, SnapFISH2 is robust to heterogeneous measurement errors and can output consistent results under different noise levels. 
:::
:::{grid-item-card} Contents
:columns: 3
```{toctree}
:maxdepth: 1

install
tutorial/index
workflow
figures/index
api/index
```
:::
::::