# ArcFISH: accurate and robust 3D genome feature discovery from chromatin tracing data

ArcFISH is a computational tool to identify chromatin loops, TADs, and A/B compartments from chromatin tracing data, where the chromatin is in-situ hybridized by fluorescence probes and imaged using high-resolution microscopes (please see the [4DN data portal](https://data.4dnucleome.org/resources/data-collections/chromatin-tracing-datasets) for a collection of available datasets).

::::{grid} 2
:::{grid-item}
:columns: 9

ArcFISH is designed to handle the heterogenous measurement error in different axes. The workflow involves:

1. Estimate a weight for each axis
2. Filter our outliers and perform 1D bias normalization
3. Process each axis separately:
   - Loops: F-test based on local background
   - TADs: F-test to compare inter-region and intra-region interactions
   - A/B compartments: second eigenvector
4. Combine axis level information with weights estimated in step 1
:::
:::{grid-item-card} Contents
:columns: 3
```{toctree}
:maxdepth: 1

install
tutorial/index
api/index
contact
```
:::
::::