"""Error Replication Analysis

Perform error replication analysis by adding various levels of Gaussian noise
along the z-axis to the original data, and calling loops and TADs on the
simulated data. Repeat the simulation multiple times for each error level.

To run this script, 
1. Change the `wkdir` variable to your working directory.
2. Change the output directory `out_dire` variable to your desired output path.
3. Submit jobs for each chromosome (1-19) using a loop, e.g.,
   ```bash
   for c in {1..19}; do
       sbatch -J err_rep_chr${c} \
        -o /path/to/logs/err_rep_chr${c}.out \
        -e /path/to/logs/err_rep_chr${c}.err \
        --wrap="python /path/to/11_err_rep.py ${c}"
   done
   ```
4. After all jobs are completed, combine the results using:
   ```bash
   sbatch -J err_rep_combine \
    -o /path/to/logs/err_rep_combine.out \
    -e /path/to/logs/err_rep_combine.err \
    --wrap="python /path/to/11_err_rep.py c"    
"""

wkdir = "/proj/yunligrp/users/hongyuyu/ArcFISH"
out_dire = f"{wkdir}/123ACElog/072525"

import os, sys
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

import arcfish as af


def err_rep_chr(adata, n=800, rep=10, errs=[0, 50, 100, 150, 200]):
    np.random.seed(1)

    loop_res = []
    domain_res = []

    for z_err in errs:
        err = {"X": 0, "Y": 0, "Z": z_err}

        for r in range(rep):
            sub_adata = ad.AnnData(
                obs=np.arange(n, dtype="int64"),
                var=adata.var,
                layers={c: adata[
                    adata.obs.sample(n=n, replace=True).index
                ].layers["X"] for c in "XYZ"},
                uns=adata.uns
            )
            for c in "XYZ":
                e = stats.norm.rvs(size=sub_adata.shape, scale=err[c])
                # sub_adata.layers[c] -= np.nanmean(sub_adata.layers[c], axis=1)[:,None]
                sub_adata.layers[c] += e
                
            af.pp.filter_normalize(sub_adata)
            
            loop_caller1 = af.tl.LoopCaller(pval_cutoff=1, gap=50e3, ltclass=af.tl.TwoSampleT)
            loop_caller2 = af.tl.LoopCaller(gap=50e3, ltclass=af.tl.AxisWiseF)

            domain_caller1 = af.tl.TADCaller(tree=False, method="insulation", 
                                            prominence=0.04, distance=2)
            domain_caller2 = af.tl.TADCaller(tree=False, method="pval", fdr_cutoff=.1)

            loop_res1 = loop_caller1.to_bedpe(loop_caller1.call_loops(sub_adata), sub_adata)
            loop_res1 = loop_res1[loop_res1["summit"]]
            loop_res1["method"] = "SnapFISH"
            loop_res1["rep"] = r
            loop_res1["err"] = z_err
            loop_res2 = loop_caller2.to_bedpe(loop_caller2.call_loops(sub_adata), sub_adata)
            loop_res2 = loop_res2[loop_res2["summit"]]
            loop_res2["method"] = "ArcFISH"
            loop_res2["rep"] = r
            loop_res2["err"] = z_err
            loop_res.extend([loop_res1, loop_res2])

            domain_res1 = domain_caller1.to_bedpe(domain_caller1.call_tads(sub_adata))
            domain_res1["method"] = "Insulation"
            domain_res1["rep"] = r
            domain_res1["err"] = z_err
            domain_res2 = domain_caller2.to_bedpe(domain_caller2.call_tads(sub_adata))
            domain_res2["method"] = "ArcFISH"
            domain_res2["rep"] = r
            domain_res2["err"] = z_err
            domain_res.extend([domain_res1, domain_res2])
            
    loop_res = pd.concat(loop_res, ignore_index=True)
    domain_res = pd.concat(domain_res, ignore_index=True)
    return loop_res, domain_res


if __name__ == "__main__":
    a = sys.argv[1]
    
    if a == "c":
        loops, domains = [], []
        for c in range(1, 20):
            c = f"chr{c}"
            loops.append(pd.read_csv(f"{out_dire}/loop_err_rep_{c}.tsv", sep="\t"))
            os.remove(f"{out_dire}/loop_err_rep_{c}.tsv")
            domains.append(pd.read_csv(f"{out_dire}/domain_err_rep_{c}.tsv", sep="\t"))
            os.remove(f"{out_dire}/domain_err_rep_{c}.tsv")
        pd.concat(loops, ignore_index=True).to_csv(f"{out_dire}/loop_err_rep_all.tsv", 
                                                   sep="\t", index=False)
        pd.concat(domains, ignore_index=True).to_csv(f"{out_dire}/domain_err_rep_all.tsv",
                                                      sep="\t", index=False)
    else:
        loader = af.pp.FOF_CT_Loader({
            "rep1": f"{wkdir}/data/takei_nature_2021/4DNFIHF3JCBY.csv",
            "rep2": f"{wkdir}/data/takei_nature_2021/4DNFIQXONUUH.csv",
        }, voxel_ratio={c: 1000 for c in "XYZ"})

        chrom = f"chr{a}"
        adata = loader.create_adata(chrom)
        
        loop_res, domain_res = err_rep_chr(adata, n=800, rep=10, errs=[0, 50, 100, 150, 200])
        
        loop_res.to_csv(f"{out_dire}/loop_err_rep_{chrom}.tsv", sep="\t", index=False)
        domain_res.to_csv(f"{out_dire}/domain_err_rep_{chrom}.tsv", sep="\t", index=False)