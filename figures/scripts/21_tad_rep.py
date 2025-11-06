"""Error Replication Analysis for TADs

Perform error replication analysis by adding various levels of Gaussian noise
along the z-axis to the original data, and calling TADs on the simulated data. Repeat the
simulation multiple times for each error level.

To run this script, 
1. Change the `wkdir` variable to your working directory.
2. Change the output directory `out_dire` variable to your desired output path.
3. Submit jobs for each chromosome (1-19) using a loop, e.g.,
   ```bash
   for c in {1..19}; do
       sbatch -J err_rep_chr${c} \
        -o /path/to/logs/err_rep_chr${c}.out \
        -e /path/to/logs/err_rep_chr${c}.err \
        --wrap="python /path/to/21_tad_rep.py ${c}"
   done
   ```
4. After all jobs are completed, combine the results using:
   ```bash
   sbatch -J err_rep_combine \
    -o /path/to/logs/err_rep_combine.out \
    -e /path/to/logs/err_rep_combine.err \
    --wrap="python /path/to/21_tad_rep.py c"    
"""
wkdir = "/proj/yunligrp/users/hongyuyu/ArcFISH"
out_dire = f"{wkdir}/output"

import os
import sys
import pandas as pd
import numpy as np
import anndata as ad

import arcfish as af


if __name__ == "__main__":
    p21 = os.path.join(out_dire, "chromosome21.h5ad")
    
    r = int(sys.argv[1])
    caller = af.tl.TADCaller(tree=False, method="pval", window=3e5)
    rep_path = os.path.join(out_dire, "tad_replicate_21_{}.csv")
    if r >=0 :
        np.random.seed(100 + r)
        reps = []
        for chr_id in ["chr21"]:
            adata = ad.read_h5ad(p21)
            for n in [100, 200, 400, 800, 1600, 3200, 6400]:
                sub_adata = adata[adata.obs.sample(n=n).index].copy()
                # Renormalize based on the subset
                del sub_adata.varp["var_X"]
                df = caller.call_tads(sub_adata)
                df["num_cells"] = n
                df["replicate"] = r
                reps.append(df)
        reps = pd.concat(reps, ignore_index=True)
        reps.to_csv(rep_path.format(r), index=False)
    
    else:  # combine results
        adata = ad.read_h5ad(p21)
        res = caller.call_tads(adata)
        res["num_cells"] = adata.shape[0]
        res["replicate"] = -1
        reps = [res]
        
        for f in os.listdir(out_dire):
            if f.startswith("tad_replicate_21_") and f.endswith(".csv"):
                rep = pd.read_csv(os.path.join(out_dire, f))
                reps.append(rep)
                os.remove(os.path.join(out_dire, f))
                
        reps = pd.concat(reps, ignore_index=True)
        reps.to_csv(rep_path.format("all"), index=False)