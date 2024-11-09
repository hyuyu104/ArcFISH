import os
import numpy as np
import pandas as pd
import snapfish2


def takei_to_mulfish(data_dire, rep_paths):
    mfrs = []
    for i, rep_path in enumerate(rep_paths):
        path = os.path.join(data_dire, rep_path)
        mfrs.append(snapfish2.MulFish(path))
        # Extract FOV ID
        mfrs[-1].data["FOV"] = mfrs[-1].data["Cell_ID"].str.extract(r"(\d+)\_\d+")
        # Add replicate ID to Trace_ID and as a separate column
        mfrs[-1].data["Trace_ID"] = f"{i}_" + mfrs[-1].data["Trace_ID"]
        mfrs[-1].data["Replicate"] = f"rep{i}"
        # Convert voxel coordinates to nm
        mfrs[-1].data["X"] *= 103
        mfrs[-1].data["Y"] *= 103
        mfrs[-1].data["Z"] *= 250
    concat_df = pd.concat([m.data for m in mfrs])
    mfr = snapfish2.MulFish(concat_df)
    return mfr


def to_covars(data):
    coor_cols = ["X", "Y", "Z"]
    pivoted = data.pivot_table(
        index="Chrom_Start", 
        columns="Trace_ID", 
        values=coor_cols,
        sort=False
    )
    covs = []
    for c in coor_cols:
        X = pivoted[c].values.T
        X = X - np.nanmean(X, axis=1)[:,None]
        cov = snapfish2.sample_covar_ma(X)
        covs.append(cov)
    return np.stack(covs)