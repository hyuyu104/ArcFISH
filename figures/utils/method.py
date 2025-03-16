import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import snapfish2

from .. import figure1


def prepare_data_takei_25kb(data_dire="data/takei_nature_2021"):
    rep_paths = ["4DNFIHF3JCBY.csv", "4DNFIQXONUUH.csv"]
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


def sample_maps(mfr, k=5):
    chr_id = "chr3"
    axes = ["x", "y", "z"]
    out = os.path.join(figure1.FIG1OUT, "sample_maps")
    if not os.path.exists(out):
        os.mkdir(out)
    for i in range(k):
        trace = mfr[chr_id, i][["X", "Y", "Z"]].values
        for j in range(trace.shape[1]):
            fig, ax = plt.subplots(figsize=(3, 3))
            snapfish2.plot.pairwise_heatmap(
                np.abs(trace[:15,[j]] - trace[:15,j]),
                cbar=False, linewidth=1, linecolor="black",
                cmap="magma", ax=ax
            )
            trace_id = pd.unique(mfr[chr_id]["Trace_ID"])[i]
            fig.savefig(
                os.path.join(out, f"{axes[j]}_{chr_id}_{trace_id}.pdf"), 
                bbox_inches="tight", 
                pad_inches=.02, 
                facecolor="black"
            )
            plt.clf()
            
            
def tad_sample():
    tad_mat = np.zeros((15, 15), dtype="int64")
    tad_mat[:5,:5] = 1
    tad_mat[7:10,7:10] = 1
    tad_mat[11:14,11:14] = 1
    fig, ax = plt.subplots(figsize=(3, 3))
    snapfish2.plot.pairwise_heatmap(tad_mat, cbar=False, cmap="Reds", ax=ax)
    fig.savefig(
        os.path.join(figure1.FIG1OUT, f"sample_tad.pdf"), 
        bbox_inches="tight", 
        pad_inches=.02, 
        facecolor="black"
    )
    plt.clf()
    
    out = os.path.join(figure1.FIG1OUT, "sample_pvals")
    if not os.path.exists(out):
        os.mkdir(out)
    np.random.seed(10)
    for i in range(6):
        mat = np.random.uniform(0, 1, (15, 15))
        fig, ax = plt.subplots(figsize=(3, 3))
        snapfish2.plot.pairwise_heatmap(
            mat,
            cbar=False, linewidth=1, linecolor="black",
            cmap="viridis", ax=ax
        )
        fig.savefig(
            os.path.join(out, f"{i}_pvals.pdf"), 
            bbox_inches="tight", 
            pad_inches=.02, 
            facecolor="black"
        )
        plt.clf()