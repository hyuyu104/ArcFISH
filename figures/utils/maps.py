import os
import warnings
import tables
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns

import snapfish2 as sf


def tad_sample(out_dire):
    tad_mat = np.zeros((15, 15), dtype="int64")
    tad_mat[:5,:5] = 1
    tad_mat[7:10,7:10] = 1
    tad_mat[11:14,11:14] = 1
    fig, ax = plt.subplots(figsize=(3, 3))
    sf.pl.pairwise_heatmap(tad_mat, cbar=False, cmap="Reds", ax=ax)
    fig.savefig(
        os.path.join(out_dire, f"sample_tad.pdf"), 
        bbox_inches="tight", 
        pad_inches=.02, 
        facecolor="black"
    )
    plt.clf()
    
    out = os.path.join(out_dire, "sample_pvals")
    if not os.path.exists(out):
        os.mkdir(out)
    np.random.seed(10)
    for i in range(6):
        mat = np.random.uniform(0, 1, (15, 15))
        fig, ax = plt.subplots(figsize=(3, 3))
        sf.pl.pairwise_heatmap(
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


def dist_diff_heatmaps(axes, adata, vmax1=60, vmax2=150):
    """Four heatmaps in one row."""
    coor_names = ["X", "Y", "Z"]
    X = np.stack([adata.layers[c] for c in coor_names])
    arr = X[:,:,:,None] - X[:,:,None,:]
    med_diff = np.nanmedian(np.abs(arr), axis=1)
    for i, axis in enumerate(coor_names):
        sf.pl.pairwise_heatmap(
            med_diff[i], ax=axes[i], 
            vmin=0, vmax=vmax1, 
            cmap="RdBu",
            rasterized=True,
            title=f"{axis}-Axis Median Difference"
        )
    dist_mat = np.nanmedian(np.sqrt(np.sum(np.square(arr), axis=0)), axis=0)
    sf.pl.pairwise_heatmap(
        dist_mat, vmax=vmax2, cmap="seismic_r",
        rasterized=True,
        ax=axes[3], title="Euclidean Distance (nm)"
    )
    

def venn_subsets(raw, noised):
    overlapped = np.sum(sf.tl.loop_overlap(
        noised[noised.summit], 
        raw[raw.summit], offset=25e3
    ).overlapped.values == 3)
    
    subsets = (
        np.sum(raw.summit) - overlapped, 
        np.sum(noised.summit) - overlapped,
        overlapped
    )
    return subsets


def venn3_loops(hicexpl, fithic2, testloo, lbl, title, ax):
    l23 = sf.tl.loop_overlap(hicexpl, fithic2, 25e3)
    l23 = l23[l23["overlapped"]==3]
    l123 = sf.tl.loop_overlap(testloo, l23, 25e3)
    l123 = np.sum(l123["overlapped"]==3)

    l23 = len(l23) - l123

    l12 = sf.tl.loop_overlap(testloo, fithic2, 25e3)
    l12 = np.sum(l12["overlapped"]==3) - l123

    l13 = sf.tl.loop_overlap(testloo, hicexpl, 25e3)
    l13 = np.sum(l13["overlapped"]==3) - l123

    l1 = len(testloo) - l12 - l13 - l123
    l2 = len(fithic2) - l12 - l23 - l123
    l3 = len(hicexpl) - l13 - l23 - l123

    venn = venn3(
        subsets=(l1, l2, l12, l3, l13, l23, l123),
        set_labels=(lbl, "FitHiC2", "HiCExplorer"),
        ax=ax, alpha=1, 
        set_colors=sns.palettes.color_palette("pastel")[:3]
    )
    for text in venn.set_labels:
        text.set_fontsize(sf.settings.fontsize)
    ax.set_ylim(-.8, .8)
    ax.set_title(title, fontsize=sf.settings.fontsize, y=0.9)
    

def loop_map(loader, chr_id, fithic2, hicexpl, res_sf1, res_sf2):
    adata = loader.create_adata(chr_id)
    med_dist = sf.pp.median_pdist(adata, inplace=False)
    d1d = adata.var.mean(axis=1).values
    x, y = x, y = list(map(lambda x: x.flatten(), np.meshgrid(d1d, d1d)))
    hm_df = pd.DataFrame({"x": x, "y": y, "dist": med_dist.flatten()})
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    for ax in axes:
        sns.scatterplot(hm_df, x="x", y="y", hue="dist", ax=ax, alpha=1,
                        palette="RdBu", s=5, marker="s", linewidth=0)
        ax.set(xlim=(d1d.min(), d1d.max()), ylim=(d1d.min(), d1d.max()))
    sf.pl.compare_loops(fithic2, res_sf1[res_sf1["final"]], chr_id, 
                        "FitHiC2", "SnapFISH", ax=axes[0])
    sf.pl.compare_loops(hicexpl, res_sf1[res_sf1["final"]], chr_id, 
                        "HiCExplorer", "SnapFISH", ax=axes[1])
    sf.pl.compare_loops(fithic2, res_sf2[res_sf2["final"]], chr_id, 
                        "FitHiC2", "SnapFISH2", ax=axes[2])
    sf.pl.compare_loops(hicexpl, res_sf2[res_sf2["final"]], chr_id, 
                        "HiCExplorer", "SnapFISH2", ax=axes[3])
    for ax in axes:
        ax.get_legend().remove()
    return fig