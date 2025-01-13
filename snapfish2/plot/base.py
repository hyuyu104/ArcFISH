import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from ..utils.load import cast_to_distmat


def trace(X, fig=None, label=False):
    if isinstance(X, pd.DataFrame):
        if "locus" in X.columns:
            newidx = np.arange(X["locus"].min(), X["locus"].max(), 1, dtype="int")
            X = X.set_index("locus").reindex(newidx)
        X = X[["X", "Y", "Z"]].values
    
    segments = []
    id_ls = []
    na_flag = True
    for i, x in enumerate(X):
        if np.all(~np.isnan(x)):
            if na_flag:
                segments.append([x])
                id_ls.append([i])
            else:
                segments[-1].append(x)
                id_ls[-1].append(i)
            na_flag = False
        else:
            na_flag = True
    segments = [np.array(t).reshape(-1, 3).T for t in segments]
    
    if fig is None:
        fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="3d")
    for ids, segment in zip(id_ls, segments):
        ax.plot(*segment, ".-")
        if label:
            for text, pos in zip(ids, segment.T):
                # just label 1/10 points to avoid overlapping
                if np.random.random() < 0.1:
                    ax.text(*pos, s=text)
    ax.set(xlabel="x (nm)", ylabel="y (nm)", zlabel="z (nm)")
    return fig, ax


def daxis_dist(
    d1d, 
    arrs, 
    xlab="1D genomic distance",
    ylab="Difference in axis",
    hue="Axis",
    fig=None
):
    mat1d = d1d[:,None] - d1d[None,:]
    mat1d_flat = mat1d[np.tril_indices_from(mat1d, -1)]
    arrs_flat = arrs[:,:,*np.tril_indices_from(mat1d, -1)]
    plt_df = []
    ax_names = [r"$x$-axis", r"$y$-axis", r"$z$-axis"]
    for d in range(arrs_flat.shape[1]):
        df = pd.DataFrame(
            arrs_flat[:,d,:].T, 
            index=mat1d_flat    
        ).reset_index(names=xlab)
        df = df.melt(id_vars=xlab, value_name=ylab, var_name=hue).dropna()
        df[hue] = ax_names[d]
        plt_df.append(df)
    plt_df = pd.concat(plt_df)
    
    if fig is None:
        fig, axes = plt.subplots(1, 2, figsize=(7, 2))
    else:
        axes = fig.subplots(1, 2)
    
    sns.pointplot(
        plt_df, 
        x=xlab, 
        y=ylab, 
        hue=hue, 
        errorbar="se",  # std/sqrt(n)
        linewidth=2,  # main line width
        markersize=3,  # size of the points
        err_kws={"linewidth": 1.5},  # line width of error bars
        native_scale=True,  # continuous instead of categorical
        ax=axes[0]
    )
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[0].set(title="Mean of axis difference")
    axes[0].legend().remove()
    
    plt_grp = plt_df.groupby([hue, xlab]).var().reset_index()
    sns.pointplot(
        plt_grp, 
        x=xlab, 
        y=ylab, 
        hue=hue, 
        errorbar=None,
        linewidth=2,  # main line width
        markersize=3,  # size of the points
        native_scale=True,  # continuous instead of categorical
        ax=axes[1]
    )
    axes[1].legend(loc="upper left")
    axes[1].set(title="Variance of axis difference", ylabel="Variance")
    
    return fig, plt_df