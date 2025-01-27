from typing import Tuple
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils import cast_to_distmat
from ..analysis.loop import LoopTestAbstract


def pairwise_heatmap(
    X1:np.ndarray,
    X2:np.ndarray=None,
    ax:plt.axis=None,
    x:str=None, 
    y:str=None, 
    title:str=None, 
    cmap:str="seismic_r", 
    **args
):
    """Single heatmap or two heatmaps in the upper and the lower 
    triangle regions.

    Parameters
    ----------
    X1 : np.ndarray
        First heatmap data, passed to `cast_to_distmat`.
    X2 : np.ndarray, optional
        Second data, passed to `cast_to_distmat`, by default None.
    ax : plt.axis, optional
        Axis to plot the heatmap, by default None.
    x : str, optional
        Label for the first heatmap, by default None.
    y : str, optional
        Label for the second heatmap, by default None.
    title : str, optional
        Title of the plot, by default None.
    cmap : str, optional
        Colormap, by default "seismic_r".
    """
    val1 = cast_to_distmat(X1).copy()
    if X2 is not None:
        val2 = cast_to_distmat(X2)
        val1.T[np.triu_indices_from(val1)] = val2[np.triu_indices_from(val2)]
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    cmap.set_bad("white")
    ax = sns.heatmap(
        val1,
        cmap=cmap,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
        **args
    )
    ax.xaxis.set_label_position("top")
    ax.set(xlabel=x, ylabel=y, title=title)
    
    
def background_model(
    i:int, j:int, 
    TestClass:LoopTestAbstract,
    bin:int|np.ndarray, 
    num_bins:int=50,
    ax:plt.axis=None
) -> plt.axis:
    """Cartoon visualization of background model.

    Parameters
    ----------
    i : int
        Index of the first locus.
    j : int
        Index of the second locus.
    TestClass : LoopTestAbstract
        The TestClass whose background model will be visualized.
    bin : int | np.ndarray
        Either a number or an array. If a number, `bin` is resolution.
        Otherwise, need an array of 1D genomic locations (d1d).
    num_bins : int, optional
        Required if bin is an integer. Number of loci to plot, by 
        default 50.
    ax : plt.axis, optional
        plt.axis object, by default None.

    Returns
    -------
    plt.axis
        plt.axis object.
    """
    if isinstance(bin, np.ndarray):
        d1d = bin
    else:
        d1d = np.arange(num_bins, dtype="int64") * int(bin)
    bkgd = TestClass.ij_background(i, j, d1d).astype("float")
    bkgd[i, j] = 0.5
    ax = sns.heatmap(
        bkgd,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        ax=ax
    )
    return ax


def compare_loops(
    df1:pd.DataFrame, df2:pd.DataFrame, 
    map1:str, map2:str,
    ax:plt.axis=None, 
    eval_func:callable=np.mean
) -> plt.axis:
    """Plot two sets of loops together.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first list of loops to plot on the upper right triangle. 
        Must contain "c1", "s1", "e1", "c2", "s2", "e2" as columns.
    df2 : pd.DataFrame
        The second list of loops to plot on the lower left triangle. 
        Must contain "c1", "s1", "e1", "c2", "s2", "e2" as columns.
    map1 : str
        Label for the first set of loops.
    map2 : str
        Label for the second set of loops.
    ax : plt.axis, optional
        plt.axis object, by default None.
    eval_func : callable, optional
        Where to plot the two loci of each loop, by default np.mean, 
        which will plot the mean of `s` and `e`. Can also be np.min, 
        which will plot `s`; or np.max, which will plot `e`.

    Returns
    -------
    plt.axis
        plt.axis object.
    """
    df1, df2 = df1.copy(), df2.copy()
    df1["val1"] = eval_func(df1[["s1", "e1"]], axis=1)
    df1["val2"] = eval_func(df1[["s2", "e2"]], axis=1)
    df2["val1"] = eval_func(df2[["s1", "e1"]], axis=1)
    df2["val2"] = eval_func(df2[["s2", "e2"]], axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(df1, x="val2", y="val1", s=3, ax=ax)
    sns.scatterplot(df2, x="val1", y="val2", s=2, ax=ax)
    ax.set(xlabel="1D genomic location", ylabel=map2, yticks=[])
    ax.invert_yaxis()
    ax2 = ax.twiny()
    ax2.set(xlabel=map1, xticks=[])
    ax.set_box_aspect(1)

    ax.spines["left"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax.spines["bottom"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    
    return ax


def add_boundary(
    ax:plt.axis, 
    boundary:np.array,
    cmap:plt.cm=plt.get_cmap("Set3"),
    label:dict=None,
    **kwargs
):
    if label is None:
        label = {k:k for k in pd.unique(boundary)}
    pos_dict = {t:[] for t in label}
    for i, r in enumerate(boundary):
        if i == 0 or r != boundary[i-1]:
            pos_dict[r].append([i, i+1])
        else:
            pos_dict[r][-1][1] = i+1

    colors = {t:cmap(i) for i, t in enumerate(pos_dict.keys())}
    legends = []
    for k, v in pos_dict.items():
        legends.append(Line2D([0], [0], color=colors[k], lw=3, label=label[k]))
        if k >= 0:
            for s, e in v:
                ax.plot([s,e,e,s,s], [s,s,e,e,s], color=colors[k], **kwargs)
    ax.legend(handles=legends, loc="lower left", labelcolor="linecolor")
    
    
def plot_TAD_boundary(
    result:pd.DataFrame, 
    bedpe:pd.DataFrame, 
    cols:list, 
    tad_colors:list, 
    line_colors:list,
    line_lim:dict | None=None,
    diff:bool=False,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot TAD boundaries along with scores.

    Parameters
    ----------
    result : pd.DataFrame
        Output from :func:`snapfish2.analysis.domain.TADCaller.by_pval` 
        or :func:`snapfish2.analysis.domain.TADCaller.by_insulation`.
    bedpe : pd.DataFrame
        Output from :func:`snapfish2.analysis.domain.TADCaller.to_bedpe`
        by inputting `result`.
    cols : list
        A list of columns in result to plot alongside the boundaries. 
        Can be stat, pval, and fdr for results called by p-values, or 
        insulation for results called by insulation scores.
    tad_colors : list
        A list of colors at least of the same length as the number of 
        unique levels in `bedpe`. This will be the TAD boundary colors.
    line_colors : list
        A list of colors at least of the same length as `cols`.
    line_lim : dict | None, optional
        The xlim of scores in `cols`, by default None.
    diff : bool, optional
        Whether to show the p-values and FDR from differential TAD 
        calling, by default False. If True, have to replace `bedpe` by
        result from :func:`snapfish2.analysis.domain.DiffRegion.diff_region`.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        fig, axes.
    """
    result = result.reset_index(drop=True)
    bedpe = bedpe.reset_index(drop=True)
    
    fig, axes = plt.subplots(
        1, len(cols) + 1, width_ratios=[3.6] + [.5]*len(cols),
        figsize=(3.6 + .5*len(cols), 3), sharey=True
    )
    if len(cols) == 0:
        axes = [axes]

    if "level" not in bedpe:
        pos_ls = result[result["peak"]].index.values
        for i in range(len(pos_ls)+1):
            s = pos_ls[i-1] if i != 0 else 0
            e = pos_ls[i] if i != len(pos_ls) else len(result)
            axes[0].plot(
                [s,e,e,s,s], [s,s,e,e,s], 
                color=tad_colors[0], **kwargs
            )
            if diff:
                text = f"Pval={row["pval"]:.2f}\nFDR={row["fdr"]:.2f}"
                axes[0].text(
                    e, s+.1, text, fontdict={"fontsize":8},
                    horizontalalignment="right", verticalalignment="top"
                )
    else:
        for lvl in np.unique(bedpe.level)[::-1]:
            df = bedpe[bedpe["level"]==lvl]
            for _, row in df.iterrows():
                s = row.idx1
                # Plotting correction for last index
                if row.idx2 != len(result) - 1:
                    e = row.idx2
                else:
                    e = row.idx2 + 1
                axes[0].plot(
                    [s,e,e,s,s], [s,s,e,e,s], 
                    color=tad_colors[lvl], **kwargs
                )
                if diff:
                    text = f"Pval={row["pval"]:.2f}\nFDR={row["fdr"]:.2f}"
                    axes[0].text(
                        e, s+.1, text, fontdict={"fontsize":8}, color="k",
                        horizontalalignment="right", verticalalignment="top"
                    )
        pos_ls = pd.unique(np.concatenate(bedpe[["idx1", "idx2"]].values))

    for i, col in enumerate(cols):
        sns.lineplot(
            x=result[col], y=result.index, color=line_colors[i], 
            orient="y", ax=axes[i+1]
        )
        # axes[i+1].xaxis.tick_top()
        axes[i+1].xaxis.set_label_position("top")
        
        if line_lim is not None and col in line_lim:
            axes[i+1].set_xlim(line_lim[col])
        xmin, xmax = axes[i+1].get_xlim()
        axes[i+1].hlines(pos_ls, xmin=xmin, xmax=xmax, linestyles="--")
        axes[i+1].set(ylabel="", xlabel=col.title())
        axes[i+1].spines[["right"]].set_visible(True)
    
    warnings.filterwarnings("ignore")
    fig.tight_layout(pad=-fig.get_size_inches()[1]/10)
    return fig, axes


def plot_AB_bars(cpmt_arr:np.ndarray, ax:plt.Axes):
    ax_divider = make_axes_locatable(ax)
    ax.spines[["left", "bottom", "right","top"]].set_visible(True)

    cax1 = ax_divider.append_axes("right", size="5%", pad="0%", sharey=ax)
    cax1.barh(np.where(cpmt_arr==0)[0], width=1, height=1, color="r")
    cax1.spines[["right","top"]].set_visible(True)
    cax1.set(xticks=[], xlabel="A")

    cax2 = ax_divider.append_axes("right", size="5%", pad="0%", sharey=ax)
    cax2.barh(np.where(cpmt_arr==1)[0], width=1, height=1, color="b")
    cax2.spines[["right","top"]].set_visible(True)
    cax2.set(xticks=[], xlabel="B")