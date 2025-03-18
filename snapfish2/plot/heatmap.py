from typing import Tuple
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from anndata import AnnData

from .utils import cast_to_distmat, rotate_df
from ..utils.eval import median_pdist, filter_normalize
from ..tools.loop import LoopTestAbstract
from ..tools.domain import TADCaller


def pairwise_heatmap(
    X1:np.ndarray,
    X2:np.ndarray=None,
    ax:plt.Axes=None,
    x:str=None, 
    y:str=None, 
    title:str=None, 
    cmap:str="RdBu", 
    **args
):
    """Single heatmap or two heatmaps in the upper and the lower 
    triangle regions.

    Parameters
    ----------
    X1 : np.ndarray
        First heatmap data, passed to 
        :func:`snapfish2.pl.cast_to_distmat`.
    X2 : np.ndarray, optional
        Second data, passed to 
        :func:`snapfish2.pl.cast_to_distmat`, by default None.
    ax : plt.Axes, optional
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
    ax:plt.Axes=None
) -> plt.Axes:
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
    ax : plt.Axes, optional
        plt.Axes object, by default None.

    Returns
    -------
    plt.Axes
        plt.Axes object.
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


def triangle_heatmap(
    mat:np.ndarray, 
    df1d:pd.DataFrame, 
    cut_hi:None|float=None,
    cmap:str|mcolors.Colormap="RdBu",
    fig:None|plt.Figure=None,
    width:float=3,
    height:None|float=None,
    alpha:float=0.8,
    xticklabels:None|list[str]=None
) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Plot a heatmap in the upper right triangle of a 2D matrix.
    
    `fig`, `width`, `height`:
    
    1. If `fig` is not None, no need to pass in `width` and `height`.
    
    2. If `width` is not None, no need to pass in `fig`. In this case, 
    if `height` is None, infer `height` from `width` and `cut_hi`. If
    `height` is not None, use `height` as the height of the figure.

    Parameters
    ----------
    mat : np.ndarray
        p by p matrix to plot.
    df1d : pd.DataFrame
        1D genomic location information. Must contain "Chrom_Start" and
        "Chrom_End" as columns. Can first call 
        :func:`snapfish2.pp.FOF_CT_Loader.create_adata` to 
        create an adata object and then use the `var` field as `df1d`.
    cut_hi : None | float, optional
        Only pixels with 1D genomic distance below `cut_hi` will be 
        plotted. Plot all pixels if `cut_hi` is None, by default None.
    cmap : str | mcolors.Colormap, optional
        Color map used, can be either a string or a Colormap object
        created by :func:`~matplotlib:matplotlib.pyplot.get_cmap`, by 
        default "RdBu".
    fig : None | plt.Figure, optional
        Figure object to plot the heatmap, by default None.
    width : float, optional
        Width of the figure, by default 3.
    height : None | float, optional
        Height of the figure, by default None.
    alpha : float, optional
        Transparency of pixels, by default 0.8.
    xticklabels : None | list[str], optional
        List of length 2, xtick labels. If None, set the xtick labels
        as 1D genomic locations with unit Mb, by default None.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes, plt.Axes]
        Figure, main ax, and colorbar ax.
    """
    d1d = ((df1d["Chrom_End"] + df1d["Chrom_Start"])/2).values
    x, y = np.meshgrid(d1d, d1d)
    uidx = np.tril_indices_from(x, -1)
    med_df = pd.DataFrame({"x":x[uidx], "y":y[uidx], "dist":mat[uidx]})
    
    med_df = rotate_df(med_df, -45)
    
    frac = 1
    if cut_hi is not None and cut_hi < d1d.max() - d1d.min():
        frac = cut_hi/(d1d.max() - d1d.min())
        med_df = med_df[np.abs(med_df["x"] - med_df["y"]) <= cut_hi]
        
    if fig is not None:
        ax = fig.subplots()
    elif height is None:
        fig, ax = plt.subplots(figsize=(width, width/2*frac))
    else:
        fig, ax = plt.subplots(figsize=(width, height))
        
    sns.scatterplot(
        med_df, x="x_rot", y="y_rot", hue="dist", alpha=alpha,
        ax=ax, marker="d", palette=cmap, linewidth=0, s=width*15
    )
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.set(
        yticks=[], ylabel="", xlabel="Genomic Position (bp)",
        xticks=[med_df["x_rot"].min(), med_df["x_rot"].max()],
        xticklabels=[
            f"{df1d["Chrom_Start"].min()/1e6:.3f}Mb", 
            f"{df1d["Chrom_End"].max()/1e6:.3f}Mb"
        ]
    )
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.xaxis.set_label_coords(0.5, -.05)
    # Show the top of the triangle if frac is 1
    if frac == 1:
        ax.set(ylim=(med_df["y_rot"].min(), ax.get_ylim()[1]))
    else:
        ax.set(ylim=(med_df["y_rot"].min(), med_df["y_rot"].max()))
    ax.grid(False)
    ax.legend().remove()
    
    norm = mcolors.Normalize(
        vmin=med_df["dist"].min(), 
        vmax=med_df["dist"].max()
    )
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Needed for colorbar
    cbar = fig.colorbar(sm, ax=ax, aspect=10, pad=0.02, fraction=0.1/width)
    
    return fig, ax, cbar


def compare_loops(
    df1:pd.DataFrame, df2:pd.DataFrame, 
    chr_id:str|None=None,
    map1:str="", map2:str="",
    c1:str="r", c2:str="b",
    ax:plt.Axes=None, 
    eval_func:callable=np.mean
) -> plt.Axes:
    """Plot two sets of loops together.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first list of loops to plot on the upper right triangle. 
        Must contain "c1", "s1", "e1", "c2", "s2", "e2" as columns.
    df2 : pd.DataFrame
        The second list of loops to plot on the lower left triangle. 
        Must contain "c1", "s1", "e1", "c2", "s2", "e2" as columns.
    chr_id : str, optional
        Chromosome ID, by default None.
    map1 : str
        Label for the first set of loops.
    map2 : str
        Label for the second set of loops.
    ax : plt.Axes, optional
        plt.Axes object, by default None.
    eval_func : callable, optional
        Where to plot the two loci of each loop, by default np.mean, 
        which will plot the mean of `s` and `e`. Can also be np.min, 
        which will plot `s`; or np.max, which will plot `e`.

    Returns
    -------
    plt.axis
        plt.axis object.
    """
    if chr_id is not None:
        df1 = df1[df1["c1"]==chr_id]
        df2 = df2[df2["c1"]==chr_id]
    df1, df2 = df1.copy(), df2.copy()
    df1["val1"] = eval_func(df1[["s1", "e1"]], axis=1)
    df1["val2"] = eval_func(df1[["s2", "e2"]], axis=1)
    df2["val1"] = eval_func(df2[["s1", "e1"]], axis=1)
    df2["val2"] = eval_func(df2[["s2", "e2"]], axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    sns.scatterplot(df1, x="val2", y="val1", s=5, linewidth=0, ax=ax, color=c1)
    sns.scatterplot(df2, x="val1", y="val2", s=5, linewidth=0, ax=ax, color=c2)
    ax.invert_yaxis()
    ax.set_box_aspect(1)
    
    # The ax is already initialized with plots
    if ax.get_xlim() != (0, 1):
        xmin, xmax = ax.get_xlim()
    else:
        xmin = min(df1["val1"].min(), df2["val1"].min(), 
                   df1["val2"].min(), df2["val2"].min())
        xmax = max(df1["val1"].max(), df2["val1"].max(), 
                   df1["val2"].max(), df2["val2"].max())
        offset = (xmax - xmin) * 0.05
        ax.set(xlim=(xmin-offset, xmax+offset), 
               ylim=(xmax+offset, xmin-offset))

    ax.spines[["left", "bottom", "right", "top"]].set_visible(True)
    
    ax.set(
        xlabel="1D Position", xticks=[xmin, xmax],
        ylabel=None, yticks=[],
        xticklabels=[f"{xmin/1e6:.3f}Mb", f"{xmax/1e6:.3f}Mb"]
    )
    ax.xaxis.set_label_coords(0.5, -.05)
    ax.grid(False)
    
    fontsize = ax.xaxis.label.get_size()
    (x1, x2), (y1, y2) = ax.get_xlim(), ax.get_ylim()
    ax.text(
        x1 + 0.03*(x2-x1), y1 + 0.03*(y2-y1), map2, fontsize=fontsize,
        verticalalignment="bottom", horizontalalignment="left",
        path_effects=[pe.withStroke(linewidth=.5, foreground="white")]
    )
    ax.text(
        x2 - 0.03*(x2-x1), y2 - 0.03*(y2-y1), map1, fontsize=fontsize,
        verticalalignment="top", horizontalalignment="right",
        path_effects=[pe.withStroke(linewidth=.5, foreground="white")]
    )
    
    return ax


def domain_boundary(
    adata:AnnData, 
    caller:TADCaller, 
    ax:plt.Axes|None=None
):
    if "var_X" not in adata.varp:
        filter_normalize(adata)
    res = caller.call_tads(adata)
    res_df = caller.to_bedpe(res)

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.3, 2))
    if caller.method == "pval":
        col, ylabel = "fdr", "FDR"
    elif caller.method == "insulation":
        col, ylabel = "insulation", "Score"

    med_dist = median_pdist(adata, inplace=False)
    pairwise_heatmap(med_dist, ax=ax)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(True)

    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("bottom", size="10%", pad="0%")
    c = sns.palettes.color_palette("dark")
    sns.lineplot(y=res[col], x=res.index, color=c[0], ax=cax)
    ylim = cax.get_ylim()
    ylim = (ylim[0]-(ylim[1]-ylim[0])*.2, ylim[1]+(ylim[1]-ylim[0])*.2)

    pos_ls = pd.unique(np.concatenate(res_df[["idx1", "idx2"]].values))[1:-1]
    cax.vlines(pos_ls, linestyles="--", color=c[2], ymin=ylim[0], ymax=ylim[1])

    cax.set(xlabel=None, ylabel=ylabel, ylim=ylim, xticks=[], yticks=[])
    cax.spines[["top", "right", "bottom", "left"]].set_visible(True)
    cax.grid(False)

    for _, row in res_df.iterrows():
        s = row.idx1
        # Plotting correction for last index
        if row.idx2 != len(res) - 1:
            e = row.idx2
        else:
            e = row.idx2 + 1
        ax.plot(
            [s,e,e,s,s], [s,s,e,e,s],
            color=c[2] if row.level == 0 else c[0]
        )

    
def plot_TAD_boundary(
    result:pd.DataFrame, 
    bedpe:pd.DataFrame, 
    cols:list, 
    ax:plt.Axes,
    tad_colors:list, 
    line_colors:list,
    line_names:list=None,
    diff:bool=False,
    rotation:float=0,
    size:str="8%",
    **kwargs
) -> list[plt.Axes]:
    """Plot TAD boundaries on a heatmap and a list of scores on the left
    of the heatmap.

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
    ax : plt.Axes
        Heatmap ax to add TAD boundaries and scores.
    tad_colors : list
        A list of colors at least of the same length as the number of 
        unique levels in `bedpe`. This will be the TAD boundary colors.
    line_colors : list
        A list of colors at least of the same length as `cols`.
    line_names : list, optional
        Same length as `cols`. Displayed names of `cols`. If None, will
        use `cols` as `line_names`, by default None.
    diff : bool, optional
        Whether to show the p-values and FDR from differential TAD 
        calling, by default False. If True, have to replace `bedpe` by
        result from 
        :func:`snapfish2.analysis.domain.DiffRegion.diff_region`.
    rotation : float, optional
        Rotation of score labels, by default 0.
    size : str, optional
        The width of score columns, by default `8%`.

    Returns
    -------
    list[plt.Axes]
        Score column axes.
    """
    result = result.reset_index(drop=True)
    bedpe = bedpe.reset_index(drop=True)
    
    ax.spines[["left", "bottom", "right","top"]].set_visible(True)

    for lvl in np.unique(bedpe.level)[::-1]:
        df = bedpe[bedpe["level"]==lvl]
        for _, row in df.iterrows():
            s = row.idx1
            # Plotting correction for last index
            if row.idx2 != len(result) - 1:
                e = row.idx2
            else:
                e = row.idx2 + 1
            ax.plot(
                [s,e,e,s,s], [s,s,e,e,s], 
                color=tad_colors[lvl], **kwargs
            )
            if diff:
                text = f"Pval={row["pval"]:.2f}\nFDR={row["fdr"]:.2f}"
                ax.text(
                    e, s+.1, text, fontdict={"fontsize":8}, color="k",
                    horizontalalignment="right", verticalalignment="top"
                )
    pos_ls = pd.unique(np.concatenate(bedpe[["idx1", "idx2"]].values))[1:-1]

    ax_divider = make_axes_locatable(ax)
    caxs = []
    for i, col in enumerate(cols):
        cax = ax_divider.append_axes("left", size=size, pad="0%", sharey=ax)
        sns.lineplot(
            x=result[col], y=result.index, ax=cax,
            orient="y", color=line_colors[i], 
        )
        cax.set(xticks=[])
        if line_names is None:
            cax.set_xlabel(col, rotation=rotation)
        else:
            cax.set_xlabel(line_names[i], rotation=rotation)
        cax.spines[["right","top"]].set_visible(True)
        xmin, xmax = cax.get_xlim()
        
        cax.hlines(
            pos_ls, linestyles="--", color=tad_colors[0], 
            xmin=xmin, xmax=xmax, **kwargs
        )
        caxs.append(cax)
    
    return caxs


def plot_AB_bars(
    cpmt_arr:np.ndarray, 
    ax:plt.Axes,
    size:str="5%",
    ca=sns.palettes.color_palette("dark")[1],
    cb=sns.palettes.color_palette("dark")[0]
):
    """Add A/B compartment assignments to a heatmap.

    Parameters
    ----------
    cpmt_arr : (p,) np.ndarray
        A/B compartment assignments. An array of zeros and ones, where
        zeros are A compartments.
    ax : plt.Axes
        Heatmap ax to add A/B compartment assignments.
    size : str, optional
        The width of A/B compartment column, by default "5%".
    ca : str, optional
        Color of A compartments, by default "r".
    cb : str, optional
        Color of B compartments, by default "b".
    """
    ax_divider = make_axes_locatable(ax)
    ax.spines[["left", "bottom", "right","top"]].set_visible(True)

    cax1 = ax_divider.append_axes("left", size=size, pad="0%", sharey=ax)
    cax1.barh(np.where(cpmt_arr==1)[0], width=1, height=1, color=cb)
    cax1.spines[["right","top"]].set_visible(True)
    cax1.set(xticks=[], xlabel="B")
    
    cax2 = ax_divider.append_axes("left", size=size, pad="0%", sharey=ax)
    cax2.barh(np.where(cpmt_arr==0)[0], width=1, height=1, color=ca)
    cax2.spines[["right","top"]].set_visible(True)
    cax2.set(xticks=[], xlabel="A")