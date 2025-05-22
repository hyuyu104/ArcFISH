import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import precision_recall_curve

import snapfish2 as sf


CMAP = {"SnapFISH": "darkorange", "SnapFISH2": "darkgreen"}


def chiapet_count_pl(chiapet_path:Path, fig:plt.figure):
    chiapet_df = pd.read_csv(chiapet_path, sep="\t", header=None)
    chiapet_df.columns = ["c1", "s1", "e1", "c2", "s2", "e2", "count"]
    x, y1, y2 = "ChIA-PET Count", "# of Loops", "Fraction of Loops"
    axes = fig.subplots(1, 2)
    plt_df = chiapet_df["count"].value_counts().to_frame(y1).reset_index()\
        .sort_values("count", ascending=False)
    plt_df[y2] = plt_df[y1].cumsum()/np.sum(plt_df[y1])
    sns.lineplot(plt_df, x="count", y=y1, ax=axes[0], color="grey", alpha=.8)
    axes[0].set(xscale="log", xlabel=x, box_aspect=1)
    sub_df = plt_df[plt_df["count"] <= 10]
    axes[1].bar(sub_df["count"], sub_df[y2], color="grey", alpha=.8)
    axes[1].set(xlabel=x, ylabel=y2, box_aspect=1, axisbelow=True)

def roc_curve_pl(df:pd.DataFrame, fig:plt.figure, title:str):
    """ROC curve from roc_df."""
    gs = GridSpec(3, 3, figure=fig)
    ax = fig.add_subplot(gs[:,:2])
    
    fpr = 0.1
    # Add the point with FPR=fpr
    df = df.copy()
    for meth, sdf in df.groupby("Method"):
        if not fpr in sdf.FPR:
            row1 = sdf[sdf.FPR - fpr < 0].iloc[-1]
            row2 = sdf[sdf.FPR - fpr > 0].iloc[0]
            k = (row2.TPR - row1.TPR)/(row2.FPR - row1.FPR)
            b = row1.TPR - k*row1.FPR
            idx = sdf[sdf.FPR - fpr < 0].index[-1]+1
            df.iloc[-1] = [fpr, k*fpr+b, meth]
    df = df.groupby("Method", sort=False)[df.columns].apply(
        lambda df: df.sort_values("FPR")
    ).reset_index(drop=True)
    
    sub_df = df[df["FPR"] <= fpr]
    sns.lineplot(
        sub_df, 
        x="FPR", y="TPR", hue="Method", 
        palette=CMAP, 
        ax=ax
    )
    ax.plot([0, fpr], [0, fpr], "--k", label="Chance")
    ax.get_legend().remove()
    for k, sdf in sub_df.groupby("Method", sort=False):
        ax.fill_between(sdf.FPR, sdf.TPR, alpha=0.1, color=CMAP[k])
    handles, labels = ax.get_legend_handles_labels()
    yticks = np.linspace(0, ax.get_ylim()[1].round(1), 3, endpoint=True)
    ax.set(xticks=[0, 0.05, 0.1], yticks=yticks, box_aspect=1)
    ax = fig.add_subplot(gs[:2,2])
    sns.lineplot(
        df, 
        x="FPR", y="TPR", hue="Method", 
        palette=CMAP, 
        ax=ax
    )
    ax.set_box_aspect(1)
    ax.plot([0, 1], [0, 1], "--k")
    ax.get_legend().remove()
    ax.spines[["right", "top"]].set_visible(True)
    ax.set(xlim=(0,1), ylim=(0,1), xticks=[0,0.5,1], yticks=[0,0.5,1])
    warnings.filterwarnings("ignore", ".*layout.*")
    plt.tight_layout()

    ax_legend = fig.add_subplot(gs[2,2])
    ax_legend.axis("off")  # Turn off axis for legend
    ax_legend.legend(handles, labels, bbox_to_anchor=(-.2,-2))
    
    fig.suptitle(title, y=1.05, fontsize=10)
    
    
def precision_recall_loop(res1, res2, true_df, ax):
    plt_df = []
    for i, res_sf in enumerate([res1, res2]):
        res_df = res_sf[res_sf["summit"]].copy()
        res_df["overlapped"] = sf.tl.loop_overlap(
            res_df, true_df, offset=-1
        )["overlapped"]
        df = pd.DataFrame(np.stack(precision_recall_curve(
            res_df["overlapped"]==3, -res_df["pval"]
        )[:2]).T, columns=["Precisoin", "# True Loops"])
        # Display the actual number of overlapped loops
        df["# True Loops"] *= np.sum(res_df["overlapped"]==3)
        df["Method"] = ["SnapFISH", "SnapFISH2"][i]
        plt_df.append(df)
    plt_df = pd.concat(plt_df, ignore_index=True)
    sns.lineplot(
        plt_df, x="# True Loops", y="Precisoin", hue="Method", 
        palette=CMAP, ax=ax, errorbar=None, linewidth=2
    )
    ax.set(ylim=(0,1))
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(True)

    res_df = res2[res2["summit"]].copy()
    res_df["overlapped"] = sf.tl.loop_overlap(
        res_df, true_df, offset=-1
    )["overlapped"]
    pt_df = []
    for cut in [1e-3, 1e-4, 1e-5, 1e-6]:
        sub_df = res_df[res_df["pval"] < cut]
        xval = (sub_df["overlapped"]==3).sum()
        yval = xval/len(sub_df)
        pt_df.append([cut, xval, yval])
    pt_df = pd.DataFrame(pt_df, columns=["cut", "x", "y"])

    ax.vlines(pt_df["x"], 0, pt_df["y"], color="darkgreen", linestyle="--")
    for _, row in pt_df.iterrows():
        ax.text(row["x"], row["y"]+.05, f"{row["cut"]:.0e}", 
                fontsize=sf.settings.fontsize*.8)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="best")
    
    
def domain_chipseq_barplot(enrich_df, ax):
    sns.barplot(
        enrich_df[enrich_df["Method"]!="Average"], x="marker",
        y="frac", hue="Method", ax=ax, width=.6, dodge=True, gap=.2,
        alpha=.8, edgecolor="k", linewidth=1,  
        palette={"SnapFISH": "darkorange", "SnapFISH2": "darkgreen"},
    )
    for con in ax.containers:
        ax.bar_label(con, labels=con.datavalues.round(2), fontsize=8, padding=1)
    for i, r in enrich_df[enrich_df["Method"]=="Average"].iterrows():
        ax.plot([-.3+i, .3+i], [r["frac"], r["frac"]], ".--k", label="t")

    n1 = enrich_df[enrich_df["Method"]=="SnapFISH"]["total"].iloc[0]
    n2 = enrich_df[enrich_df["Method"]=="SnapFISH2"]["total"].iloc[0]
    label_dict = {
        "SnapFISH": f"Insulation Scores ({n1} Boundaries)",
        "SnapFISH2": f"SnapFISH2 ({n2} Boundaries)",
        "t": "Average (Boundary & Non-Boundary)"
    }
    handles, labels = ax.get_legend_handles_labels()
    labels = [label_dict[t] for t in labels[:3]]
    ax.legend(handles=handles[:3], labels=labels, loc="best")
    ax.set(xlabel=None, ylim=(0,1), ylabel="Enrichment Boundary Fraction")
    ax.grid(False)