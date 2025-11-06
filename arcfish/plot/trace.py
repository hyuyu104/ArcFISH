import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


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