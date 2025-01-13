from itertools import combinations
import numpy as np
import pandas as pd

def overlap(ints1:np.ndarray, ints2:np.ndarray, offset:float=0) -> np.ndarray:
    """Given two list of intervals, for each interval in the first list, check
    if it overlaps with any interval from the second list. If `offset` is not 
    0, overlap is defined as within `offset` length away from each other.

    Parameters
    ----------
    ints1 : (n1, 2) np.ndarray
        A list of intervals. First column is the left end, and the second 
        column is the right end of the interval.
    ints2 : (n2, 2) np.ndarray
        A list of intervals. Same format as `ints1`.
    offset : float, optional
        How to define overlapped intervals, by default 0.

    Returns
    -------
    (n1) np.ndarray
        Whether each interval in the first list overlaps with intervals from
        the second list.
    """
    a = ints1[:,[1]] - ints2[:,0] < offset
    b = ints2[:,1] - ints1[:,[0]] < offset
    return np.any(~(a|b), axis=1)


def loop_overlap(
    test_df:pd.DataFrame|str, 
    true_df:pd.DataFrame|str, 
    offset:float=0
) -> pd.DataFrame:
    """For each locus pair (row) in `test_df`, return 3 if the locus pair also
    presents in `true_df` (pair presents -> both loci overlapped); return 1 if 
    one of the two loci presents in `true_df`; and return 2 if both loci 
    present but they never present in the same row of `true_df`. If `offset` 
    is not 0, overlap is defined as within `offset` away from each other.

    Parameters
    ----------
    test_df : pd.DataFrame | str
        If a DataFrame, must has "c1", "s1", "e1", "c2", "s2", "e2" as column
        names. If a str, will read from the file named `test_df`. The file is
        delimited by tab and either has the column names listed above or has 
        no column names.
    true_df : pd.DataFrame | str
        Same format as `test_df`.
    offset : float, optional
        How to define overlapped intervals, by default 0.

    Returns
    -------
    pd.DataFrame
        Same format and length as `test_df`, with an additional `overlapped` 
        column.
    """
    cols = ["c1", "s1", "e1", "c2", "s2", "e2"]
    
    if isinstance(test_df, str):
        p1 = test_df
        test_df = pd.read_csv(test_df, sep="\t")
        if test_df.columns[0] != cols[0]:
            test_df = pd.read_csv(p1, sep="\t", header=None)
            test_df.columns = cols + test_df.columns[len(cols):].tolist()
    assert np.all(test_df["s2"] - test_df["s1"] >= 0), \
        "Locus 1 should precede locus 2"
    
    if isinstance(true_df, str):
        p2 = true_df
        true_df = pd.read_csv(true_df, sep="\t")
        if true_df.columns[0] != cols[0]:
            true_df = pd.read_csv(p2, sep="\t", header=None)
            true_df.columns = cols + true_df.columns[len(cols):].tolist()
    assert np.all(true_df["s2"] - true_df["s1"] >= 0), \
        "Locus 1 should precede locus 2"
    
    odfs = []
    for (c1, c2), tdf in test_df.groupby(["c1", "c2"], sort=False):
        vdf = true_df[(true_df["c1"]==c1)&(true_df["c2"]==c2)]
        
        a1 = tdf["s1"].values[:,None] - vdf["e1"].values > offset
        b1 = vdf["s1"].values[:,None] - tdf["e1"].values > offset
        a2 = tdf["s2"].values[:,None] - vdf["e2"].values > offset
        b2 = vdf["s2"].values[:,None] - tdf["e2"].values > offset
        
        odf = tdf.copy()[cols]
        l1, l2 = (~(a1|b1.T)), (~(a2|b2.T))
        odf["overlapped"] = np.any(l1&l2, axis=1).astype("int")
        odf["overlapped"] += np.any(l1, axis=1).astype("int")
        odf["overlapped"] += np.any(l2, axis=1).astype("int")
        odfs.append(odf)
    if len(odfs) != 0:
        return pd.concat(odfs)
    return pd.DataFrame(columns=test_df.columns.tolist() + ["overlapped"])


def all_possible_pairs(d1df:pd.DataFrame):
    all_df = []
    for c, df in d1df.groupby("Chrom", sort=False):
        comb_obj = combinations((df[["Chrom_Start", "Chrom_End"]].values), 2)
        vals = np.stack([np.hstack(t) for t in comb_obj])
        df = pd.DataFrame(vals, columns=["s1", "e1", "s2", "e2"])
        df["c1"] = df["c2"] = c
        all_df.append(df)
    all_df = pd.concat(all_df)
    return all_df[["c1", "s1", "e1", "c2", "s2", "e2"]]