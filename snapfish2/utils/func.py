import warnings
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.integrate import quad

__all__ = [
    "overlap",
    "loop_overlap",
    "all_possible_pairs",
    "sample_covar_ma"
]

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


def sample_covar_loh(X:np.ndarray) -> np.ndarray:
    """Naive covariance estimator when data is incomplete.

    Parameters
    ----------
    X : (N, R) np.ndarray
        Data matrix with NaN entries.

    Returns
    -------
    (R, R) np.ndarray
        Covariance estimate.
    """
    Avail = ~np.isnan(X)
    p_hat = np.mean(Avail)
    N = X.shape[0]

    Y = np.where(np.isnan(X), 0, X)
    G = (Y.T@Y)/(N*p_hat**2)
    # the diagonal should be divided by p, not p^2
    np.fill_diagonal(G, np.diag(G)*p_hat)
    return G


def sample_covar_ma(X:np.ndarray) -> np.ndarray:
    """Raw covariance matrix with masked arrays.

    Parameters
    ----------
    X : (N, R) np.ndarray
        Data matrix with NaN entries.

    Returns
    -------
    (R, R) np.ndarray
        Covariance estimate.
    """
    Avail = (~np.isnan(X)).astype("int")
    num_avail = Avail.T@Avail

    Xma = np.ma.masked_invalid(X)
    G = np.ma.dot(Xma.T, Xma)/num_avail
    return G.data
    

def _sample_covar_heteroPCA(
        G:np.ndarray, r:int=5, T:int=100
) -> np.ndarray:
    """Estimate covariance with HeteroPCA.

    Parameters
    ----------
    G : (R, R) np.ndarray
        Sample covariance matrix from sample_covar_loh.
    r : int, optional
        Rank of the covariance matrix, by default 5.
    T : int, optional
        Number of iterations, by default 100.

    Returns
    -------
    (R, R) np.ndarray
        HeteroPCA estimate of the covariance matrix.
    """
    Gdiag = np.copy(G)
    # 1st step of HeteroPCA: set diagonal to 0
    np.fill_diagonal(Gdiag, 0)

    for _ in range(T):
        L, V = np.linalg.eig(Gdiag)
        Lkept, Vkept = np.diag(L[:r]), V[:,:r]
        Gest = Vkept@Lkept@Vkept.T
        # impute the diagonal with the new estimate
        np.fill_diagonal(Gdiag, np.diag(Gest))
    
    return Gest


def _matrixA(R:int) -> np.ndarray:
    """Linear transformation that maps the raw covariance matrix to the 
    covariance matrix of data with shifts subtracted.

    Parameters
    ----------
    R : int
        Dimension of the covariance matrix.

    Returns
    -------
    (R^2, R^2) np.ndarray
        The calculated linear transformation.
    """
    s = np.zeros(R*R)
    s[:R] = 1
    s = np.stack([np.roll(s, i*R) for i in range(R)])
    s1 = np.tile(s, (R, 1))
    s2 = np.repeat(s, R, axis=0)
    A = np.identity(R*R) + 1/R**2 - s1/R - s2/R
    return A


def _machenko_pastur_pdf(x, g, s) -> float:
    hp = (1 + g**.5)**2
    hm = (1 - g**.5)**2
    if x <= s*hm or x >= s*hp:
        return 0
    f = ((x - s*hm)*(s*hp - x))**.5/(2*np.pi*s*x*min(g, 1))
    return f


def _machenko_pastur_cdf(x, g, s) -> float:
    warnings.filterwarnings("ignore")
    F = quad(
        _machenko_pastur_pdf, 
        a=-np.inf, b=x, args=(g, s), 
        limit=500
    )[0]
    warnings.filterwarnings("default")
    return F


def _machenko_pastur_quantile(k, g, s) -> float:
    if k < 0 or k > 1:
        raise ValueError("Quantile must be between 0 and 1.")
    # Slightly increase the search range to resolve numerical issues
    u = (s*(1 + g**.5)**2) * 1.0001
    l = (s*(1 - g**.5)**2) * 0.9999
    while True:
        val = (u+l)/2
        q = _machenko_pastur_cdf(val, g, s)
        if np.abs(q - k) < 1e-6:
            return val
        elif q > k:
            u = val
        else:
            l = val


def _bema_var_hat(lambdas, p, n, alpha):
    pt = min(p, n)
    l = int(np.ceil(alpha * pt))
    u = int(np.floor((1 - alpha) * pt))
    qks = np.array([
        _machenko_pastur_quantile(1-k/pt, g=p/n, s=1) 
        for k in range(l, u+1)
    ])
    return np.sum(qks*lambdas[l:u+1])/np.sum(np.square(qks))
    