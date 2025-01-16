from typing import Callable
import numpy as np
import pandas as pd

class MulFish:
    def __init__(self, input):
        if isinstance(input, str):
            self.read_info(input)
            self.read_data(input)
        if isinstance(input, pd.DataFrame):
            self.data = input
            
    default_cols = ["Chrom", "Trace_ID"]

    def read_info(self, path):
        with open(path, "r") as f:
            info_lines, info_dict = [], {}
            while len(info_lines) == 0 or "#" in info_lines[-1]:
                info_lines.append(f.readline().strip())
            for line in info_lines[:-1]:
                # Cannot simply set start = 0 as some rows start with "#
                start = line.find("#")
                if "##" in line:
                    sep = line.find("=")
                    info_dict[line[start+2:sep]] = line[sep+1:].strip(",")
                else:
                    sep = line.find(": ")
                    info_dict[line[start+1:sep]] = line[sep+2:].strip(",")
        self.info = info_dict

    def read_data(self, path:str, **kwargs):
        self.info["columns"] = self.info["columns"].strip("()").split(",")
        self.info["columns"] = [t.strip() for t in self.info["columns"]]
        data = pd.read_csv(
            path, 
            skiprows=len(self.info),
            header=None,
            names=self.info["columns"]
        )

        table_type = self.info["Table_namespace"]
        if "FOF-CT_core" in table_type:
            self.data = self._read_fof_ct_core(data, **kwargs)
        else:
            self.data = data

    @staticmethod
    def _fill_nan_cols(cols_fill, ref_col, df):
        for col in cols_fill:
            end_map = (
                df[[ref_col, col]]
                .dropna() # exclude NaN rows in the map
                .drop_duplicates() # create unique map
                .set_index(ref_col) # ref_col as the key
            )
            df[col] = df[ref_col].map(end_map[col])
        return df

    @staticmethod
    def _read_fof_ct_core(
        data:pd.DataFrame,
        cols_w_bin_ref:list=["Chrom_End"],
        cols_w_trace:list=["Cell_ID", "Chrom"]
    ) -> pd.DataFrame:
        types = data.dtypes
        df_ls = []
        for c, df in data.groupby("Chrom", sort=False):
            ###################
            df = df.drop_duplicates(["Chrom", "Trace_ID", "Chrom_Start"])
            ###################

            start_ls = np.unique(df["Chrom_Start"])
            # insert NaN rows for unavailable observations
            df = df.set_index("Chrom_Start").groupby(
                "Trace_ID", 
                sort=False
            ).apply(
                lambda x: x.reindex(start_ls),
                include_groups=False
            ).reset_index()

            bin_ref_col = "Chrom_Start"

            # Chrom_End and bin columns
            bin_map = {v:i for i, v in enumerate(start_ls)}
            # create indices for Chrom_Start
            df["locus"] = df[bin_ref_col].map(bin_map)

            df = MulFish._fill_nan_cols(cols_w_bin_ref, bin_ref_col, df)
            trace_ref_col = "Trace_ID"
            df = MulFish._fill_nan_cols(cols_w_trace, trace_ref_col, df)

            df_ls.append(df)
        
        data = (
            pd.concat(df_ls)
            .reset_index(drop=True)
            .astype(types, errors="ignore")
        )
        data["Chrom"] = data["Chrom"].astype("str")
        data["Trace_ID"] = data["Trace_ID"].astype("str")
        return data
    
    @staticmethod
    def _get_helper_(col, k, df):
        if isinstance(k, tuple) and len(k) == 1:
            k = k[0]
        if isinstance(k, str):
            return df[df[col]==k]
        elif isinstance(k, int):
            chr_id = pd.unique(df[col])[k]
            return df[df[col]==chr_id]
        elif isinstance(k, slice):
            chr_id = pd.unique(df[col])[k]
            return df[df[col].isin(chr_id)]
        else:
            vals = pd.unique(df[col])
            if isinstance(k[0], int):
                return df[df[col].isin(vals[slice(*k)])]
            elif isinstance(k[0], str):
                l, u = np.where((vals==k[0])|(vals==k[1]))[0]
                if len(k) == 3:
                    s = slice(l, u, k[2])
                    return df[df[col].isin(vals[s])]
                return df[df[col].isin(vals[l:u])]
    
    def __getitem__(self, key):
        if isinstance(key, tuple):
            df = self.data
            for i, k in enumerate(key):
                if isinstance(k, tuple):
                    df = self._get_helper_(
                        col=k[0], k=k[1:], df=df
                    )
                elif k in df.columns:
                    df = self._get_helper_(
                        col=k, k=key[1:], df=df
                    )
                    break
                else:
                    df = self._get_helper_(
                        col=self.default_cols[i],
                        k=k, df=df
                    )
            return df
        else:
            return self._get_helper_(
                col=self.default_cols[0],
                k=key, df=self.data
            )
        
    
def to_very_wide(df, func=lambda a,b: a-b):
    chr_df_pivoted = df.pivot_table(
        index="Chrom_Start", 
        columns="Trace_ID", 
        values=["X", "Y", "Z"],
        sort=False
    )
    
    val_cols = ["X", "Y", "Z"]
    # N x T x D
    X = np.stack([
        chr_df_pivoted[v].values for v in val_cols
    ]).transpose(2, 1, 0)
    arrs = []
    for x in X:
        d = func(x.T[:,None,:], x.T[:,:,None])
        arrs.append(d.transpose(0, 2, 1))
    # N x D x T x T
    arrs = np.stack(arrs)
    return chr_df_pivoted, arrs


def cast_to_distmat(X:np.ndarray, func:Callable=np.nanmean) -> np.ndarray:
    """Cast the input array to a p by p matrix. Specfically,
    1. X is a 1d array: treat X as the flattened upper triangle of a p*p
    matrix and refill it into a p*p matrix.
    2. X is a p*p symmetric matrix: return X.
    3. X is a p*d asymmetric matrix, where p might or might not equal to
    d: treat as the coordinates of a single trace, calculate the 
    pairwise distance matrix.
    4. X is a n*p*p matrix, where X[i] is symmetric: apply func to each
    entry (e.g. func = np.nanmean, then this is averaging each entry).
    5. X is a n*p*d matrix, where p might or might not equal to d: first
    convert to n*p*p by applying 3 to X[i] and then apply 4.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    func : Callable, optional
        How to calculate average when the dimension of the input is at 
        least 3, by default np.nanmean.

    Returns
    -------
    np.ndarray
        Output p*p symmetric matrix.
    """
    if len(X.shape) == 1:
        N = int((1 + (1 + 8 * len(X)) ** 0.5) / 2)
        mat = np.zeros((N, N))*np.nan
        mat[np.triu_indices(N, 1)] = X
        mat.T[np.triu_indices(N, 1)] = mat[np.triu_indices(N, 1)]
    elif len(X.shape) == 2 and X.shape[0] == X.shape[1] and \
        np.allclose(X[~np.isnan(X)], X.T[~np.isnan(X)]):
            mat = X
    elif len(X.shape) == 2:
        # p x d
        outer_diff = np.stack([
            x[:,None] - x[None,:] for x in X.T
        ])
        mat = np.sqrt(np.sum(np.square(outer_diff), axis=0))
    elif X.shape[1] == X.shape[2] and \
        np.allclose(X[~np.isnan(X)], X.transpose(0,2,1)[~np.isnan(X)]):
            # print("very same")
            mat = func(X, axis=0)
    else:
        arrs = []
        for x in X:
            outer_diff = np.stack([
                a[:,None] - a[None,:] for a in x.T
            ])
            d = np.sqrt(np.sum(np.square(outer_diff), axis=0))
            arrs.append(d)
        arrs = np.stack(arrs)
        mat = func(arrs, axis=0)
    return mat
            