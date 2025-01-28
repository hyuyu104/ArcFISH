import os
import warnings
import logging
import numpy as np
import pandas as pd
import dask.array as da
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from statsmodels.stats import multitest as multi
from sklearn.cluster import KMeans

from .loop import AxisWiseF, DiffLoop
from ..utils.load import ChromArray, to_very_wide

# Define the display order
__all__ = [
    "TADCaller",
    "ABCaller",
    "DiffRegion"
]

class TADCaller:
    """TAD calling class.

    Parameters
    ----------
    data : pd.DataFrame
        Data in FOF-CT_core format with no repeating rows.
    window : int
        Domain size in bp to calculate intra- and inter-domain distance.
    """
    def __init__(
        self,
        data:pd.DataFrame,
        window:float,
        zarr_dire:str
    ):
        self._data = data
        self._window = window
        
        if not os.path.exists(zarr_dire):
            os.mkdir(zarr_dire)
        self._zarr_dire = zarr_dire
        
    def by_pval(
        self,
        fdr_cutoff:float
    ) -> pd.DataFrame:
        """Call TADs by thresholding FDR values.

        Parameters
        ----------
        fdr_cutoff : float
            FDR cutoff used to define TAD boundaries.

        Returns
        -------
        pd.DataFrame
            A dataframe with length equal to the number of loci. Columns
            include c (chromosome name), 1D (1D genomic location of each
            locus), stat, pval, fdr (test results), fdr_peak (whether
            the locus has a FDR < cutoff), raw_peak (whether the row is
            a local maximum), and peak (intersection of fdr_peak and 
            raw_peak).
        """
        results = []
        for chr_id in pd.unique(self._data["Chrom"]):
            carr = ChromArray(self._data[self._data["Chrom"]==chr_id])
            carr.load_write(os.path.join(self._zarr_dire, chr_id))
            result = self._single_chr_tad_by_pval(
                carr=carr,
                window=self._window, 
                fdr_cutoff=fdr_cutoff
            )
            result["c1"] = chr_id
            results.append(result)
        result_df = pd.concat(results, ignore_index=True)
        cols = ["c1"] + result_df.columns.drop("c1").to_list()
        return result_df[cols]
        
    def by_insulation(
        self,
        prominence:float,
        distance:int
    ) -> pd.DataFrame:
        """Call TADs by insulation score. Method from
        Su, J.-H., Zheng, P., Kinrot, S. S., Bintu, B. & Zhuang, X. 
        Genome-Scale Imaging of the 3D Organization and Transcriptional 
        Activity of Chromatin. Cell 182, 1641-1659.e26 (2020).
        
        Parameters
        ----------
        prominence : float
            Least height difference in normalized insulation score in 
            order for the locus to be defined as a peak. Passed to 
            scipy's `find_peaks` function.
        distance : int
            Least number of loci between two peaks. Passed to scipy's
            `find_peaks` function.

        Returns
        -------
        pd.DataFrame
            A dataframe with length equal to the number of loci. Columns
            include c (chromosome name), 1D (1D genomic location of each
            locus), insulation (insulation score), and peak (whether the
            row is a boundary).
        """
        results = []
        for chr_id in pd.unique(self._data["Chrom"]):

            result = self._single_chr_tad_by_insulation(
                chr_df=self._data[self._data["Chrom"]==chr_id], 
                window=self._window, 
                prominence=prominence, distance=distance
            )
            result["c1"] = chr_id
            results.append(result)
        result_df = pd.concat(results, ignore_index=True)
        cols = ["c1"] + result_df.columns.drop("c1").to_list()
        return result_df[cols]
    
    @staticmethod
    def _single_chr_tad_by_pval(
        carr:ChromArray,
        window:float,
        fdr_cutoff:float,
    ) -> pd.DataFrame:
        """Call TAD by axis-wise F-test."""
        carr.normalize_inplace()
        entry_var = da.nanmean(da.square(carr.arr), axis=0).compute()
        count = da.sum(~da.isnan(carr.arr), axis=0).compute()
        weights = carr.axis_weights()
        d1d = carr.d1d

        # Jumps in d1d, no entries for inter region
        warnings.filterwarnings("ignore", ".*divide by zero")
        rows = []
        for i in range(len(d1d)):
            if d1d[i] - d1d[0] < window/2 or d1d[-1] - d1d[i] < window/2:
                rows.append([d1d[i], np.nan, np.nan])
                continue

            left, right = np.where(np.abs(d1d - d1d[i]) <= window)[0][[0,-1]]
            ls, rs = slice(left, i), slice(i+1, right+1)

            lcount = count[:,ls,ls][:,*np.triu_indices(i-left,1)]
            rcount = count[:,rs,rs][:,*np.triu_indices(right-i,1)]
            intra_count = np.hstack([lcount, rcount]).sum(axis=1)
            lvar = entry_var[:,ls,ls][:,*np.triu_indices(i-left,1)]
            rvar = entry_var[:,rs,rs][:,*np.triu_indices(right-i,1)]
            var_nume = np.hstack([
                lvar*(lcount/intra_count[:,None]),
                rvar*(rcount/intra_count[:,None])
            ]).sum(axis=1)

            inter_count = np.sum(count[:,ls,rs], axis=(1,2))
            wts = count[:,ls,rs]/inter_count[:,None,None]
            var_deno = np.sum(entry_var[:,ls,rs]*wts, axis=(1,2))

            f_stats = var_nume/var_deno
            f_pvals = stats.f.cdf(f_stats, intra_count, inter_count)
            
            act_stat = np.sum(weights*np.tan((0.5 - f_pvals)*np.pi))
            p_val_act = 1 - stats.cauchy.cdf(act_stat)
            rows.append([d1d[i], *f_stats, *f_pvals, act_stat, p_val_act])
            
        result = pd.DataFrame(rows, columns=[
            "1D", "stat_x", "stat_y", "stat_z",
            "pval_x", "pval_y", "pval_z", "stat", "pval"
        ])
        
        fdr_arr = result["pval"].copy()
        avail_idx = ~np.isnan(fdr_arr)
        fdr_arr[avail_idx] = multi.multipletests(
            fdr_arr[avail_idx], method="fdr_bh"
        )[1]
        result["fdr"] = fdr_arr
        result["fdr_peak"] = result["fdr"] < fdr_cutoff
        
        result["raw_peak"] = False
        result.loc[find_peaks(result.stat.values)[0],"raw_peak"] = True
        result["peak"] = result["raw_peak"]&result["fdr_peak"]
        return result

    @staticmethod
    def _single_chr_tad_by_insulation(
        chr_df:pd.DataFrame,
        window:float,
        prominence:float,
        distance:int
    ) -> pd.DataFrame:
        """Call TAD from a single chromosome by insulation."""
        chr_df_pivoted, arr = to_very_wide(chr_df)
        d1d = chr_df_pivoted.index.values
        
        median_dist = np.nanmedian(
            np.sqrt(np.sum(np.square(arr), axis=1)), axis=0
        )
        
        scores = []
        for i in range(median_dist.shape[0]):
            # Score = 0 if half of the window is unavailable
            if d1d[i] - d1d[0] < window/2 or d1d[-1] - d1d[i] < window/2:
                scores.append(0)
                continue
            
            left, right = np.where(np.abs(d1d - d1d[i]) <= window)[0][[0,-1]]
            ls, rs = slice(left, i), slice(i+1, right+1)
            
            intra1 = median_dist[ls,ls][np.triu_indices(i-left,1)]
            intra2 = median_dist[rs,rs][np.triu_indices(right-i,1)]
            intra = np.concatenate([intra1, intra2])
            inter = median_dist[ls,rs].flatten()
            
            if np.all(np.isnan(intra)) or np.all(np.isnan(inter)):
                scores.append(0)
                continue
            
            # Normalized insulation score
            intra_median = np.nanmedian(intra)
            inter_median = np.nanmedian(inter)
            # 0 < score < 1 as inter_median > intra_median
            score = (inter_median - intra_median)/(inter_median + intra_median)
            scores.append(score)
        
        scores = np.array(scores)
        peak_idx = find_peaks(
            scores, 
            prominence=prominence, 
            distance=distance
        )[0]
        peaks = np.zeros_like(scores, dtype="bool")
        peaks[peak_idx] = True
        result = pd.DataFrame({"1D":d1d, "insulation":scores, "peak":peaks})
        return result
        
    def to_bedpe(
        self, 
        result:pd.DataFrame, 
        score_col:str,
        tree:bool=True,
        out:str | None=None
    ) -> pd.DataFrame | None:
        """Convert TAD calling result to a dataframe where each row is a
        TAD (a pair of boundaries instead of a single boundary).

        Parameters
        ----------
        result : pd.DataFrame
            Result returned by :func:`by_insulation` or 
            :func:`by_pval`.
        score_col : str
            "insulation" for result form :func:`by_insulation` and 
            "stat" for result from :func:`by_pval`.
        tree : bool, optional
            Whether to return hierarchical TADs, by default True.
        out : str | None, optional
            Output file name. If None, will return the dataframe; save
            the dataframe as a tab delimited file otherwise, by default
            None.

        Returns
        -------
        pd.DataFrame | None
            Dataframe where each row is a TAD. Columns include c1, s1, 
            e1, c2, s2, e2, {score_col}1, {score_col}2, level, idx1, and
            idx2, representing the two boundaries of the TAD. If `tree` 
            is false, then level is 0 for all rows; otherwise, it will 
            be integers increasing from the smaller to larger TADs.
        """
        rows = []
        for chr_id, val in result.groupby("c1", sort=False):
            val = val.reset_index(drop=True)
            end_map = (
                self._data[self._data["Chrom"]==chr_id]
                .drop_duplicates(["Chrom_Start"])
                .set_index("Chrom_Start")
            )["Chrom_End"][val["1D"].values]
            
            # Set the first and the last loci as peaks
            val.loc[val.index[0], "peak"] = True
            val.loc[val.index[-1], "peak"] = True
            
            sub_df = val[val["peak"]]
            level = 0
            prev, ip = sub_df.iloc[0], 0
            for i, row in sub_df.iloc[1:].iterrows():
                rows.append([
                    chr_id, prev["1D"], end_map[prev["1D"]],
                    chr_id, row["1D"], end_map[row["1D"]],
                    prev[score_col], row[score_col],
                    level, ip, i
                ])
                prev, ip = row, i
            
            if tree:
                df = sub_df.copy()
                idxs = df[1:-1].sort_values(score_col).index
                for i in idxs:
                    iloc = df.index.tolist().index(i)
                    df = df.drop(i, axis=0)
                    level += 1
                    r1, r2 = df.iloc[iloc-1], df.iloc[iloc]
                    rows.append([
                        chr_id, r1["1D"], end_map[r1["1D"]],
                        chr_id, r2["1D"], end_map[r2["1D"]],
                        r1[score_col], r2[score_col],
                        level, df.index[iloc-1], df.index[iloc]
                    ])
        
        cols = [
            "c1", "s1", "e1", "c2", "s2", "e2", 
            f"{score_col}1", f"{score_col}2",
            "level", "idx1", "idx2"
        ]
        out_df = pd.DataFrame(rows, columns=cols)
        if out is None:
            return out_df
        out_df.to_csv(out, sep="\t", index=False)
        
        
class ABCaller:
    """Call A/B compartments from multiplexed imaging data using PCA.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data in FOF-CT_core format with no repeating rows.
    min_comp_size : float
        Minimum compartment size in bp.
    zarr_dire : str
        Directory to store zarr file generated during calculation.
    """
    def __init__(
        self, 
        data:pd.DataFrame, 
        min_cpmt_size:float,
        zarr_dire:str
    ):
        self._data = data
        self._min_cpmt_size = min_cpmt_size
        if not os.path.exists(zarr_dire):
            os.mkdir(zarr_dire)
        self._zarr_dire = zarr_dire
        
    def by_axes_pc(self) -> dict:
        """Call A/B compartments by weighting the 2nd PC from different
        axes.

        Returns
        -------
        dict
            A dictionary with key being the Chrom name and value being
            an (8, p) array, with the first column being the 1D genomic
            location, the second column being integers of 0 and 1, where
            0 indicates A compartments, the 3-5 columns are the second
            eigenvectors from each axis, and the 6-8 columns are the 
            weighted second eigenvectors.
        """
        result = {}
        for chr_id in pd.unique(self._data["Chrom"]):
            carr = ChromArray(self._data[self._data["Chrom"]==chr_id])
            carr.load_write(os.path.join(self._zarr_dire, chr_id))
            result[chr_id] = self._single_chr_ab_by_axes_pc(
                carr=carr, 
                min_cpmt_size=self._min_cpmt_size
            )
        return result

    @staticmethod
    def _single_chr_ab_by_axes_pc(
        carr:ChromArray, min_cpmt_size:float
    ) -> np.ndarray:
        """Perform PCA on each axis, weight axes, KMeans clustering, and
        merge small compartments.
        """
        carr.normalize_inplace()

        if min_cpmt_size > np.ptp(carr.d1d):
            raise ValueError(
                "Minimum compartment size larger than the imaging region."
            )
        med_sq = da.nanmedian(da.square(carr.arr), axis=0).compute()
        # Already normalized. 
        # Hollowed or not does not matter. Same eigenspace.
        Vs = np.linalg.eigh(np.exp(-med_sq))[1]
        wts = carr.axis_weights()
        cpmt_arr = KMeans(
            n_clusters=2, 
            random_state=0    
        ).fit_predict((Vs[:,:,-2]*wts[:,None]).T)
        
        cpmt_arr = ABCaller._filter_small_cmpt(
            cpmt_arr=cpmt_arr,
            min_cpmt_size=min_cpmt_size,
            d1d=carr.d1d
        )
        
        idx0 = (cpmt_arr==0)[:,None]*(cpmt_arr==0)[None,:]
        med0 = np.nanmedian(med_sq[:,idx0], axis=1)

        idx1 = (cpmt_arr==1)[:,None]*(cpmt_arr==1)[None,:]
        med1 = np.nanmedian(med_sq[:,idx1], axis=1)

        mode = stats.mode((med0 < med1).astype("int64"))[0]
        if mode == 1:
            cpmt_arr = np.where(cpmt_arr==0, 1, 0)
        
        return np.stack([
            carr.d1d, 
            cpmt_arr, 
            *Vs[:,:,-2],
            *(Vs[:,:,-2]*wts[:,None])
        ])
        
    def by_first_pc(self, cutoff:float, sigma:float=1) -> dict:
        """Call A/B compartments by first PCA. Adopted from
        Su, J.-H., Zheng, P., Kinrot, S. S., Bintu, B. & Zhuang, X. 
        Genome-Scale Imaging of the 3D Organization and Transcriptional 
        Activity of Chromatin. Cell 182, 1641-1659.e26 (2020).
        
        Parameters
        ----------
        cutoff: float
            Distance below `cutoff` is defined as contact.
        sigma : float, optional
            Gaussian kernel size, by default 1.

        Returns
        -------
        dict
            A dictionary with key being the Chrom name and value being
            an (2, p) array, with the first column being the 1D genomic
            location, and the second column being integers of 0 and 1.
        """
        result = {}
        for chr_id in pd.unique(self._data["Chrom"]):
            carr = ChromArray(self._data[self._data["Chrom"]==chr_id])
            carr.load_write(os.path.join(self._zarr_dire, chr_id))
            result[chr_id] = self._single_chr_ab_by_first_pc(
                carr=carr,
                min_cpmt_size=self._min_cpmt_size,
                cutoff=cutoff,
                sigma=sigma
            )
        return result
            
    @staticmethod
    def _single_chr_ab_by_first_pc(
        carr:ChromArray,
        min_cpmt_size:float,
        cutoff:float,
        sigma:float
    ) -> np.ndarray:
        """Call A/B compartments from a single chromosome.

        Parameters
        ----------
        carr : ChromArray
            Pairwise difference array.
        min_comp_size : float
            Minimum compartment size in bp.
        cutoff : float
            Distance below cutoff is treated as contact.
        sigma : float, optional
            Gaussian kernel size.

        Returns
        -------
        np.ndarray
            An array of 0 and 1 of the same length as the number of
            genomic loci. 0 represents A compartments and 1 represents
            B compartments.

        Raises
        ------
        ValueError
            If the minimum compartment size is larger than the imaging
            region.
        """
        if min_cpmt_size > np.ptp(carr.d1d):
            raise ValueError(
                "Minimum compartment size larger than the imaging region."
            )
        # (n, p, p) pairwise distance matrices for each trace
        dist_mats = da.sqrt(da.sum(da.square(carr.arr), axis=1))
        # (p, p) pseudo contact map
        contact_mat = da.sum(dist_mats < cutoff, axis=0)\
            /da.sum(~da.isnan(dist_mats), axis=0)
        contact_mat = contact_mat.compute()
            
        d1d_mat = np.abs(carr.d1d[:,None] - carr.d1d[None,:])
        uidx = np.triu_indices_from(d1d_mat, 1)
        d1d_flat = d1d_mat[uidx]
        contact_flat = contact_mat[uidx]
        kept = (d1d_flat > 0) & (contact_flat > 0)
        log_d1d = np.log(d1d_flat[kept])
        lr = stats.linregress(log_d1d, np.log(contact_flat[kept]))
        
        contact_expected = np.exp(log_d1d*lr.slope + lr.intercept)
        contact_flat_norm = np.where(
            kept,
            contact_flat[kept]/contact_expected,
            0  # no pseudo count -> set to 0
        )
        contact_mat_norm = np.zeros_like(contact_mat, dtype="float64")
        contact_mat_norm[uidx] = contact_flat_norm
        # Fill both the upper and the lower triangle with normalized counts
        contact_mat_norm = contact_mat_norm + contact_mat_norm.T
        # Diagonal assumed to have normalized counts equal to one
        np.fill_diagonal(contact_mat_norm, 1)
        
        # Treat each row of the Gaussian smoothed normalized contact matrix
        # as a sample, and calculate the sample correlation matrix.
        # Equivalent to:
        # mat_m = smooth_norm_mat - np.mean(smooth_norm_mat, axis=0)
        # cov_mat = mat_m.T@mat_m
        # var_arr = np.diag(cov_mat)[:,None]
        # contact_corr = cov_mat/np.sqrt(var_arr*var_arr.T)
        smooth_norm_mat = gaussian_filter(contact_mat_norm, sigma)
        contact_corr = np.corrcoef(smooth_norm_mat)
        
        X = contact_corr - np.nanmean(contact_corr, axis=0)
        U, S, Vh = np.linalg.svd(X)
        cpmt_arr = ((np.sign(U[1]*S[1])+1)/2).astype("int")
        
        cpmt_arr = ABCaller._filter_small_cmpt(
            cpmt_arr=cpmt_arr,
            min_cpmt_size=min_cpmt_size,
            d1d=carr.d1d
        )
        
        idx0 = (cpmt_arr==0)[:,None]*(cpmt_arr==0)[None,:]
        med0 = np.nanmedian(contact_mat_norm[idx0])

        idx1 = (cpmt_arr==1)[:,None]*(cpmt_arr==1)[None,:]
        med1 = np.nanmedian(contact_mat_norm[idx1])

        if med0 > med1:
            cpmt_arr = np.where(cpmt_arr==0, 1, 0)
        
        return np.stack([carr.d1d, cpmt_arr])
    
    @staticmethod
    def _filter_small_cmpt(
        cpmt_arr:np.ndarray, 
        min_cpmt_size:float, 
        d1d:np.ndarray
    ) -> np.ndarray:
        """Merge small compartments by flipping their assignments."""
        # Flip the sign if the cpmtartment is too small
        while True:
            pos_ls = []
            for i, r in enumerate(cpmt_arr):
                if i == 0 or r != cpmt_arr[i-1]:
                    pos_ls.append([i, i])
                else:
                    pos_ls[-1][1] = i
            cpmt_lens = np.array([d1d[t[1]]-d1d[t[0]] for t in pos_ls])
            if np.sum(cpmt_lens < min_cpmt_size) == 0:
                break
            min_s, min_e = pos_ls[np.argmin(cpmt_lens)]
            cpmt_arr[min_s:min_e+1] = 1 - cpmt_arr[min_s:min_e+1]
        return cpmt_arr
    
    def to_bedpe(
        self, 
        result:dict, 
        out:str | None=None
    ) -> pd.DataFrame | None:
        """Convert A/B compartment calling result to a dataframe with 
        columns c1, s1, e1, c2, s2, e2, cmp, where c1, s1, e1 are the 
        chromosome, starting position, and ending position of the first
        locus of a compartment, c2, s2, e2 are the values for the last 
        locus of the compartment, and cmp indicates whether the 
        compartment is an A or an B compartment.

        Parameters
        ----------
        result : dict
            Returned by AB_from_all_chr.
        out : str | None, optional
            Output file name. If None, will return the dataframe; save
            the dataframe as a tab delimited file otherwise, by default
            None.

        Returns
        -------
        pd.DataFrame | None
            Combined dataframe. Return None if `out` is not None.
        """
        dfs = []
        for chr_id, val in result.items():
            end_map = (
                self._data[self._data["Chrom"]==chr_id]
                .drop_duplicates(["Chrom_Start"])
                .set_index("Chrom_Start")
            )["Chrom_End"][val[0]]

            df = pd.DataFrame(val.T, columns=["s1", "cmpt"])
            df["e1"] = df["s1"].map(end_map)
            df["c1"] = chr_id
            dfs.append(df[["c1", "s1", "e1", "cmpt"]])
        
        out_df = pd.concat(dfs, ignore_index=True)
        if out is None:
            return out_df
        out_df.to_csv(out, sep="\t", index=False)
        
        
class DiffRegion(DiffLoop):
    """Differential analysis of chromatin TADs.

    Parameters
    ----------
    data1 : pd.DataFrame
        Condition 1 data in FOF-CT_core format with no repeating rows.
    data2 : pd.DataFrame
        Condition 2 data in FOF-CT_core format with no repeating rows.
    """
    
    def diff_region(
        self, 
        region_df:pd.DataFrame
    ) -> pd.DataFrame:
        """Test differential regions within each chromosome.

        Parameters
        ----------
        region_df : pd.DataFrame
            List of regions to check. Columns: c1, s1, e1, c2, s2, e2.

        Returns
        -------
        pd.DataFrame
            Same format as the input, with additional columns stat, 
            pval, fdr, and log_fdr.
        """
        # Only keep regions within the same chromosome
        region_df = region_df[region_df["c1"]==region_df["c2"]]
        
        result_ls = []
        for chr_id in pd.unique(region_df["c1"]):
            chr_df = self._data[self._data["Chrom"]==chr_id]
            carr = ChromArray(chr_df)
            carr.load_write(os.path.join(self._zarr_dire, chr_id))
            carr.normalize_inplace(nstds=4)
            
            f_pvals = self.entry_pvals(chr_df, carr)
            
            weights = carr.axis_weights()
            
            df = self.region_pvals(
                weights=weights,
                f_pvals=f_pvals,
                d1d=carr.d1d,
                region_df=region_df[region_df["c1"]==chr_id]
            )
            result_ls.append(df)
            
        result = pd.concat(result_ls, ignore_index=True)
        return result
        
    @staticmethod
    def region_pvals(
        weights:np.ndarray, 
        f_pvals:np.ndarray,
        d1d:np.ndarray, 
        region_df:pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate p-values within a region and across axis. Assign
        uniform weights for entries within the same axis.

        Parameters
        ----------
        weights : (d,) np.ndarray
            Weight for each axis.
        f_pvals : (d, p, p) np.ndarray
            Entry-wise p-values returned by 
            func:`snapfish2.analysis.loop.DiffLoop.entry_pvals`.
        d1d : np.ndarray
            1D genomic locations.
        region_df : pd.DataFrame
            A dataframe containing a list of regions to check.

        Returns
        -------
        pd.DataFrame
            Same format as `region_df`, with additional columns stat,
            pval, fdr, and log_fdr.
        """
        act_stats = []
        for _, row in region_df.iterrows():
            idx = (d1d >= row["s1"])&(d1d <= row["s2"])
            sub_pvals = f_pvals[:,idx][:,:,idx]
            uidx = np.triu_indices(sub_pvals.shape[-1], 1)
            flat_pvals = sub_pvals[:,*uidx]
            
            wts = (weights/np.sum(~np.isnan(flat_pvals), axis=1))[:,None]
            act_stat = np.nansum(wts*np.tan((0.5 - flat_pvals)*np.pi))
            act_stats.append(act_stat)
        
        act_stats = np.array(act_stats)
        agg_pvals = 1 - stats.cauchy.cdf(act_stats)
        fdr_vals = multi.multipletests(agg_pvals, method="fdr_bh")[1]
        
        result_df = region_df.copy()
        result_df["stat"] = act_stats
        result_df["pval"] = agg_pvals
        result_df["fdr"] = fdr_vals
        result_df["log_fdr"] = np.log(fdr_vals)
        return result_df