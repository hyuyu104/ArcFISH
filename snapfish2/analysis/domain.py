import warnings
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from statsmodels.stats import multitest as multi
from sklearn.cluster import KMeans

from .loop import AxisWiseF, DiffLoop
from ..utils.load import to_very_wide

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
        window:float
    ):
        self._data = data
        self._window = window
        
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
            result = self._single_chr_tad_by_pval(
                chr_df=self._data[self._data["Chrom"]==chr_id],
                window=self._window, fdr_cutoff=fdr_cutoff
            )
            result["c"] = chr_id
            results.append(result)
        result_df = pd.concat(results, ignore_index=True)
        cols = ["c"] + result_df.columns.drop("c").to_list()
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
            result["c"] = chr_id
            results.append(result)
        result_df = pd.concat(results, ignore_index=True)
        cols = ["c"] + result_df.columns.drop("c").to_list()
        return result_df[cols]
    
    @staticmethod
    def _single_chr_tad_by_pval(
        chr_df:pd.DataFrame,
        window:float,
        fdr_cutoff:float,
    ) -> pd.DataFrame:
        """Call TAD by axis-wise F-test."""
        chr_df_pivoted, arr_raw = to_very_wide(chr_df)
        d1d = chr_df_pivoted.index.values
        ltobj = AxisWiseF(chr_df)
        arr = ltobj.preprocess(d1d, arr_raw)
        d, p = arr.shape[1:3]

        rows = []
        for i in range(p):
            if d1d[i] - d1d[0] < window/2 or d1d[-1] - d1d[i] < window/2:
                rows.append([d1d[i], np.nan, np.nan])
                continue

            left, right = np.where(np.abs(d1d - d1d[i]) <= window)[0][[0,-1]]
            ls, rs = slice(left, i), slice(i+1, right+1)
            
            lflat = arr[:,:,ls,ls][:,:,*np.triu_indices(i-left,1)]
            lflat = lflat.transpose((1,0,2)).reshape(d,-1)
            rflat = arr[:,:,rs,rs][:,:,*np.triu_indices(right-i,1)]
            rflat = rflat.transpose((1,0,2)).reshape(d,-1)
            
            flat_intra = np.hstack([lflat, rflat])
            var_nume = np.nanmean(np.square(flat_intra), axis=1)
            
            flat_inter = arr[:,:,ls,rs].transpose((1,0,2,3)).reshape(d,-1)
            var_deno = np.nanmean(np.square(flat_inter), axis=1)
            
            f_stats = var_nume/var_deno
            n1 = np.sum(~np.isnan(flat_intra), axis=1)
            n2 = np.sum(~np.isnan(flat_inter), axis=1)
            f_pvals = stats.f.cdf(f_stats, n1, n2)
            
            act_stat = np.sum(ltobj.weights*np.tan((0.5 - f_pvals)*np.pi))
            p_val_act = 1 - stats.cauchy.cdf(act_stat)
            rows.append([d1d[i], act_stat, p_val_act])
            
        result = pd.DataFrame(rows, columns=["1D", "stat", "pval"])
        
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
        for chr_id, val in result.groupby("c", sort=False):
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
    """
    def __init__(
        self, 
        data:pd.DataFrame, 
        min_cpmt_size:float
    ):
        self._data = data
        self._min_cpmt_size = min_cpmt_size
        
    def by_axes_pc(self) -> dict:
        """Call A/B compartments by weighting the 2nd PC from different
        axes.

        Returns
        -------
        dict
            A dictionary with key being the Chrom name and value being
            an (2, p) array, with the first column being the 1D genomic
            location, and the second column being integers of 0 and 1.
        """
        result = {}
        for chr_id in pd.unique(self._data["Chrom"]):
            chr_df = self._data[self._data["Chrom"]==chr_id]
            result[chr_id] = self._single_chr_ab_by_axes_pc(
                chr_df, min_cpmt_size=self._min_cpmt_size
            )
        return result

    @staticmethod
    def _single_chr_ab_by_axes_pc(
        chr_df:pd.DataFrame, min_cpmt_size:float
    ) -> np.ndarray:
        """Perform PCA on each axis, weight axes, KMeans clustering, and
        merge small compartments.
        """
        chr_df_pivoted, arr = to_very_wide(chr_df)
        d1d = chr_df_pivoted.index.values
        
        if min_cpmt_size > np.ptp(d1d):
            raise ValueError(
                "Minimum compartment size larger than the imaging region."
            )
                
        uidx = np.triu_indices(arr.shape[-1], 1)
        abs_axes = np.abs(arr)
        cutoffs = np.nanquantile(abs_axes[:,:,*uidx], .75, axis=[0,2])
        frac_mat = np.stack([
            np.sum(arr[:,i,:,:] < c, axis=0)
            for i, c in enumerate(cutoffs)
        ])/np.sum(~np.isnan(abs_axes), axis=0)
        
        if np.sum(np.isnan(frac_mat)) > 0:
            raise ValueError("Pairwise difference matrix has holes.")    
        v2s = np.linalg.eigh(frac_mat)[1][:,:,-2]
        # Flip to align eigenvectors
        for i, v in enumerate(v2s[1:]):
            diff1 = np.sum(np.abs(np.sign(v2s[0]) - np.sign(v)))
            diff2 = np.sum(np.abs(np.sign(v2s[0]) + np.sign(v)))
            if diff1 > diff2:
                v2s[i+1] = -v
                
        wts = AxisWiseF(chr_df).weights
        wtarr = v2s*wts[:,None]
        
        cpmt_arr = KMeans(2, random_state=0).fit_predict(wtarr.T)
        cpmt_arr = ABCaller._filter_small_cmpt(cpmt_arr, min_cpmt_size, d1d)
        return np.stack([d1d, cpmt_arr])
        
    def by_first_pc(self, sigma:float=1) -> dict:
        """Call A/B compartments by first PCA. Adopted from
        Su, J.-H., Zheng, P., Kinrot, S. S., Bintu, B. & Zhuang, X. 
        Genome-Scale Imaging of the 3D Organization and Transcriptional 
        Activity of Chromatin. Cell 182, 1641-1659.e26 (2020).
        
        Parameters
        ----------
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
            chr_df = self._data[self._data["Chrom"]==chr_id]
            chr_df_pivoted, arr = to_very_wide(chr_df)
            d1d = chr_df_pivoted.index.values
            if self._min_cpmt_size > np.ptp(d1d):
                raise ValueError(
                    "Minimum compartment size larger than the imaging region."
                )
            
            dist_mats = np.sqrt(np.sum(np.square(arr), axis=1))
            triu_val = dist_mats[:,*np.triu_indices(len(d1d), 1)]
            cutoff = np.quantile(triu_val[~np.isnan(triu_val)], .75)
            logging.info(f"{chr_id} cutoff set to {cutoff}")
            
            cpmt_arr = self._single_chr_ab_by_first_pc(
                arr=arr, 
                d1d=d1d, 
                cutoff=cutoff,
                sigma=sigma
            )
            cpmt_arr = self._filter_small_cmpt(
                cpmt_arr=cpmt_arr,
                min_cpmt_size=self._min_cpmt_size,
                d1d=d1d
            )
            result[chr_id] = np.stack([d1d, cpmt_arr])
            
        return result
            
    @staticmethod
    def _single_chr_ab_by_first_pc(
        arr:np.ndarray,
        d1d:np.ndarray,
        cutoff:float,
        sigma:float=1
    ) -> np.ndarray:
        """Call A/B compartments from a single chromosome.

        Parameters
        ----------
        arr : (n, d, p, p) np.ndarray
            Pairwise difference matrices of all chromatin traces.
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.
        min_comp_size : float
            Minimum compartment size in bp.
        cutoff : float
            Distance below cutoff is treated as contact.
        sigma : float, optional
            Gaussian kernel size, by default 1.

        Returns
        -------
        np.ndarray
            An array of 0 and 1 of the same length as the number of
            genomic loci. 0 represents B compartments and 1 represents
            A compartments.

        Raises
        ------
        ValueError
            If the minimum compartment size is larger than the imaging
            region.
        """
        # (n, p, p) pairwise distance matrices for each trace
        dist_mats = np.sqrt(np.sum(np.square(arr), axis=1))
        # (p, p) pseudo contact map
        contact_mat = np.sum(dist_mats < cutoff, axis=0)\
            /np.sum(~np.isnan(dist_mats), axis=0)
            
        d1d_mat = np.abs(d1d[:,None] - d1d[None,:])
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
        
        return cpmt_arr
    
    @staticmethod
    def _filter_small_cmpt(
        cpmt_arr:np.ndarray, 
        min_cpmt_size:float, 
        d1d:np.ndarray
    ) -> np.ndarray:
        """Merge small compartments by flipping their signs."""
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
            f_pvals = self.entry_pvals(chr_df)
            weights = self.compute_weights(chr_df)
            
            df = self.region_pvals(
                weights=weights,
                f_pvals=f_pvals,
                chr_df=chr_df,
                region_df=region_df[region_df["c1"]==chr_id]
            )
            result_ls.append(df)
            
        result = pd.concat(result_ls, ignore_index=True)
        return result
        
    @staticmethod
    def region_pvals(
        weights:np.ndarray, 
        f_pvals:np.ndarray,
        chr_df:pd.DataFrame, 
        region_df:pd.DataFrame
    ) -> pd.DataFrame:
        """Aggregate p-values within a region and across axis. Assign
        uniform weights for entries within the same axis.

        Parameters
        ----------
        weights : (d,) np.ndarray
            Weight for each axis calculated by 
            :func:`compute_weights`.
        f_pvals : (d, p, p) np.ndarray
            Entry-wise p-values returned by func:`entry_pvals`.
        chr_df : pd.DataFrame
            Data of a single chromosome from both conditions in 
            FOF-CT_core format.
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
            d1d = to_very_wide(chr_df)[0].index.values
            
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