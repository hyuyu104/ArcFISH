import os
from abc import ABC, abstractmethod
import logging
import warnings
from typing import Tuple
from itertools import combinations
import numpy as np
import pandas as pd
import dask.array as da
from scipy import stats
from statsmodels.stats import multitest as multi

from ..utils.load import ChromArray, to_very_wide
from ..utils.func import sample_covar_ma


# Define the display order
__all__ = [
    "LoopTestAbstract",
    "TwoSampleT",
    "AxisWiseF",
    "LoopCaller",
    "DiffLoop"
]


class LoopTestAbstract(ABC):
    """Abstract class for each test alternative. Every testing object
    passed to :class:`LoopCaller` must implement the methods in this 
    class.
    
    Parameters
    ----------
    carr : ChromArray
        Data of a single chromosome, loaded already.
    """
    @abstractmethod
    def __init__(self, carr:ChromArray):
        pass
    
    @abstractmethod
    def preprocess(self):
        """Preprocess the pairwise differences in each axis.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def ij_background(
        i:int, 
        j:int, 
        d1d:np.ndarray,
        outer_cut:int
    ) -> np.ndarray:
        """Define the local background for the (i,j) entry.

        Parameters
        ----------
        i : int
            Index of the first locus.
        j : int
            Index of the second locus.
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.
        outer_cut : int
            Loci with 1D genomic distance within `outer_cut` from the
            target locus is included in the local background.

        Returns
        -------
        (p, p) np.ndarray
            A boolean matrix with background entries equal to `True`.
        """
        pass
    
    @abstractmethod
    def append_pval(
        self, 
        result:dict, 
        cut_lo:int, 
        cut_up:int,
        outer_cut:int,
    ):
        """Performing tests and append the test results to the 
        dictionary passed in. Must add `stat` and `pval` as keys to 
        `result`, with the values being (p,p) matrix. Can also add other
        key-value pairs.

        Parameters
        ----------
        result : dict
            The result dictionary to add testing results.
        cut_lo : float
            Minimum loop size.
        cut_up : float
            Maximum loop size.
        outer_cut : int
            Loci with 1D genomic distance within `outer_cut` from the
            target locus is included in the local background.
        """
        pass
    
    @abstractmethod
    def append_summit(self, result:dict):
        """Treat the entry with the smallest p-value in each cluster as
        the summit. No additional filtering.

        Parameters
        ----------
        result : dict
            The result dictionary to add summit.
        """
        pass
    
    
class TwoSampleT(LoopTestAbstract):
    """Test 3D distance by two sample T-test. This the same test as
    implemented in the original SnapFISH paper:
    Lee, L. et al. SnapFISH: a computational pipeline to identify 
    chromatin loops from multiplexed DNA FISH data. Nat. Commun. 14, 
    4873 (2023).

    Parameters
    ----------
    carr : ChromArray
        Data of a single chromosome, loaded already.
    """
    def __init__(self, carr:ChromArray):
        if carr.normalized:
            raise ValueError(
                "Input pairwise difference array must be unnormalized."
            )
        self._d1d = carr.d1d
        self._arr = carr.arr.compute()
        
    def preprocess(self):
        """No additional preprocessing.
        """
        pass
    
    @staticmethod
    def ij_background(
        i:int,
        j:int,
        d1d:np.ndarray,
        outer_cut:int
    ) -> np.ndarray:
        """The entries that are between 25kb and `outer_cut` away from 
        the (i,j) entry are treated as the background.
        
        Parameters
        ----------
        i : int
            Index of the first locus.
        j : int
            Index of the second locus.
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.
        outer_cut : int
            Loci with 1D genomic distance within `outer_cut` from the
            target locus is included in the local background.

        Returns
        -------
        (p, p) np.ndarray
            A boolean matrix with background entries equal to `True`.
        """
        kept = np.zeros((len(d1d), len(d1d)), dtype="bool")
        
        # Select the outer square
        a = (np.abs(d1d-d1d[i])<=outer_cut).astype("int")
        b = (np.abs(d1d-d1d[j])<=outer_cut).astype("int")
        kept[(a[:,None]+b[None,:])==2] = True
        
        # Exclude the inner square
        a = (np.abs(d1d-d1d[i])<=25e3).astype("int")
        b = (np.abs(d1d-d1d[j])<=25e3).astype("int")
        kept[(a[:,None]+b[None,:])==2] = False
        
        kept[np.tril_indices_from(kept)] = False
        return kept
    
    def append_pval(
        self, 
        result:dict, 
        cut_lo:int, 
        cut_up:int,
        outer_cut:int,
    ):
        """Perform two-sample t-tests.

        Parameters
        ----------
        result : dict
            The result dictionary to add testing results.
        cut_lo : float
            Minimum loop size.
        cut_up : float
            Maximum loop size.
        outer_cut : int
            Loci with 1D genomic distance within `outer_cut` from the
            target locus is included in the local background.
        """
        p = len(self._d1d)
        d1map = self._d1d[None,:] - self._d1d[:,None]
        
        result["stat"] = np.zeros((p,p), dtype="float64")*np.nan
        result["pval"] = np.zeros((p,p), dtype="float64")*np.nan
        
        warnings.filterwarnings("ignore", ".*Mean of empty slice")
        for i, j in zip(*np.where((d1map>=cut_lo)&(d1map<=cut_up))):
            kept = self.ij_background(i, j, self._d1d, outer_cut)
            if np.sum(kept) == 0:
                continue
            
            loop_dist = np.sqrt(np.sum(np.square(self._arr[:,:,i,j]), axis=1))
            loop_dist = loop_dist[~np.isnan(loop_dist)]
            
            bkgd_dist = np.sqrt(np.sum(np.square(self._arr[:,:,kept]), axis=1))
            bkgd_mean = np.nanmean(bkgd_dist, axis=1)
            bkgd_mean = bkgd_mean[~np.isnan(bkgd_mean)]
            stat, pval = stats.ttest_ind(
                loop_dist, 
                bkgd_mean, 
                equal_var=False, 
                alternative="less"
            )
            result["stat"][i,j] = result["stat"][j,i] = stat
            result["pval"][i,j] = result["pval"][j,i] = pval
            
    def append_summit(self, result:dict):
        """Treat the entry with the smallest p-value in each cluster as
        a potential summit. Filter summits by contact frequency. If the 
        summit is a singleton (i.e. from only one candidate), then it is
        marked as summit if contact frequency is larger than 1/2. If the
        summit is not a singleton (i.e. from multiple candidates), then 
        it is marked as summit if contact frequency is larger than 1/3.

        Parameters
        ----------
        result : dict
            The result dictionary to add summit.
        """
        arr, d1d = self._arr, self._d1d
        dmaps = np.sqrt(np.sum(np.square(arr), axis=1))
        freq_dists = dmaps[:,np.abs(d1d[:,None] - d1d[None,:])==25e3]
        warnings.filterwarnings("ignore", r".*invalid value")
        freq_mat = np.sum(dmaps<np.nanmean(freq_dists), axis=0)/\
            np.sum(~np.isnan(dmaps), axis=0)
            
        labeled = result["label"]
        summit = np.zeros_like(labeled, dtype="bool")
        
        for lab in np.unique(labeled[~np.isnan(labeled)]):
            # Keep only the triu part to compare p values
            idx = np.where(np.triu(labeled)==lab)
            max_i = np.argmin(result["pval"][idx])
            
            i1, i2 = idx[0][max_i], idx[1][max_i]
            if len(idx[0]) == 2:  # singleton (symmetric)
                if freq_mat[i1,i2] > 1/2:
                    summit[i1,i2] = summit[i2,i1] = True
            elif freq_mat[i1,i2] > 1/3:
                summit[i1,i2] = summit[i2,i1] = True
        
        result["summit"] = summit

            
class AxisWiseF(LoopTestAbstract):
    """Perform axis-wise F-test and combine p-values by Cauchy 
    aggregation test. 

    Parameters
    ----------
    carr : ChromArray
        Data of a single chromosome, loaded already.
    """
    def __init__(self, carr:ChromArray):
        self._carr = carr
        
    def preprocess(self):
        """Remove outliers and normalize by 1D genomic distance by 
        :func:`snapfish2.utils.load.ChromArray.normalize_inplace`.
        """
        self._carr.normalize_inplace()
        
    @staticmethod
    def ij_background(
        i:int, 
        j:int, 
        d1d:np.ndarray,
        outer_cut:int
    ) -> np.ndarray:
        """The entries that are between 25kb and `outer_cut` away from 
        the (i,j) entry are treated as the background.
        
        Parameters
        ----------
        i : int
            Index of the first locus.
        j : int
            Index of the second locus.
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.
        outer_cut : int
            Loci with 1D genomic distance within `outer_cut` from the
            target locus is included in the local background.

        Returns
        -------
        (p, p) np.ndarray
            A boolean matrix with background entries equal to `True`.
        """
        kept = np.zeros((len(d1d), len(d1d)), dtype="bool")
        
        # Select the outer square
        a = (np.abs(d1d-d1d[i])<=outer_cut).astype("int")
        b = (np.abs(d1d-d1d[j])<=outer_cut).astype("int")
        kept[(a[:,None]+b[None,:])==2] = True
        
        # Exclude the inner square
        a = (np.abs(d1d-d1d[i])<=25e3).astype("int")
        b = (np.abs(d1d-d1d[j])<=25e3).astype("int")
        kept[(a[:,None]+b[None,:])==2] = False
        
        kept[np.tril_indices_from(kept)] = False
        return kept
        
    def append_pval(
        self, 
        result:dict, 
        cut_lo:int, 
        cut_up:int,
        outer_cut:int,
    ):
        """Perform tests in each axis. In addition to `stat` and `pval`,
        also append `axis_stat` and `axis_pval`, which are both (d,p,p)
        matrices.

        Parameters
        ----------
        result : dict
            The result dictionary to add testing results.
        cut_lo : float
            Minimum loop size.
        cut_up : float
            Maximum loop size.
        outer_cut : int
            Loci with 1D genomic distance within `outer_cut` from the
            target locus is included in the local background.
        """
        n, p, d = self._carr.X.shape
        d1map = self._carr.d1d[None,:] - self._carr.d1d[:,None]

        entry_var = da.nanmean(da.square(self._carr.arr), axis=0).compute()
        count = da.sum(~da.isnan(self._carr.arr), axis=0).compute()

        result["axis_stat"] = np.zeros((d,p,p), dtype="float64")*np.nan
        result["axis_pval"] = np.zeros((d,p,p), dtype="float64")*np.nan

        for i, j in zip(*np.where((d1map>=cut_lo)&(d1map<=cut_up))):
            bkgd_map = self.ij_background(i, j, self._carr.d1d, outer_cut)
            if np.sum(bkgd_map) == 0:
                continue
            
            num_unloop = np.sum(count[:,bkgd_map], axis=1)
            wts = count[:,bkgd_map]/num_unloop[:,None]
            denom = np.sum(wts*entry_var[:,bkgd_map], axis=1)
            
            f_stats = entry_var[:,i,j]/denom
            result["axis_stat"][:,i,j] = result["axis_stat"][:,j,i] = f_stats
            f_pvals = stats.f.cdf(f_stats, count[:,i,j], num_unloop)
            result["axis_pval"][:,i,j] = result["axis_pval"][:,j,i] = f_pvals
            
        weights = self._carr.axis_weights()[:,None,None]
        result["stat"] = np.sum(weights*np.tan(
            (0.5 - result["axis_pval"])*np.pi
        ), axis=0)
        result["pval"] = 1 - stats.cauchy.cdf(result["stat"])
    
    def append_summit(self, result:dict):
        """Treat the entry with the smallest p-value in each cluster as
        the summit. No additional filtering.

        Parameters
        ----------
        result : dict
            The result dictionary to add summit.
        """
        labeled = result["label"]
        summit = np.zeros_like(labeled, dtype="bool")
        for lab in np.unique(labeled[~np.isnan(labeled)]):
            # Keep only the triu part to compare p values
            idx = np.where(np.triu(labeled)==lab)
            max_i = np.argmin(result["pval"][idx])
            i1, i2 = idx[0][max_i], idx[1][max_i]
            summit[i1,i2] = summit[i2,i1] = True
        result["summit"] = summit
        

class LoopCaller:
    """Call chromatin loops from multiplexed imaging data.

    Parameters
    ----------
    data : pd.DataFrame
        Data in FOF-CT_core format with no repeating rows.
    zarr_dire : str
        Directory to store zarr file generated during calculation.
    fdr_cutoff: float
        FDR cut-off for chromatin loops, by default 0.1.
    cut_lo : float, optional
        Minimum loop size (1D genomic distance between the first locus
        and the second locus), by default 1e5.
    cut_up : float, optional
        Maximum loop size, by default 1e6.
    gap : float, optional
        Loop candidates `gap` away from each other are considered 
        candidates for the same summit, by default 50e3.
    outer_cut : float, optional
        Loci with 1D genomic distance within `outer_cut` from the target 
        locus is included in the local background, by default 50e3.
    """
    def __init__(
        self,
        data:pd.DataFrame,
        zarr_dire:str,
        fdr_cutoff:float=0.1,
        cut_lo:float=1e5,
        cut_up:float=1e6,
        gap:float=50e3,
        outer_cut:float=50e3
    ):
        self._data = data
        if not os.path.exists(zarr_dire):
            os.mkdir(zarr_dire)
        self._zarr_dire = zarr_dire
        
        self._fdr_cutoff = fdr_cutoff
        self._cut_lo = int(cut_lo)
        self._cut_up = int(cut_up)
        
        self._gap = int(gap)
        self._outer_cut = int(outer_cut)
        
    @property
    def zarr_dire(self):
        """str : Directory to store zarr files."""
        return self._zarr_dire
        
    @property
    def fdr_cutoff(self):
        """float : FDR cut-off for chromatin loops."""
        return self._fdr_cutoff
    
    @property
    def loop_range(self):
        """Tuple[int, int] : Loop size considered."""
        return (self._cut_lo, self._cut_up)
    
    @property
    def gap(self):
        """int : Loop candidates `gap` away from each other are 
        considered candidates for the same summit.
        """
        return self._gap
    
    @property
    def outer_cut(self):
        """int : Loci with 1D genomic distance within `outer_cut` from 
        the target locus is included in the local background.
        """
        return self._outer_cut
        
    def loops_from_all_chr(
        self, 
        ltclass:LoopTestAbstract
    ) -> pd.DataFrame:
        """Identify chromatin loops from all chromosomes in the data.

        Parameters
        ----------
        ltclass : LoopTestAbstract
            Test method used.

        Returns
        -------
        pd.DataFrame
            Each row being an (i,j) pair, with the `summit` column
            indicates whether this pair is a chromatin loop.
        """
        out = []
        for chr_id in pd.unique(self._data["Chrom"]):
            results = self.loops_from_single_chr(chr_id, ltclass)
            out_c = self.to_bedpe(results, chr_id)
            out.append(out_c)
        if len(out) > 0:
            return pd.concat(out).reset_index(drop=True)
    
    def loops_from_single_chr(
        self, 
        chr_id:str, 
        ltclass:LoopTestAbstract
    ) -> dict:
        """Call chromatin loops from a single chromosome.

        Parameters
        ----------
        chr_id : str
            The chromosome name.
        ltclass : LoopTestAbstract
            Test method used.

        Returns
        -------
        dict
            A dictionary with keys stat, pval, fdr, candidate, label, 
            summit. Values are (p,p) matrices.
        """
        carr = ChromArray(self._data[self._data["Chrom"]==chr_id])
        carr.load_write(f"{self._zarr_dire}/{chr_id}")
        test_class = ltclass(carr)
        test_class.preprocess()
        
        result = {}
        test_class.append_pval(
            result=result,
            cut_lo=self._cut_lo,
            cut_up=self._cut_up,
            outer_cut=self._outer_cut
        )
        
        pvals = result["pval"].copy()
        uidx = np.triu_indices_from(pvals, 1)
        triu_pvals = pvals[uidx].copy()
        fdr_vals = multi.multipletests(
            triu_pvals[~np.isnan(triu_pvals)], method="fdr_bh"
        )[1]
        # reshape to a symmetric matrix
        triu_pvals[~np.isnan(triu_pvals)] = fdr_vals
        pvals[uidx] = triu_pvals
        pvals.T[uidx] = pvals[uidx]
        result["fdr"] = pvals
        
        result["candidate"] = result["fdr"] < self._fdr_cutoff
        result["label"] = self.spread_label(result["candidate"], carr.d1d)
        
        test_class.append_summit(result)
        carr.close()
            
        return result
    
    def spread_label(
        self, 
        candidates:np.ndarray, 
        d1d:np.ndarray
    ) -> np.ndarray:
        """Group loop candidates. Each group corresponds to one summit.

        Parameters
        ----------
        candidates : (p,p) np.ndarray
            Boolean matrix with candidate entries being True.
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.

        Returns
        -------
        (p, p) np.ndarray
            Integer valued matrix. Entries with the same value define 
            the same loop summit.
        """
        idxs = set(zip(*np.where(candidates)))
        gap, lab, need_check = self.gap, 0, set()
        label_mat = np.zeros_like(candidates)*np.nan
        unlabeled = idxs.copy()
        while len(unlabeled) > 0:
            if len(need_check) == 0:
                idx = unlabeled.pop()
                lab += 1
            else:
                idx = need_check.pop()
            label_mat[idx] = label_mat[idx[1],idx[0]] = lab
            unlabeled = set(zip(*np.where(np.isnan(label_mat))))&idxs
            
            f1 = (np.abs(d1d-d1d[idx[0]])<=gap).astype("int")
            f2 = (np.abs(d1d-d1d[idx[1]])<=gap).astype("int")
            within_gap = set(zip(*np.where((f1[:,None]+f2[None,:])==2)))
            need_check = (need_check|within_gap)&unlabeled
        return label_mat

    def to_bedpe(
        self, 
        result:dict, 
        chr_id:str, 
        out:str=None
    ) -> pd.DataFrame | None:
        """Convert loop calling result to a data frame with columns c1,
        s1, e1, c2, s2, e2, stat, pval, fdr, candidate, label, summit.
        c1, s1, e1 are the chromosome, starting position, and ending 
        position of the first locus of the pair, c2, s2, e2 are the 
        values for the second locus. stat, pval, fdr are test results
        for the pair. Candidate is boolean valued, label is integer
        valued, and summit is boolean valued.

        Parameters
        ----------
        result : dict
            Output from :func:`LoopCaller.loops_from_single_chr`.
        chr_id : str
            Chromosome name.
        out : str, optional
            Output file name. If None, will return the dataframe; save
            the dataframe as a tab delimited file otherwise, by default
            None.

        Returns
        -------
        pd.DataFrame | None
            Combined dataframe. Return None if `out` is not None.
        """
        d1df = self._data[self._data["Chrom"]==chr_id][
            ["Chrom_Start", "Chrom_End"]
        ].drop_duplicates()
        a = d1df["Chrom_Start"].values
        a1 = a[:,None] - np.zeros_like(a)
        a2 = a - np.zeros_like(a)[:,None]

        b = d1df["Chrom_End"].values
        b1 = b[:,None] - np.zeros_like(b)
        b2 = b - np.zeros_like(b)[:,None]

        uidxs = np.triu_indices_from(a1, 1)

        out_df = pd.DataFrame(
            np.stack([a1, b1, a2, b2])[:,uidxs[0], uidxs[1]].T,
            columns=["s1", "e1", "s2", "e2"]
        )
        out_df["c1"] = out_df["c2"] = chr_id
        out_df = out_df[["c1", "s1", "e1", "c2", "s2", "e2"]]

        for k, v in result.items():
            if v.shape == (len(d1df),len(d1df)):
                out_df[k] = v[uidxs]
        # out_df = out_df.dropna(subset="stat")
        
        if out is None:
            return out_df
        out_df.to_csv(out, sep="\t", index=False)


class DiffLoop:
    """Differential analysis of chromatin loops.

    Parameters
    ----------
    data1 : pd.DataFrame
        Condition 1 data in FOF-CT_core format with no repeating rows.
    data2 : pd.DataFrame
        Condition 2 data in FOF-CT_core format with no repeating rows.
    """
    def __init__(self, data1:pd.DataFrame, data2:pd.DataFrame):
        d1, d2 = data1.copy(), data2.copy()
        d1["sf_grp"] = 1
        d2["sf_grp"] = 2
        self._data = pd.concat([d1, d2], ignore_index=True)
        
    @staticmethod
    def compute_weights(chr_df:pd.DataFrame) -> np.ndarray:
        """Compute the weight for each axis by calling
        :func:`snapfish2.loop.caller.AxisWiseF.compute_cov`.

        Parameters
        ----------
        chr_df : pd.DataFrame
            Data of a single chromosome in FOF-CT_core format.

        Returns
        -------
        (d,) np.ndarray
            Weight for each axis.
        """
        covs = AxisWiseF.compute_cov(chr_df)
        weights = np.array([1/np.median(c) for c in covs])
        weights = weights/np.sum(weights)
        return weights
    
    def diff_loops(
        self, 
        summit_df:pd.DataFrame,
        s:float=5e3,
        full_agg_pval=True
    ) -> dict:
        """Call differential chromatin interactions.

        Parameters
        ----------
        summit_df : pd.DataFrame
            List of loops to check. Have columns c1, s1, e1, c2, s2, e2.
        s : float, optional
            Gaussian kernel size in bp, by default 5e3.
        full_agg_pval : bool, optional
            Whether to calculate the p-value for all entries, by default 
            True. If False, will only calculate entries in `summit_df`.

        Returns
        -------
        dict
            Differential loop testing result with keys f_pvals, 
            agg_pvals, and fdr. Values are (p,p) matrices.
        """
        result_all = {}
        for chr_id in pd.unique(self._data["Chrom"]):
            results = {}
            summit_chr = summit_df[
                (summit_df["c1"]==chr_id)&(summit_df["c2"]==chr_id)
            ]
            chr_df = self._data[self._data["Chrom"]==chr_id]
            
            results["f_pvals"] = self.entry_pvals(chr_df)
            
            weights = self.compute_weights(chr_df)
            
            agg_p_mat, idx = self.loop_pvals(
                weights, 
                results["f_pvals"], 
                chr_df, 
                summit_chr, 
                s=s, 
                full_agg_pval=full_agg_pval
            )
            results["agg_pvals"] = agg_p_mat
            
            results["fdr"] = self.pval_to_fdr(agg_p_mat, idx)
            result_all[chr_id] = results
        return result_all
        
    @staticmethod
    def entry_pvals(chr_df:pd.DataFrame) -> np.ndarray:
        """Calculate axis-wise entry-wise p-values by F-tests.

        Parameters
        ----------
        chr_df : pd.DataFrame
            Data of a single chromosome from both conditions in 
            FOF-CT_core format.

        Returns
        -------
        (d, p, p) np.ndarray
            Axis-wise entry-wise p-values.
        """
        chr_df_pivoted, arrs = to_very_wide(chr_df)
        d1d = chr_df_pivoted.index.values
        # Normalize both groups together
        arrs = AxisWiseF.preprocess(d1d, arrs)

        grp1_tids = chr_df[chr_df["sf_grp"]==1].Trace_ID.unique()
        grp2_tids = chr_df[chr_df["sf_grp"]==2].Trace_ID.unique()

        arr1 = arrs[chr_df_pivoted["X"].columns.isin(grp1_tids)]
        arr2 = arrs[chr_df_pivoted["X"].columns.isin(grp2_tids)]

        warnings.filterwarnings("ignore", "Mean of empty slice")
        var1 = np.nanmean(np.square(arr1), axis=0)
        count1 = np.sum(~np.isnan(arr1), axis=0)
        var2 = np.nanmean(np.square(arr2), axis=0)
        count2 = np.sum(~np.isnan(arr2), axis=0)

        warnings.filterwarnings("ignore", "divide by zero")
        f_pvals = stats.f.cdf(var1/var2, count1, count2)
        # Two-sided: either var1 > var2 or var1 < var2
        f_pvals = 2*np.min(np.stack([f_pvals, 1 - f_pvals]), axis=0)
        return f_pvals
        
    @staticmethod
    def loop_pvals(
        weights:np.ndarray, 
        f_pvals:np.ndarray, 
        chr_df:pd.DataFrame, 
        summit_chr:pd.DataFrame, 
        s:float=5e3, 
        full_agg_pval:bool=True
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculate p-value for each loop considered. The p-value is 
        aggregated 1) from all axis by weights inversely proportional to 
        measurement errors 2) from entries within the same axis by a 
        Gaussian kernel with size `s`.

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
        summit_chr : pd.DataFrame
            List of loops within the chromosome to check.
        s : float, optional
            Gaussian kernel size in bp, by default 5e3
        full_agg_pval : bool, optional
            Whether to calculate the p-value for all entries, by default
            True. If False, will only calculate entryes in `summit_chr`.

        Returns
        -------
        Tuple[(p, p) np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Aggregated p-values and indices of the summits.
        """
        chr_df_pivoted = to_very_wide(chr_df)[0]
        d1d = chr_df_pivoted.index.values
        
        # Select entries corresponding to loop summits
        d1d_sr = pd.Series(np.arange(len(d1d)), index=d1d)
        iidx = d1d_sr[summit_chr.s1].values
        jidx = d1d_sr[summit_chr.s2].values
        if not full_agg_pval:
            all_idx = zip(iidx, jidx)
        else:
            all_idx = combinations(range(d1d.shape[0]), 2)
        
        act_stats = np.zeros_like(f_pvals, dtype="float64")*np.nan
        for i, j in all_idx:
            for c in range(f_pvals.shape[0]):
                    rsq = ((d1d - d1d[i])**2)[:,None] \
                        + ((d1d - d1d[j])**2)[None,:]
                    wt_mat = np.exp(-rsq/(2*s**2))
                    # Remove the lower triangular region
                    wt_mat[np.tril_indices_from(wt_mat, 1)] = np.nan
                    # Remove positions with no p-values
                    wt_mat[np.isnan(f_pvals[c])] = np.nan
                    # So that sum equals the axis's total weight
                    wt_mat = wt_mat/np.nansum(wt_mat) * weights[c]
                    
                    act_stats[c,i,j] = np.nansum(
                        wt_mat*np.tan((0.5 - f_pvals[c])*np.pi)
                    )
        agg_p_mat = 1 - stats.cauchy.cdf(np.sum(act_stats, axis=0))
        uidxs = np.triu_indices_from(agg_p_mat)
        agg_p_mat.T[uidxs] = agg_p_mat[uidxs]
        return agg_p_mat, (iidx, jidx)
    
    @staticmethod
    def pval_to_fdr(agg_p_mat:np.ndarray, idx:tuple) -> np.ndarray:
        """Adjust multiple testing by FDR. Only adjust for p-values 
        specified in `idx`.

        Parameters
        ----------
        agg_p_mat : (p, p) np.ndarray
            P-value matrix.
        idx : tuple
            Positions in p-value matrix to adjust. Format like the 
            output of `np.where`.

        Returns
        -------
        (p, p) np.ndarray
            FDR matrix. NaN at positions outside `idx`.
        """
        fdrs = np.zeros_like(agg_p_mat)*np.nan
        pvals = agg_p_mat[idx]
        fdr_vals = multi.multipletests(
            pvals[~np.isnan(pvals)], method="fdr_bh"
        )[1]
        
        fdrs[idx] = fdr_vals
        fdrs.T[idx] = fdrs[idx]
        return fdrs
        
    def to_bedpe_loop(
        self, 
        result_all:dict, 
        fdr_cutoff:float,
        out:str | None=None
    ) -> pd.DataFrame | None:
        """Convert differential loop result to dataframe, similar format
        as `summit_df`.

        Parameters
        ----------
        result_all : dict
            Dictionary returned by :func:`diff_loops`.
        out : str | None, optional
            Output file name. If None, will return the dataframe; save
            the dataframe as a tab delimited file otherwise, by default
            None.

        Returns
        -------
        pd.DataFrame | None
            Output dataframe. Return None if `out` is not None.
        """
        bedpe_df = []
        for k, v in result_all.items():
            chr_df = self._data[self._data["Chrom"]==k]
            d1 = chr_df[["Chrom_Start", "Chrom_End"]].drop_duplicates().values
            s2, s1 = np.meshgrid(d1.T[0], d1.T[0])
            e2, e1 = np.meshgrid(d1.T[1], d1.T[1])

            result_arr = np.stack([s1, e1, s2, e2, v["agg_pvals"], v["fdr"]])
            arr_triu = result_arr[:,*np.triu_indices_from(result_arr[0], 1)]
            # Keep only rows with FDR not being NaN
            # Changing full_agg_pval will change the number of rows kept
            avail_idx = np.where(~np.isnan(arr_triu[-1]))

            cols = ["s1", "e1", "s2", "e2", "pval", "fdr"]
            df = pd.DataFrame(arr_triu[:,*avail_idx].T, columns=cols)
            df = df.astype({c:"int" for c in cols[:4]})
            df["c1"] = df["c2"] = k
            
            bedpe_df.append(df)
        
        bedpe_df = pd.concat(bedpe_df)[
            ["c1", "s1", "e1", "c2", "s2", "e2", "pval", "fdr"]
        ]
        bedpe_df["log_fdr"] = np.log(bedpe_df["fdr"])
        bedpe_df["diff"] = bedpe_df["fdr"] < fdr_cutoff
        
        logging.info(
            f"Found {np.sum(bedpe_df["diff"])} differential loops, "\
            + f"while {np.sum(~bedpe_df["diff"])} are not differential loops."
        )
        
        if out is None:
            return bedpe_df
        bedpe_df.to_csv(out, sep="\t", index=False)