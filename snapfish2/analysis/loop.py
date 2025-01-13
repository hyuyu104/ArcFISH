from abc import ABC, abstractmethod
import logging
import warnings
from typing import Tuple
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest as multi

from ..utils import to_very_wide
from ..utils import sample_covar_ma


# Define the display order
__all__ = [
    "LoopTestAbstract",
    "TwoSampleT",
    "AxisWiseF",
    "AxisWiseT",
    "AxisWiseChi2",
    "LoopCaller",
    "DiffLoop"
]


class LoopTestAbstract(ABC):
    """Abstract class for each test alternative. Every testing object
    passed to :class:`LoopCaller` must implement the methods in this 
    class.
    
    Parameters
    ----------
    chr_df : pd.DataFrame
        Data of a single chromosome in FOF-CT_core format.
    """
    @abstractmethod
    def __init__(self, chr_df:pd.DataFrame):
        pass
    
    @staticmethod
    @abstractmethod
    def preprocess(d1d:np.ndarray, arr:np.ndarray) -> np.ndarray:
        """Preprocess the pairwise differences in each axis.

        Parameters
        ----------
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.
        arr : (n, d, p, p) np.ndarray
            Pairwise difference matrices of all chromatin traces.

        Returns
        -------
        (n, d, p, p) np.ndarray
            Processed pairwise difference matrices.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def ij_background(i:int, j:int, d1d:np.ndarray) -> np.ndarray:
        """Define the local background for the (i,j) entry.

        Parameters
        ----------
        i : int
            Index of the first locus.
        j : int
            Index of the second locus.
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.

        Returns
        -------
        (p, p) np.ndarray
            A boolean matrix with background entries equal to `True`.
        """
        pass
    
    @abstractmethod
    def test_func(
        self, 
        loop_arr:np.ndarray, 
        bkgd_arr:np.ndarray
    ) -> Tuple[float, float]:
        """Test whether the `loop_arr` (loop entry) has smaller average 
        distance compared to `bkgd_arr` (local background).

        Parameters
        ----------
        loop_arr : (n, d) np.ndarray
            Pairwise differences of all chromatin traces and all axes at
            the loop entry.
        bkgd_arr : (n, d, r) np.ndarray
            Pairwise differences of all chromatin traces and all axes at
            the local background entries, where `r` is the number of 
            entries in the local background model.

        Returns
        -------
        Tuple[float, float]
            Test statistic and p-value.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def filter_summits(
        summits:np.ndarray, 
        freq_mat:np.ndarray, 
        idx:np.ndarray, 
        max_i:int
    ):
        """Filter summits identified by contact frequency.

        Parameters
        ----------
        summits : (p, p) np.ndarray
            Boolean matrix with entries identified as summits equal to
            `True`, will be modified inplace.
        freq_mat : (p, p) np.ndarray
            Contact frequency matrix, where the (i,j) entry is defined 
            as the number of traces with the (i,j) distances below a
            threshold.
        idx : (k, k) np.ndarray
            The indices of the loop candidates associated with the 
            summit considered, returned by `np.where(candidate_mat)`.
        max_i : int
            The index of the summit within the candidates, so `max_i` is
            between 1 and k-1.
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
    chr_df : pd.DataFrame
        Data of a single chromosome in FOF-CT_core format.
    """
    def __init__(self, chr_df:pd.DataFrame):
        pass
    
    @staticmethod
    def preprocess(d1d:np.ndarray, arr:np.ndarray) -> np.ndarray:
        """Return the same array.
        
        Parameters
        ----------
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.
        arr : (n, d, p, p) np.ndarray
            Pairwise difference matrices of all chromatin traces.

        Returns
        -------
        (n, d, p, p) np.ndarray
            Processed pairwise difference matrices.
        """
        return arr
    
    @staticmethod
    def ij_background(i:int, j:int, d1d:np.ndarray) -> np.ndarray:
        """The entries that are between 25kb and 50kb away from the 
        (i,j) entry are treated as the background.
        
        Parameters
        ----------
        i : int
            Index of the first locus.
        j : int
            Index of the second locus.
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.

        Returns
        -------
        (p, p) np.ndarray
            A boolean matrix with background entries equal to `True`.
        """
        kept = np.zeros((len(d1d), len(d1d)), dtype="bool")
        
        # select the outer square
        a = (np.abs(d1d-d1d[i])<=50e3).astype("int")
        b = (np.abs(d1d-d1d[j])<=50e3).astype("int")
        kept[(a[:,None]+b[None,:])==2] = True
        
        # exclude the inner square
        a = (np.abs(d1d-d1d[i])<=25e3).astype("int")
        b = (np.abs(d1d-d1d[j])<=25e3).astype("int")
        kept[(a[:,None]+b[None,:])==2] = False
        
        kept[np.tril_indices_from(kept)] = False
        return kept
    
    def test_func(
        self, 
        loop_arr:np.ndarray, 
        bkgd_arr:np.ndarray
    ) -> Tuple[float, float]:
        """Convert the axis differences to Euclidean distances. Average
        the `r` local background distances within each trace. The final 
        testing values are a size (n,) array for loop and a size (n,)
        array for background. This is tested by a one-sided T-test with 
        unequal variance.

        Parameters
        ----------
        loop_arr : (n, d) np.ndarray
            Pairwise differences of all chromatin traces and all axes at
            the loop entry.
        bkgd_arr : (n, d, r) np.ndarray
            Pairwise differences of all chromatin traces and all axes at
            the local background entries, where `r` is the number of 
            entries in the local background model.

        Returns
        -------
        Tuple[float, float]
            T-test statistic and p-value.
        """
        loop_dist = np.sqrt(np.sum(np.square(loop_arr), axis=1))
        loop_dist = loop_dist[~np.isnan(loop_dist)]
        
        bkgd_dist = np.sqrt(np.sum(np.square(bkgd_arr), axis=1))
        bkgd_mean = np.nanmean(bkgd_dist, axis=1)
        bkgd_mean = bkgd_mean[~np.isnan(bkgd_mean)]
        t_stat, p_val_t = stats.ttest_ind(
            loop_dist, bkgd_mean, equal_var=False, alternative="less"
        )
        return t_stat, p_val_t
    
    @staticmethod
    def filter_summits(
        summits:np.ndarray, 
        freq_mat:np.ndarray, 
        idx:np.ndarray, 
        max_i:int
    ):
        """Filter summits identified by contact frequency. If the summit
        is a singleton (i.e. from only one candidate), then it is marked
        as summit if contact frequency is larger than 1/2. If the summit
        is not a singleton (i.e. from multiple candidates), then it is 
        marked as summit if contact frequency is larger than 1/3.

        Parameters
        ----------
        summits : (p, p) np.ndarray
            Boolean matrix with entries identified as summits equal to
            `True`, will be modified inplace.
        freq_mat : (p, p) np.ndarray
            Contact frequency matrix, where the (i,j) entry is defined 
            as the number of traces with the (i,j) distances below a
            threshold.
        idx : (k, k) np.ndarray
            The indices of the loop candidates associated with the 
            summit considered, returned by `np.where(candidate_mat)`.
        max_i : int
            The index of the summit within the candidates, so `max_i` is
            between 1 and k-1.
        """
        i1, i2 = idx[0][max_i], idx[1][max_i]
        if len(idx[0]) == 2:  # singleton (symmetric)
            if freq_mat[i1,i2] > 1/2:
                summits[i1,i2] = summits[i2,i1] = True
        elif freq_mat[i1,i2] > 1/3:
            summits[i1,i2] = summits[i2,i1] = True
            
            
class AxisWiseF(LoopTestAbstract):
    """Perform axis-wise F-test and combine p-values by Cauchy 
    aggregation test. 

    Parameters
    ----------
    chr_df : pd.DataFrame
        Data of a single chromosome in FOF-CT_core format.
    """
    def __init__(self, chr_df:pd.DataFrame):
        covs = self.compute_cov(chr_df)
        weights = []
        for cov in covs:
            # lambdas = np.linalg.eigvals(cov)
            # var = bema_var_hat(lambdas, X.shape[1], X.shape[0], 0.2)
            var = np.median(np.diag(cov))
            weights.append(1/var)
        self.weights = weights/np.sum(weights)
        # wstr = [f"{c}({w})" for c, w in zip(coor_cols, self.weights.round(3))]
        # print("Weights:", *wstr)
        
    @staticmethod
    def compute_cov(chr_df:pd.DataFrame) -> np.ndarray:
        """Compute the sample covariance matrices for each axis. Handle
        missing values and thus the sample covariance matrix might not
        be PSD.

        Parameters
        ----------
        chr_df : pd.DataFrame
            Data of a single chromosome in FOF-CT_core format.

        Returns
        -------
        (d, p, p) np.ndarray
            Sample covariance matrices for each axis.
        """
        coor_cols = ["X", "Y", "Z"]
        pivoted = chr_df.pivot_table(
            index="Chrom_Start", 
            columns="Trace_ID", 
            values=coor_cols,
            sort=False
        )
        covs = []
        for c in coor_cols:
            X = pivoted[c].values.T
            X = X - np.nanmean(X, axis=1)[:,None]
            covs.append(sample_covar_ma(X))
        return np.stack(covs)
        
    @staticmethod
    def preprocess(
        d1d:np.ndarray, 
        arr:np.ndarray, 
        nstds:float=4
    ) -> np.ndarray:
        """Stratify by 1D genomic distance and remove entries that are
        `nstds` standard deviations away from the mean. Also remove 1D
        genomic distance bias by dividing each stratum by its std.

        Parameters
        ----------
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.
        arr : (n, d, p, p) np.ndarray
            Pairwise difference matrices of all chromatin traces.
        nstds : float, optional
            Number of standard deviations to use as the cutoff. Remove 
            values more than `nstds` standard deviations away from the
            mean, by default 4.

        Returns
        -------
        np.ndarray
            Processed pairwise difference matrices.
        """
        d1map = d1d[None,:] - d1d[:,None]
        narr = arr.copy()
        for i in range(arr.shape[1]):
            for d in np.unique(d1map[d1map>0]):
                idx = np.where(d1map==d)
                med_std = np.nanmedian(np.square(arr[:,i,*idx]))**.5
                
                sidx = np.where(np.abs(d1map)==d)
                sub_arr = np.where(
                    np.abs(arr[:,i,*sidx]) > med_std*nstds, 
                    np.nan, arr[:,i,*sidx]
                )
                narr[:,i,*sidx] = sub_arr/np.nanstd(sub_arr)
                
        return narr
        
    @staticmethod
    def ij_background(i:int, j:int, d1d:np.ndarray) -> np.ndarray:
        """The entries that are between 25kb and 50kb away from the 
        (i,j) entry are treated as the background.
        
        Parameters
        ----------
        i : int
            Index of the first locus.
        j : int
            Index of the second locus.
        d1d : (p,) np.ndarray
            Array of 1D genomic locations of imaging loci.

        Returns
        -------
        (p, p) np.ndarray
            A boolean matrix with background entries equal to `True`.
        """
        kept = np.zeros((len(d1d), len(d1d)), dtype="bool")
        
        # select the outer square
        a = (np.abs(d1d-d1d[i])<=50e3).astype("int")
        b = (np.abs(d1d-d1d[j])<=50e3).astype("int")
        kept[(a[:,None]+b[None,:])==2] = True
        
        # exclude the inner square
        a = (np.abs(d1d-d1d[i])<=25e3).astype("int")
        b = (np.abs(d1d-d1d[j])<=25e3).astype("int")
        kept[(a[:,None]+b[None,:])==2] = False
        
        kept[np.tril_indices_from(kept)] = False
        return kept
        
    def test_func_axis(
        self, 
        loop_arr:np.ndarray, 
        bkgd_arr:np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform F-test in each axis.

        Parameters
        ----------
        loop_arr : (n, d) np.ndarray
            Pairwise differences of all chromatin traces and all axes at
            the loop entry.
        bkgd_arr : (n, d, r) np.ndarray
            Pairwise differences of all chromatin traces and all axes at
            the local background entries, where `r` is the number of 
            entries in the local background model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Size (d,) test statistics and p-values.
        """
        loop_arr = loop_arr[np.all(~np.isnan(loop_arr), axis=1)]
        bkgd_arr = np.vstack(bkgd_arr.transpose(0, 2, 1))
        bkgd_arr = bkgd_arr[np.all(~np.isnan(bkgd_arr), axis=1)]
        
        var_loop_est = np.mean(np.square(loop_arr), axis=0)
        var_unloop_est = np.mean(np.square(bkgd_arr), axis=0)
        
        f_stats = var_loop_est/var_unloop_est
        nalleles_loop = loop_arr.shape[0]
        nalleles_unloop = bkgd_arr.shape[0]
        f_pvals = stats.f.cdf(f_stats, nalleles_loop, nalleles_unloop)
        return f_stats, f_pvals
        
    def test_func(
        self, 
        loop_arr:np.ndarray, 
        bkgd_arr:np.ndarray
    ) -> Tuple[float, float]:
        """Call :func:`AxisWiseF.test_func_axis` and combine p-values
        by Cauchy aggregation test.

        Parameters
        ----------
        loop_arr : (n, d) np.ndarray
            Pairwise differences of all chromatin traces and all axes at
            the loop entry.
        bkgd_arr : (n, d, r) np.ndarray
            Pairwise differences of all chromatin traces and all axes at
            the local background entries, where `r` is the number of 
            entries in the local background model.

        Returns
        -------
        Tuple[float, float]
            ACT statistic and p-value.
        """
        f_pvals = self.test_func_axis(loop_arr, bkgd_arr)[1]
        
        act_stat = np.sum(self.weights*np.tan((0.5 - f_pvals)*np.pi))
        p_val_act = 1 - stats.cauchy.cdf(act_stat)
        return act_stat, p_val_act
    
    @staticmethod
    def filter_summits(
        summits:np.ndarray, 
        freq_mat:np.ndarray, 
        idx:np.ndarray, 
        max_i:int
    ):
        """Keep all summits, no filtering.

        Parameters
        ----------
        summits : (p, p) np.ndarray
            Boolean matrix with entries identified as summits equal to
            `True`, will be modified inplace.
        freq_mat : (p, p) np.ndarray
            Contact frequency matrix, where the (i,j) entry is defined 
            as the number of traces with the (i,j) distances below a
            threshold.
        idx : (k, k) np.ndarray
            The indices of the loop candidates associated with the 
            summit considered, returned by `np.where(candidate_mat)`.
        max_i : int
            The index of the summit within the candidates, so `max_i` is
            between 1 and k-1.
        """
        i1, i2 = idx[0][max_i], idx[1][max_i]
        summits[i1,i2] = summits[i2,i1] = True
        
        
class AxisWiseChi2(AxisWiseF):
    
    @staticmethod
    def ij_background(i, j, d1d):
        pass
    
    def test_func(self, loop_arr, _):
        loop_arr = loop_arr[np.all(~np.isnan(loop_arr), axis=1)]
        var_loop_est = np.var(loop_arr, ddof=0, axis=0)
        
        chi2_stats = var_loop_est * loop_arr.shape[0]
        chi2_pvals = stats.chi2.cdf(chi2_stats, df=loop_arr.shape[0]-1)
        
        act_stat = np.sum(self.weights*np.tan((0.5 - chi2_pvals)*np.pi))
        p_val_act = 1 - stats.cauchy.cdf(act_stat)
        return act_stat, p_val_act
    
    
class AxisWiseT(AxisWiseF):
    
    def test_func(self, loop_arr, bkgd_arr):
        loop_arr = loop_arr[np.all(~np.isnan(loop_arr), axis=1)]
        bkgd_arr = np.vstack(bkgd_arr.transpose(0, 2, 1))
        bkgd_arr = bkgd_arr[np.all(~np.isnan(bkgd_arr), axis=1)]
        
        loop_dist = np.abs(loop_arr).T
        unloop_dist = np.abs(bkgd_arr).T
        
        t_pvals = []
        for l, u in zip(loop_dist, unloop_dist):
            t_pvals.append(stats.ttest_ind(
                l, u, equal_var=False, alternative="less"
            )[1])
        t_pvals = np.array(t_pvals)
        
        act_stat = np.sum(self.weights*np.tan((0.5 - t_pvals)*np.pi))
        p_val_act = 1 - stats.cauchy.cdf(act_stat)
        return act_stat, p_val_act


class LoopCaller:
    """Call chromatin loops from multiplexed imaging data.

    Parameters
    ----------
    data : pd.DataFrame
        Data in FOF-CT_core format with no repeating rows.
    fdr_cutoff: float
        FDR cut-off for chromatin loops, by default .1.
    cut_lo : float, optional
        Minimum loop size (1D genomic distance between the first locus
        and the second locus), by default 1e5.
    cut_up : float, optional
        Maximum loop size, by default 1e6.
    gap : float, optional
        Loop candidates `gap` away from each other are considered 
        candidates for the same summit, by default 50e3.
    freq : float, optional
        Use the median of all pairwise distances with 1D genomic 
        distance equal to `freq` as the cutoff to define the frequency
        map, by default 25e3.
    """
    def __init__(
        self,
        data:pd.DataFrame,
        fdr_cutoff:float=.1,
        cut_lo:float=1e5,
        cut_up:float=1e6,
        gap:float=50e3,
        freq:float=25e3
    ):
        self._data = data
        self._fdr_cutoff = fdr_cutoff
        self.cut_lo = int(cut_lo)
        self.cut_up = int(cut_up)
        self.gap = int(gap)
        self.freq = int(freq)
        
    @property
    def fdr_cutoff(self):
        """FDR cut-off for chromatin loops."""
        return self._fdr_cutoff
        
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
            # out.append(out_c[out_c["summit"]])
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
            A dictionary with keys 'stat', 'pval', 'fdr', 'candidate', 
            'label', 'summit'. Values are (p,p) matrices.
        """
        chr_df = self._data[self._data["Chrom"]==chr_id]
        ltobj = ltclass(chr_df)
        chr_df_pivoted, arr = to_very_wide(chr_df)
        
        warnings.filterwarnings("ignore", "Mean of empty slice")
        d1d = chr_df_pivoted.index.values
        arr = ltobj.preprocess(d1d, arr)
        I = d1d.shape[0]
        result = {
            "stat":np.zeros((I,I))*np.nan,
            "pval":np.zeros((I,I))*np.nan
        }
        for i in range(I):
            for j in range(i+1, I):
                dist_1d = np.abs(d1d[i] - d1d[j])
                if dist_1d < self.cut_lo or dist_1d > self.cut_up:
                    continue
                kept = ltobj.ij_background(i, j, d1d)
                if np.sum(kept) == 0:
                    continue
                stat, pval = ltobj.test_func(arr[:,:,i,j], arr[:,:,kept])
                result["stat"][i,j] = result["stat"][j,i] = stat
                result["pval"][i,j] = result["pval"][j,i] = pval
                    
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
        
        self._fdr_to_summit(result, self._fdr_cutoff, d1d, arr, ltobj)
            
        return result
    
    def _to_freq_mat(self, arr, d1d):
        """Calculate the contact frequency matrix."""
        dmaps = np.sqrt(np.sum(np.square(arr), axis=1))
        freq_dists = dmaps[:,np.abs(d1d[:,None] - d1d[None,:])==self.freq]
        warnings.filterwarnings("ignore", r".*invalid value")
        freq_mat = np.sum(dmaps<np.nanmean(freq_dists), axis=0)/\
            np.sum(~np.isnan(dmaps), axis=0)
        return freq_mat
    
    def _fdr_to_summit(self, result, fdr_cutoff, d1d, arr, ltobj):
        """Identify summits from the FDR cutoff. `ltobj` can be 
        uninitialized.
        """
        result["candidate"] = result["fdr"] < fdr_cutoff
        # print("FDR cutoff:", ltobj.fdr_cutoff)
        labeled = self.spread_label(result["candidate"], d1d)
        
        summits = np.zeros_like(labeled, dtype="bool")
        freq_mat = self._to_freq_mat(arr, d1d)
        for lab in np.unique(labeled[~np.isnan(labeled)]):
            # keep only the triu part to compare p values
            idx = np.where(np.triu(labeled)==lab)
            max_i = np.argmin(result["pval"][idx])
            ltobj.filter_summits(summits, freq_mat, idx, max_i)
        
        result["label"] = labeled
        result["summit"] = summits
    
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