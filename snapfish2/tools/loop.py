from abc import ABC, abstractmethod
import logging
import warnings
from typing import Tuple
import numpy as np
import pandas as pd
from anndata import AnnData, concat
from scipy import stats
from statsmodels.stats import multitest as multi

from ..utils.eval import axis_weight, joint_filter_normalize


class LoopTestAbstract(ABC):
    """Abstract class for each test alternative. Every testing object
    passed to :class:`LoopCaller` must implement the methods in this 
    class.
    
    Parameters
    ----------
    adata : AnnData
        adata of a single chromosome, created by 
        :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.
    """
    @abstractmethod
    def __init__(self, adata:AnnData):
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
        """How to define loop summit from each loop cluster.
        
        Loop candidates within the same cluster will have the same 
        number in `result["label"]`. Add a boolean (p,p) matrix to
        `result["summit"]`, where entries equal to `True` are the 
        selected loop summit for each cluster.

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
    adata : AnnData
        adata of a single chromosome, created by 
        :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.
    """
    def __init__(self, adata:AnnData):
        self._d1d = adata.var["Chrom_Start"].values
        X = np.stack([adata.layers[c] for c in ["X", "Y", "Z"]])
        arr = X[:,:,:,None] - X[:,:,None,:]
        self._pdists = np.sqrt(np.sum(np.square(arr), axis=0))
    
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
            
            loop_dist = self._pdists[:,i,j]
            loop_dist = loop_dist[~np.isnan(loop_dist)]
            
            bkgd_dist = self._pdists[:,kept]
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
        a potential summit. Filter summits by contact frequency. 
        
        1. If the summit is a singleton (i.e. from only one candidate), 
        then it is marked as summit if contact frequency is larger than 
        1/2. 
        
        2. If the summit is not a singleton (i.e. from multiple 
        candidates), then it is marked as summit if contact frequency is
        larger than 1/3.

        Parameters
        ----------
        result : dict
            The result dictionary to add summit.
        """
        d1d, dmaps = self._d1d, self._pdists
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
    """Perform axis-wise F-test and combine p-values by aggregated 
    Cauchy test. 

    Parameters
    ----------
    adata : AnnData
        adata of a single chromosome, created by 
        :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.
    """
    def __init__(self, adata:AnnData):
        self._d1d = adata.var["Chrom_Start"].values
        val_cols = ["X", "Y", "Z"]
        self._var = np.stack([adata.varp[f"var_{c}"] for c in val_cols])
        self._count = np.stack([adata.varp[f"count_{c}"] for c in val_cols])
        
        if "weight" not in adata.uns:
            axis_weight(adata)
        self._wt = np.array([adata.uns["weight"][c] for c in val_cols])
        
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
        also append `axis_stat` and `axis_pval`, which are both (3,p,p)
        arrays.

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

        result["axis_stat"] = np.zeros((3,p,p), dtype="float64")*np.nan
        result["axis_pval"] = np.zeros((3,p,p), dtype="float64")*np.nan

        for i, j in zip(*np.where((d1map>=cut_lo)&(d1map<=cut_up))):
            bkgd_map = self.ij_background(i, j, self._d1d, outer_cut)
            if np.sum(bkgd_map) == 0:
                continue
            
            num_unloop = np.sum(self._count[:,bkgd_map], axis=1)
            wts = self._count[:,bkgd_map]/num_unloop[:,None]
            denom = np.sum(wts*self._var[:,bkgd_map], axis=1)
            
            f_stats = self._var[:,i,j]/denom
            result["axis_stat"][:,i,j] = result["axis_stat"][:,j,i] = f_stats
            f_pvals = stats.f.cdf(f_stats, self._count[:,i,j], num_unloop)
            result["axis_pval"][:,i,j] = result["axis_pval"][:,j,i] = f_pvals
            
        weights = self._wt[:,None,None]
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
    """Class for chromatin loop calling.

    Parameters
    ----------
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
        fdr_cutoff:float=0.1,
        cut_lo:float=1e5,
        cut_up:float=1e6,
        gap:float=50e3,
        outer_cut:float=50e3
    ):
        self._fdr_cutoff = fdr_cutoff
        self._cut_lo = int(cut_lo)
        self._cut_up = int(cut_up)
        
        self._gap = int(gap)
        self._outer_cut = int(outer_cut)

    @property
    def fdr_cutoff(self) -> float:
        """float : FDR cut-off for chromatin loops."""
        return self._fdr_cutoff
    
    @property
    def loop_range(self) -> Tuple[int, int]:
        """Tuple[int, int] : Loop size considered."""
        return (self._cut_lo, self._cut_up)
    
    @property
    def gap(self) -> int:
        """int : Loop candidates `gap` away from each other are 
        considered candidates for the same summit.
        """
        return self._gap
    
    @property
    def outer_cut(self) -> int:
        """int : Loci with 1D genomic distance within `outer_cut` from 
        the target locus is included in the local background.
        """
        return self._outer_cut
    
    def call_loops(
        self, 
        adata:AnnData,
        ltclass:LoopTestAbstract=AxisWiseF
    ) -> dict:
        """Call chromatin loops from a single chromosome.

        Parameters
        ----------
        adata : AnnData
            adata of a single chromosome, created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.
        ltclass : LoopTestAbstract
            Test method used, by default :class:`AxisWiseF`.

        Returns
        -------
        dict
            A dictionary with keys stat, pval, fdr, candidate, label, 
            summit. Values are (p,p) matrices. Might also contain other
            values depending on the test class used.
        """
        test_class = ltclass(adata)
        
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
        result["label"] = self.spread_label(
            result["candidate"], 
            adata.var["Chrom_Start"].values
        )
        
        test_class.append_summit(result)
            
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
        adata:AnnData
    ) -> pd.DataFrame:
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
        adata : AnnData
            adata of a single chromosome, created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.
            
        Returns
        -------
        pd.DataFrame
            Output dataframe in bedpe format.
        """
        a = adata.var["Chrom_Start"].values
        a1 = a[:,None] - np.zeros_like(a)
        a2 = a - np.zeros_like(a)[:,None]

        b = adata.var["Chrom_End"].values
        b1 = b[:,None] - np.zeros_like(b)
        b2 = b - np.zeros_like(b)[:,None]

        uidxs = np.triu_indices_from(a1, 1)

        out_df = pd.DataFrame(
            np.stack([a1, b1, a2, b2])[:,uidxs[0], uidxs[1]].T,
            columns=["s1", "e1", "s2", "e2"]
        )
        out_df["c1"] = out_df["c2"] = adata.uns["Chrom"]
        out_df = out_df[["c1", "s1", "e1", "c2", "s2", "e2"]]

        for k, v in result.items():
            if v.shape == (adata.n_vars, adata.n_vars):
                out_df[k] = v[uidxs]
        return out_df


class DiffLoop:
    """Differential analysis of chromatin loops.

    Parameters
    ----------
    adata1 : pd.DataFrame
        Condition 1 data adata created by 
        :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.
    adata2 : pd.DataFrame
        Condition 2 data adata created by 
        :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.
    """
    def __init__(
        self, 
        adata1:AnnData, 
        adata2:AnnData,
    ):
        if adata1.uns["Chrom"] != adata2.uns["Chrom"]:
            raise ValueError("Chrom different for adata1 and adata2.")
        self._adata1 = adata1
        self._adata2 = adata2
    
    def diff_loops(
        self, 
        summit_df:pd.DataFrame,
        s:float=5e3
    ) -> dict:
        """Call differential chromatin interactions.

        Parameters
        ----------
        summit_df : pd.DataFrame
            List of loops to check. Have columns c1, s1, e1, c2, s2, e2.
        s : float, optional
            Gaussian kernel size in bp, by default 5e3.

        Returns
        -------
        dict
            Differential loop testing result with keys f_pvals, 
            agg_pvals, and fdr. Values are (p,p) matrices.
        """
        warnings.filterwarnings("ignore", r".+initializing view")
        result = {}
        chr_id = self._adata1.uns["Chrom"]
        summit_chr = summit_df[
            (summit_df["c1"]==chr_id)&(summit_df["c2"]==chr_id)
        ]
        joint_filter_normalize(self._adata1, self._adata2)
        
        result["f_pvals"] = self.entry_pvals(self._adata1, self._adata2)
        adataj = concat([self._adata1, self._adata2])
        weights = axis_weight(adataj, inplace=False)
        
        agg_p_mat, idx = self.loop_pvals(
            weights, 
            result["f_pvals"], 
            self._adata1.var["Chrom_Start"].values, 
            summit_chr, 
            s=s
        )
        result["agg_pvals"] = agg_p_mat
        
        result["fdr"] = self.pval_to_fdr(agg_p_mat, idx)
        return result
        
    @staticmethod
    def entry_pvals(
        adata1:AnnData,
        adata2:AnnData
    ) -> np.ndarray:
        """Calculate axis-wise entry-wise p-values by F-tests.

        Parameters
        ----------
        adata1 : pd.DataFrame
            Condition 1 data adata created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.
        adata2 : pd.DataFrame
            Condition 2 data adata created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.

        Returns
        -------
        (3, p, p) np.ndarray
            Axis-wise entry-wise p-values.
        """
        warnings.filterwarnings("ignore", ".*divide")
        f_pvals = []
        for c in ["X", "Y", "Z"]:
            f_stats = adata1.varp[f"var_{c}"]/adata2.varp[f"var_{c}"]
            count1 = adata1.varp[f"count_{c}"]
            count2 = adata2.varp[f"count_{c}"]
            f_pvals.append(stats.f.cdf(f_stats, count1, count2))
        f_pvals = np.stack(f_pvals)
        return f_pvals
        
    @staticmethod
    def loop_pvals(
        weights:np.ndarray, 
        f_pvals:np.ndarray, 
        d1d:np.ndarray, 
        summit_chr:pd.DataFrame, 
        s:float=5e3
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculate p-value for each loop considered. The p-value is 
        aggregated 1) from all axis by weights inversely proportional to 
        measurement errors 2) from entries within the same axis by a 
        Gaussian kernel with size `s`.

        Parameters
        ----------
        weights : (d,) np.ndarray
            Weight for each axis.
        f_pvals : (d, p, p) np.ndarray
            Entry-wise p-values returned by func:`entry_pvals`.
        d1d : np.ndarray
            1D genomic locations.
        summit_chr : pd.DataFrame
            List of loops within the chromosome to check.
        s : float, optional
            Gaussian kernel size in bp, by default 5e3

        Returns
        -------
        Tuple[(p, p) np.ndarray, Tuple[np.ndarray, np.ndarray]]
            Aggregated p-values and indices of the summits.
        """
        # Select entries corresponding to loop summits
        d1d_sr = pd.Series(np.arange(len(d1d)), index=d1d)
        iidx = d1d_sr[summit_chr.s1].values
        jidx = d1d_sr[summit_chr.s2].values
        all_idx = zip(iidx, jidx)
        
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
        
    def to_bedpe(
        self, 
        result:dict, 
        fdr_cutoff:float=0.1
    ) -> pd.DataFrame:
        """Convert differential loop result to dataframe, similar format
        as `summit_df`.

        Parameters
        ----------
        result : dict
            Dictionary returned by :func:`diff_loops`.
        fdr_cutoff : float, optional
            Loops with p-values below this cutoff are defined as 
            differential loop, by default 0.1.
            
        Returns
        -------
        pd.DataFrame
            Output dataframe in bedpe format.
        """
        d1 = self._adata1.var.values
        s2, s1 = np.meshgrid(d1.T[0], d1.T[0])
        e2, e1 = np.meshgrid(d1.T[1], d1.T[1])

        result_arr = np.stack(
            [s1, e1, s2, e2, result["agg_pvals"], result["fdr"]]
        )
        arr_triu = result_arr[:,*np.triu_indices_from(result_arr[0], 1)]
        # Keep only rows with FDR not being NaN
        avail_idx = np.where(~np.isnan(arr_triu[-1]))

        cols = ["s1", "e1", "s2", "e2", "pval", "fdr"]
        bedpe_df = pd.DataFrame(arr_triu[:,*avail_idx].T, columns=cols)
        bedpe_df = bedpe_df.astype({c:"int" for c in cols[:4]})
        bedpe_df["c1"] = bedpe_df["c2"] = self._adata1.uns["Chrom"]
        
        bedpe_df = bedpe_df[
            ["c1", "s1", "e1", "c2", "s2", "e2", "pval", "fdr"]
        ]
        bedpe_df["log_fdr"] = np.log(bedpe_df["fdr"])
        bedpe_df["diff"] = bedpe_df["fdr"] < fdr_cutoff
        
        logging.info(
            f"Found {np.sum(bedpe_df["diff"])} differential loops, "\
            + f"while {np.sum(~bedpe_df["diff"])} are not differential loops."
        )
        
        return bedpe_df