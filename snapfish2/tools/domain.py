import warnings
from typing import Literal
import numpy as np
import pandas as pd
import dask.array as da
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from statsmodels.stats import multitest as multi
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from anndata import AnnData, concat

from .loop import DiffLoop
from ..utils.eval import (
    axis_weight, 
    median_pdist, 
    joint_filter_normalize
)


class TADCaller:
    """TAD calling class.

    Parameters
    ----------
    fdr_cutoff : float, optional
        Boundary peaks with FDR below this cutoff are defined as TAD
        boundaries, by default 0.1.
    window : float, optional
        Domain size in bp to calculate intra- and inter-domain distance,
        by default 1e5.
    tree : bool, optional
        Whether to return hierarchical TADs, by default True.
    min_tad_size : float, optional
        The minimum TAD size allowed, by default 0.
    prominence : float, optional
        Required only if method is "insulation", by default None.
        Least height difference in normalized insulation score in order 
        for the locus to be defined as a peak. Passed to
        :func:`~scipy.signal.find_peaks`.
    distance : float, optional
        Required only if method is "insulation", by default None.
        Least number of loci between two peaks. Passed to
        :func:`~scipy.signal.find_peaks`.
    method :  Literal["pval", "insulation"], optional
        TAD calling algorithm used, by default "pval".
    """
    def __init__(
        self,
        fdr_cutoff:float=0.1,
        window:float=1e5,
        tree:bool=True,
        min_tad_size:float=0,
        prominence:float|None=None,
        distance:int|None=None,
        method:Literal["pval", "insulation"]="pval"
    ):
        self._fdr_cutoff = fdr_cutoff
        self._window = int(window)
        self._tree = tree
        self._min_tad_size = int(min_tad_size)
        self._method = method
        
        self._prominence = prominence
        self._distance = distance
        
    @property
    def fdr_cutoff(self) -> float:
        """float : FDR cut-off for TAD boundaries."""
        return self._fdr_cutoff
    
    @property
    def window(self) -> int:
        """int : size of the window used to compute inter/intra 
        contacts.
        """
        return self._window
    
    @property
    def tree(self) -> bool:
        """bool : Call hierarchical TADs."""
        return self._tree
    
    @property
    def min_tad_size(self) -> int:
        """int : Minimum TAD size."""
        return self._min_tad_size
    
    @property
    def prominence(self) -> float:
        """float : Prominence of peaks."""
        return self._prominence
    
    @property
    def distance(self) -> int:
        """int : Minimum distance between peaks."""
        return
    
    @property
    def method(self) -> Literal["pval", "insulation"]:
        """str : TAD calling method used."""
        return self._method
    
    def call_tads(self, adata:AnnData) -> pd.DataFrame:
        """Call TADs from `adata`.

        Parameters
        ----------
        adata : AnnData
            adata of a single chromosome, created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.

        Returns
        -------
        pd.DataFrame
            A dataframe with length equal to the number of loci. The 
            column `peak` defines whether the position is a boundary.
        """
        if self._method == "pval":
            return self.by_pval(adata)
        if self._method == "insulation":
            return self.by_insulation(adata)
        
    def by_pval(
        self,
        adata:AnnData
    ) -> pd.DataFrame:
        """Call TADs by thresholding FDR values.

        Parameters
        ----------
        adata : AnnData
            adata of a single chromosome, created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.

        Returns
        -------
        pd.DataFrame
            A dataframe with length equal to the number of loci. The 
            column `peak` defines whether the position is a boundary.
        """
        val_cols = ["X", "Y", "Z"]
        count = np.stack([adata.varp[f"count_{c}"] for c in val_cols])
        entry_var = np.stack([adata.varp[f"var_{c}"] for c in val_cols])
        weights = axis_weight(adata, inplace=False)
        d1d = adata.var["Chrom_Start"].values
        d1de = adata.var["Chrom_End"].values

        # Jumps in d1d, no entries for inter region
        warnings.filterwarnings("ignore", ".*divide by zero")
        warnings.filterwarnings("ignore", ".*invalid value encountered")
        rows = []
        window = self._window
        for i in range(len(d1d)):
            if d1d[i] - d1d[0] < window/2 or d1d[-1] - d1d[i] < window/2:
                rows.append([d1d[i], d1de[i], np.nan, np.nan])
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
            rows.append([
                d1d[i], d1de[i], *f_stats, 
                *f_pvals, act_stat, p_val_act
            ])
            
        result = pd.DataFrame(rows, columns=[
            "Chrom_Start", "Chrom_End", "stat_x", "stat_y", "stat_z",
            "pval_x", "pval_y", "pval_z", "stat", "pval"
        ])
        
        fdr_arr = result["pval"].copy()
        avail_idx = ~np.isnan(fdr_arr)
        fdr_arr[avail_idx] = multi.multipletests(
            fdr_arr[avail_idx], method="fdr_bh"
        )[1]
        result["fdr"] = fdr_arr
        result["fdr_peak"] = result["fdr"] < self._fdr_cutoff
        
        result["raw_peak"] = False
        result.loc[find_peaks(result.stat.values)[0],"raw_peak"] = True
        result["peak"] = result["raw_peak"]&result["fdr_peak"]
        
        result["c1"] = adata.uns["Chrom"]
        cols = ["c1"] + result.columns.drop("c1").to_list()
        return result[cols]
        
    def by_insulation(
        self,
        adata:AnnData
    ) -> pd.DataFrame:
        """Call TADs by insulation score. Method from
        Su, J.-H., Zheng, P., Kinrot, S. S., Bintu, B. & Zhuang, X. 
        Genome-Scale Imaging of the 3D Organization and Transcriptional 
        Activity of Chromatin. Cell 182, 1641-1659.e26 (2020).
        
        Parameters
        ----------
        adata : AnnData
            adata of a single chromosome, created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.

        Returns
        -------
        pd.DataFrame
            A dataframe with length equal to the number of loci. The
            column `peak` defines whether the row is a boundary.
        """
        if "med_dist" in adata.varp:
            median_dist = adata.varp["med_dist"]
        else:
            median_dist = median_pdist(adata, False)
        d1d = adata.var["Chrom_Start"].values
        d1de = adata.var["Chrom_End"].values
        
        scores = []
        window = self._window
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
            prominence=self._prominence, 
            distance=self._distance
        )[0]
        peaks = np.zeros_like(scores, dtype="bool")
        peaks[peak_idx] = True
        result = pd.DataFrame({
            "Chrom_Start":d1d, "Chrom_End":d1de,
            "insulation":scores, "peak":peaks
        })
        
        result["c1"] = adata.uns["Chrom"]
        cols = ["c1"] + result.columns.drop("c1").to_list()
        return result[cols]
    
    def to_bedpe(
        self,
        result:pd.DataFrame
    ) -> pd.DataFrame | None:
        """Convert TAD calling result to a dataframe where each row is a
        TAD (a pair of boundaries instead of a single boundary).

        Parameters
        ----------
        result : pd.DataFrame
            Result returned by :func:`by_insulation` or 
            :func:`by_pval`.

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
        val = result.reset_index(drop=True)
        chr_id = val.c1.iloc[0]
        
        # Use "stat" if called by_pval, otherwise "insulation"
        score_col = "stat" if "stat" in val.columns else "insulation"
        
        # Set the first and the last loci as peaks
        val.loc[val.index[0], "peak"] = True
        val.loc[val.index[-1], "peak"] = True
        
        sub_df = val[val["peak"]]
        level = 0
        prev, ip = sub_df.iloc[0], 0
        for i, row in sub_df.iloc[1:].iterrows():
            rows.append([
                chr_id, prev["Chrom_Start"], prev["Chrom_End"],
                chr_id, row["Chrom_Start"], row["Chrom_End"],
                prev[score_col], row[score_col],
                level, ip, i
            ])
            prev, ip = row, i
        
        if self._tree:
            df = sub_df.copy()
            idxs = df[1:-1].sort_values(score_col).index
            for i in idxs:
                iloc = df.index.tolist().index(i)
                df = df.drop(i, axis=0)
                level += 1
                r1, r2 = df.iloc[iloc-1], df.iloc[iloc]
                rows.append([
                    chr_id, r1["Chrom_Start"], r1["Chrom_End"],
                    chr_id, r2["Chrom_Start"], r2["Chrom_End"],
                    r1[score_col], r2[score_col],
                    level, df.index[iloc-1], df.index[iloc]
                ])
        
        cols = [
            "c1", "s1", "e1", "c2", "s2", "e2", 
            f"{score_col}1", f"{score_col}2",
            "level", "idx1", "idx2"
        ]
        out_df = pd.DataFrame(rows, columns=cols)
        out_df = out_df[(out_df["s2"] - out_df["s1"])>self._min_tad_size]
        return out_df
        
        
class ABCaller:
    """Call A/B compartments from multiplexed imaging data using PCA.
    
    Parameters
    ----------
    min_comp_size : float
        Minimum compartment size in bp.
    cutoff: float, optional
        Required only if method is "pca".
        Distance below `cutoff` is defined as contact, by dafault None.
    sigma : float, optional
        Required only if method is "pca".
        Gaussian kernel size, by default 1.
    method : Literal["axes", "pca"], optional
        A/B compartment calling algorithm used, by default "axes".
    """
    def __init__(
        self, 
        min_cpmt_size:float,
        cutoff:float|None=None,
        sigma:float|None=1,
        method:Literal["axes", "pca"]="axes"
    ):
        self._min_cpmt_size = min_cpmt_size
        self._cutoff = cutoff
        self._sigma = sigma
        self._method = method
        
    def call_cpmt(self, adata:AnnData) -> pd.DataFrame:
        """Call A/B compartments from `adata`.

        Parameters
        ----------
        adata : AnnData
            adata of a single chromosome, created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.

        Returns
        -------
        pd.DataFrame
            A dataframe with length equal to the number of locus. The 
            column `cpmt` indicates A/B compartment assignments: 0
            indicates A compartment and 1 indicates B compartment.
        """
        if self._method == "axes":
            return self.by_axes_pc(adata)
        if self._method == "pca":
            return self.by_first_pc(adata)
        
    def by_axes_pc(self, adata:AnnData) -> pd.DataFrame:
        """Call A/B compartments by weighting the 2nd PC from different
        axes.
        
        Parameters
        ----------
        adata : AnnData
            adata of a single chromosome, created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.

        Returns
        -------
        pd.DataFrame
            A dataframe with length equal to the number of locus. The 
            column `cpmt` indicates A/B compartment assignments: 0
            indicates A compartment and 1 indicates B compartment.
        """
        d1d = adata.var["Chrom_Start"].values
        if self._min_cpmt_size > np.ptp(d1d):
            raise ValueError(
                "Minimum compartment size larger than the region."
            )
        med_sq = np.stack([
            adata.varp[f"var_{c}"] for c in ["X", "Y", "Z"]
        ])
        # Already normalized. 
        # Hollowed or not does not matter. Same eigenspace.
        V = np.linalg.eigh(np.exp(-med_sq))[1][:,:,-2]
        wts = axis_weight(adata, inplace=False)
        wtV = V*wts[:,None]
        cpmt_arr = KMeans(
            n_clusters=2, 
            random_state=0    
        ).fit_predict((wtV).T)
        
        cpmt_arr = ABCaller._filter_small_cmpt(
            cpmt_arr=cpmt_arr,
            min_cpmt_size=self._min_cpmt_size,
            d1d=d1d
        )
        
        idx0 = (cpmt_arr==0)[:,None]*(cpmt_arr==0)[None,:]
        med0 = np.nanmedian(med_sq[:,idx0], axis=1)

        idx1 = (cpmt_arr==1)[:,None]*(cpmt_arr==1)[None,:]
        med1 = np.nanmedian(med_sq[:,idx1], axis=1)

        mode = stats.mode((med0 < med1).astype("int64"))[0]
        if mode == 1:
            cpmt_arr = np.where(cpmt_arr==0, 1, 0)
        
        result = pd.DataFrame({
            "s1":d1d, 
            "e1":adata.var["Chrom_End"].values,
            "cpmt":cpmt_arr, 
            "eig_x":V[0], "eig_y":V[1], "eig_z":V[2],
            "wteig_x":wtV[0], "wteig_y":wtV[1], "wteig_z":wtV[2],
        })
        return result
        
    def by_first_pc(self, adata:AnnData) -> dict:
        """Call A/B compartments by first PCA. Adopted from
        Su, J.-H., Zheng, P., Kinrot, S. S., Bintu, B. & Zhuang, X. 
        Genome-Scale Imaging of the 3D Organization and Transcriptional 
        Activity of Chromatin. Cell 182, 1641-1659.e26 (2020).
        
        Parameters
        ----------
        adata : AnnData
            adata of a single chromosome, created by 
            :func:`snapfish2.pp.FOF_CT_Loader.create_adata`.

        Returns
        -------
        pd.DataFrame
            A dataframe with length equal to the number of locus. The 
            column `cpmt` indicates A/B compartment assignments: 0
            indicates A compartment and 1 indicates B compartment.
        """
        d1d = adata.var["Chrom_Start"].values
        if self._min_cpmt_size > np.ptp(d1d):
            raise ValueError(
                "Minimum compartment size larger than the region."
            )
            
        n = adata.shape[0]
        c = round(1 * (n * 3 * 8 / 1e9) ** -0.5)
        X = da.from_array(np.stack([
            adata.layers[c] for c in ["X", "Y", "Z"]
        ]), chunks=(3,n,c))
        arr = X[:,:,:,None] - X[:,:,None,:]
        # (n, p, p) pairwise distance matrices for each trace
        dist_mats = da.sqrt(da.sum(da.square(arr), axis=0))
        # (p, p) pseudo contact map
        contact_mat = da.sum(dist_mats < self._cutoff, axis=0)\
            /da.sum(~da.isnan(dist_mats), axis=0)
        contact_mat = contact_mat.compute()
            
        d1d_mat = np.abs(d1d[:,None] - d1d[None,:])
        uidx = np.triu_indices_from(d1d_mat, 1)
        d1d_flat = d1d_mat[uidx]
        contact_flat = contact_mat[uidx]
        kept = (d1d_flat > 0) & (contact_flat > 0)
        log_d1d = np.log(d1d_flat[kept])
        lr = stats.linregress(log_d1d, np.log(contact_flat[kept]))
        
        contact_expected = np.exp(log_d1d*lr.slope + lr.intercept)
        contact_flat_norm = np.zeros_like(kept, dtype="float64")
        contact_flat_norm[kept] = contact_flat[kept]/contact_expected
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
        smooth_norm_mat = gaussian_filter(contact_mat_norm, self._sigma)
        contact_corr = np.corrcoef(smooth_norm_mat)
        
        # X = contact_corr - np.nanmean(contact_corr, axis=0)
        # U, S, Vh = np.linalg.svd(X)
        # cpmt_arr = ((np.sign(U[1]*S[1])+1)/2).astype("int")
        pc1 = PCA(1).fit_transform(contact_corr).reshape(-1)
        cpmt_arr = ((np.sign(pc1)+1)/2).astype("int")
        
        cpmt_arr = ABCaller._filter_small_cmpt(
            cpmt_arr=cpmt_arr,
            min_cpmt_size=self._min_cpmt_size,
            d1d=d1d
        )
        
        idx0 = (cpmt_arr==0)[:,None]*(cpmt_arr==0)[None,:]
        med0 = np.nanmedian(contact_mat_norm[idx0])

        idx1 = (cpmt_arr==1)[:,None]*(cpmt_arr==1)[None,:]
        med1 = np.nanmedian(contact_mat_norm[idx1])

        if med0 > med1:
            cpmt_arr = np.where(cpmt_arr==0, 1, 0)
        
        result = pd.DataFrame({
            "s1":d1d, 
            "e1":adata.var["Chrom_End"].values,
            "cpmt":cpmt_arr
        })
        return result
    
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
        chr_id = self._adata1.uns["Chrom"]
        region_df = region_df[
            (region_df["c1"]==chr_id)&(region_df["c2"]==chr_id)
        ]
        joint_filter_normalize(self._adata1, self._adata2)
        f_pvals = self.entry_pvals(self._adata1, self._adata2)
        adataj = concat([self._adata1, self._adata2])
        weights = axis_weight(adataj, inplace=False)
        
        result = self.region_pvals(
            weights=weights,
            f_pvals=f_pvals,
            d1d=self._adata1.var["Chrom_Start"].values,
            region_df=region_df
        )
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
    
    def to_bedpe(
        self, 
        result:pd.DataFrame, 
        fdr_cutoff:float=0.1
    ) -> pd.DataFrame:
        """Add a column `diff` to the result dataframe, indicating
        whether the domain is differentially significant.

        Parameters
        ----------
        result : pd.DataFrame
            Differential domain result from :func:`diff_region`.
        fdr_cutoff : float, optional
            Domains with differential FDR below this cutoff are marked
            as significant, by default 0.1.

        Returns
        -------
        pd.DataFrame
            Same format as `result`, with an additional column `diff`.
        """
        result = result.copy()
        result["diff"] = result["fdr"] < fdr_cutoff
        return result