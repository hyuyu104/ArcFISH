from typing import Literal
import warnings
from io import BytesIO, TextIOWrapper
import gzip
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from anndata import AnnData
import dask.array as da
import requests

from ..utils.eval import axis_weight, filter_normalize
from ..tools.func import overlap


class ABCaller:
    """Call A/B compartments from multiplexed imaging data using PCA.
    
    Parameters
    ----------
    min_comp_size : float
        Minimum compartment size in bp.
    ref_genome : str, optional
        Reference genome assembly ID used to assign A/B compartment
        based on clustering result, by default None. 
        
        If None, use the average pairwise distance to assign A/B 
        compartment (smaller distance is A and larger distance is B). 
        It is highly recommended to pass in a reference genome string.
        
        See available assembly IDs at: 
        `UCSC Genome browser <https://genome.ucsc.edu/cgi-bin/hgGateway>`_.
        Common assembly IDs are: "hg19", "hg38", "mm10".
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
        ref_genome:str|None=None,
        cutoff:float|None=None,
        sigma:float|None=1,
        method:Literal["axes", "pca"]="axes"
    ):
        self._min_cpmt_size = min_cpmt_size
        self._ref_genome = self._ref_genome_parser(ref_genome)
        self._cutoff = cutoff
        self._sigma = sigma
        self._method = method
        
    @property
    def tss(self) -> pd.DataFrame|None:
        """Transcript start sites (TSS) of the reference genome."""
        if hasattr(self, "_tss"):
            return self._tss
        return None
        
    def _ref_genome_parser(self, ref_genome:str|None) -> str:
        """Add transcript start sites as `self._tss ` """
        if ref_genome is None:
            return None
        ref_genome = ref_genome.lower()
        pre = "https://hgdownload.soe.ucsc.edu/goldenPath"
        url = f"{pre}/{ref_genome}/bigZips/genes/{ref_genome}.refGene.gtf.gz"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            warnings.warn(
                f"{ref_genome} not found in UCSC. \n" + 
                "See available assembly IDs at: " +
                "https://genome.ucsc.edu/cgi-bin/hgGateway"
            )
            return None
        
        with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz:
            gtf = pd.read_csv(
                TextIOWrapper(gz), sep="\t", 
                header=None, usecols=[0, 2, 3, 4]
            )
            gtf.columns = ["Chrom", "feature", "Chrom_Start", "Chrom_End"]
            tss = gtf[gtf["feature"]=="transcript"].drop(columns="feature")
            del gtf
        self._tss = tss
        return ref_genome
    
    def _assign_cpmt_ref_genome(self, result, med_sq):
        """Determine which cluster is A and which is B by comparing the
        the median of the normalized contact matrix or the TSS density
        using a reference genome."""
        cpmt_arr = result["cpmt"].values
        if self._ref_genome is None:
            idx0 = (cpmt_arr==0)[:,None]*(cpmt_arr==0)[None,:]
            med0 = np.nanmedian(med_sq[:,idx0], axis=1)

            idx1 = (cpmt_arr==1)[:,None]*(cpmt_arr==1)[None,:]
            med1 = np.nanmedian(med_sq[:,idx1], axis=1)

            mode = stats.mode((med0 < med1).astype("int64"))[0]
            if mode == 1:
                cpmt_arr = np.where(cpmt_arr==0, 1, 0)
        else:
            chr_id = result["c1"].values[0]
            tss = self._tss[self._tss["Chrom"]==chr_id][
                ["Chrom_Start", "Chrom_End"]
            ].values
            ints0 = result[result["cpmt"]==0][["s1", "e1"]].values
            overlap0 = sum(overlap(tss, ints0))
            ints1 = result[result["cpmt"]==1][["s1", "e1"]].values
            overlap1 = sum(overlap(tss, ints1))
            if overlap1 > overlap0:
                cpmt_arr = np.where(cpmt_arr==0, 1, 0)
        return cpmt_arr
        
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
            if "var_X" not in adata.varp:
                filter_normalize(adata)
            return self.by_axes_pc(adata)
        if self._method == "pca":
            return self.by_first_pc(adata)
    
    @staticmethod
    def _spectral_clustering(
        adata:AnnData, 
        med_sq:np.ndarray, 
        n_clusters:int
    ) -> np.ndarray:
        """Spectral clustering on axis eigenvectors. Use -1 to 
        -n_clusters eigenvectors."""
        V = np.linalg.eigh(np.exp(-med_sq))[1][:,:,-n_clusters:-1]
        wts = axis_weight(adata, inplace=False)
        wtV = V*wts[:,None,None]
        wtV = wtV.transpose(0, 2, 1).reshape(-1, wtV.shape[1])
        cpmt_arr = KMeans(
            n_clusters=n_clusters, 
            random_state=0    
        ).fit_predict((wtV).T)
        return cpmt_arr
        
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
        
        result = pd.DataFrame({
            "c1":[adata.uns["Chrom"]]*len(d1d),
            "s1":d1d, 
            "e1":adata.var["Chrom_End"].values,
            "cpmt":cpmt_arr, 
            "eig_x":V[0], "eig_y":V[1], "eig_z":V[2],
            "wteig_x":wtV[0], "wteig_y":wtV[1], "wteig_z":wtV[2],
        })
        result["cpmt"] = self._assign_cpmt_ref_genome(
            result=result, med_sq=med_sq
        )
        return result
        
    def by_first_pc(self, adata:AnnData) -> pd.DataFrame:
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
        
        result = pd.DataFrame({
            "c1":[adata.uns["Chrom"]]*len(d1d),
            "s1":d1d, 
            "e1":adata.var["Chrom_End"].values,
            "cpmt":cpmt_arr
        })
        med_sq = np.stack([
            adata.varp[f"var_{c}"] for c in ["X", "Y", "Z"]
        ])
        result["cpmt"] = self._assign_cpmt_ref_genome(
            result=result, med_sq=med_sq
        )
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