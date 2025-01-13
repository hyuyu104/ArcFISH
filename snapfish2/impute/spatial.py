import numpy as np
import pandas as pd

from ..utils.covar import (
    sample_covar_heteroPCA,
    sample_covar_ma
)


class TraceCovEstimator:
    def __init__(self, data):
        self.data = data
        self._tau_sq_dict = {}
        self._diag_sample_size = {}
        self._prec_estimates = {}
        self._cov_initial = {}
        self._cov_estimates = {}
        self._tau_sq_hat = []
        self._chroms = pd.unique(data["Chrom"])

    @property
    def tau_sq_dict(self):
        return self._tau_sq_dict
    
    @property
    def diag_sample_size(self):
        return self._diag_sample_size
    
    @property
    def prec_estimates(self):
        return self._prec_estimates
    
    @property
    def cov_estimates(self):
        return self._cov_estimates
    
    @property
    def tau_sq_hat(self):
        return self._tau_sq_hat
    
    @property
    def chroms(self):
        return self._chroms

    def estimate_single_region(self, chrom, r=5, T=10):
        df = self.data[self.data["Chrom"]==chrom]
        df = df.pivot(index="Trace_ID", values=["X", "Y", "Z"], columns="Chrom_Start")
        arr = np.array([df[axis].values for axis in ["X", "Y", "Z"]])
        print(f"{chrom}: N = {arr.shape[1]}, R = {arr.shape[2]}")

        Ss = {"Sini":[], "Shet":[]}
        for X in arr:
            X = X - np.nanmean(X, axis=1)[:,None]
            Sini = sample_covar_ma(X)
            Ss["Sini"].append(Sini)
            Shet = sample_covar_heteroPCA(Sini, r=r, T=T)
            Ss["Shet"].append(Shet)
        self._cov_initial[chrom] = Ss["Sini"]

        diag_diffs, sample_sizes = [], []
        prec_ls = []
        for Sini, Shet in zip(Ss["Sini"], Ss["Shet"]):
            # diagonal difference is an unbiased estimator of \tau^2
            diag_diff = np.diag(Sini) - np.diag(Shet)
            diag_diffs.append(diag_diff)

            ns = np.sum(~np.isnan(arr[0]), axis=0)
            sample_sizes.append(ns)

            # precision matrix estimator
            L, V = np.linalg.eig(Sini)
            prec_hat = V[:,:r] @ np.diag(1/L[:r]) @ V[:,:r].T
            prec_ls.append(prec_hat)
        
        self._tau_sq_dict[chrom] = np.array(diag_diffs)
        self._diag_sample_size[chrom] = np.array(sample_sizes)
        self._prec_estimates[chrom] = np.array(prec_ls)

    def estimate_all_regions(self, r=5, T=10):
        for chrom in self._chroms:
            self.estimate_single_region(chrom, r=r, T=T)

        # estimate \tau^2 using all regions
        for i in range(3):
            samples_sizes = np.concatenate(
                [self.diag_sample_size[c][i] for c in self._chroms]
            )
            tau_sqrt = np.concatenate(
                [self.tau_sq_dict[c][i] for c in self._chroms]
            )
            # pooled variance estimator
            tau_sq_est = np.sum(samples_sizes*tau_sqrt)/np.sum(samples_sizes)
            self._tau_sq_hat.append(tau_sq_est)

    def estimate_K(self):
        for chrom in self._chroms:
            Kws = []
            for t, c in zip(self.tau_sq_hat, self._cov_initial[chrom]):
                K_est = c + t/c.shape[0]
                K_est = K_est - np.identity(c.shape[0])*t
                # print(f"{np.median(c)} -> {np.median(K_est)}")

                # weighted by the inverse of variance
                Kw = K_est/t
                Kws.append(Kw)

            # inverse-variance method to aggregate three axes
            Kws_sum = np.sum(Kws, axis=0)
            K_hat = Kws_sum/np.sum(1/np.array(self.tau_sq_hat))
            self._cov_estimates[chrom] = K_hat


class ReferenceFreeKriging:
    def __init__(self, tce:TraceCovEstimator):
        self.data = tce.data
        self.tce = tce

    def trace_kriging(self):
        pass