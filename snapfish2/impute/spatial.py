import numpy as np
import pandas as pd
from scipy import stats

from ..utils.func import (
    _sample_covar_heteroPCA,
    sample_covar_ma
)


def exp_covar(x:np.ndarray, s:float, r:float):
    """Exponential covariance function.

    Parameters
    ----------
    x : np.ndarray
        Pairwise difference of all locations.
    s : float
        Overall scale parameter.
    r : float
        Scale parameter that divides difference.

    Returns
    -------
    _type_
        _description_
    """
    return s * np.exp(-np.abs(x)/r)

class TraceSample:
    def __init__(
            self,
            N:int, 
            R:int, 
            K:np.ndarray,
            tau:np.array,
            obs_p:float, 
            shifts:np.ndarray,
            random_state:int=None
    ):
        """Initialize a TraceSample with input parameters.

        Parameters
        ----------
        N : int
            The number of individual traces to sample.
        R : int
            The number of loci on each trace.
        K : np.ndarray
            The covariance matrix of shape (R, R).
        tau : np.array
            Array of length 3. Measurement error in each axis.
        obs_p : float
            The observed probability/detection efficiency.
        shifts : (3, N) np.ndarray
            Shifts of each trace.
        random_state : int, optional
            Random state for sample generation, by default None.
        """
        self.N = N
        self.R = R
        self.K = K
        self.tau_sq = np.square(tau)
        self.obs_p = obs_p
        self.shifts = shifts
        self.rs = random_state
        self.__rvs__()

    @property
    def samples(self):
        return self._samples
    
    def __rvs__(self):
        """Generate N random traces, each of length R.
        """
        sample_ls = []

        for i, t in enumerate(self.tau_sq):
            # measurement error + shift for each trace
            Sigma = self.K + np.identity(self.R)*t
            sample = stats.multivariate_normal.rvs(
                cov=Sigma, size=self.N, random_state=self.rs
            ) + self.shifts[i][:,None]

            # missing at random with observed probability self.obs_p
            mask_ber = stats.bernoulli.rvs(
                self.obs_p, size=self.R*self.N, random_state=self.rs
            )
            mask_ber = mask_ber.reshape(sample.shape)
            sample = np.where(mask_ber==1, sample, np.nan)

            sample_ls.append(sample)

        samples = np.stack(sample_ls)
        self._samples = samples


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
            Shet = _sample_covar_heteroPCA(Sini, r=r, T=T)
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