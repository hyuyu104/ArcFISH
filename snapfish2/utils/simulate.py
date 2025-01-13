import numpy as np
from scipy import stats

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