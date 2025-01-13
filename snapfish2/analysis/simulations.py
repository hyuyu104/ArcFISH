import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest as multi


def add_test(tests_dict:dict):
    """A double decorator to add tests.

    Parameters
    ----------
    tests_dict : dict
        Dictionary to store available tests. Key is the name of the test, 
        value is the function.

    Returns
    -------
    function
        Decorator.
    """
    # print("executed 1") -> executed when imported
    def decorator(func):
        # print("executed 2") -> executed when imported
        tests_dict[func.__name__] = func
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


class LoopTest:

    available_tests = {}

    def __init__(
            self, 
            sigma_loop:np.array, 
            sigma_unloop:np.array, 
            sigma_error:np.array,
            conf_level:float
    ):
        """
        Parameters
        ----------
        sigma_loop : np.array
            The looped state standard deviation (in each axis, `length=3`).
        sigma_unloop : np.array
            The unlooped state standard deviation (in each axis, `length=3`).
        sigma_error : np.array
            The measurement error in each axis.
        conf_level : float
            Confidence level/(1-type I error) of the tests.
        """
        self.sigma_loop = sigma_loop 
        self.sigma_unloop = sigma_unloop
        self.sigma_error = sigma_error
        self.type1 = 1 - conf_level

    def sample(
            self, 
            sigma_loop:np.array, 
            sigma_unloop:np.array, 
            nalleles_loop:int, 
            nalleles_unloop:int
    ) -> dict:
        """Generate one sample and run all tests.

        Parameters
        ----------
        sigma_loop : np.array
            The looped state standard deviation (in each axis, `length=3`).
        sigma_unloop : np.array
            The unlooped state standard deviation (in each axis, `length=3`).
        nalleles_loop : int
            Sample size of the looped state.
        nalleles_unloop : int
            Sample size of the unlooped state.

        Returns
        -------
        dict
            Keys are the names of the tests and values are True (reject) or False (accept).
        """
        means = np.array([0]*3)
        # Variance of the looped state
        var_loop = np.square(sigma_loop) + np.square(self.sigma_error)
        # Variance of the unlooped state
        var_unloop = np.square(sigma_unloop) + np.square(self.sigma_error)
        # Generate samples (assume x, y, z are independent)
        samples_loop = stats.multivariate_normal.rvs(means, np.diag(var_loop), size=nalleles_loop)
        samples_unloop = stats.multivariate_normal.rvs(means, np.diag(var_unloop), size=nalleles_unloop)
        
        reject_dict = {}

        for name, func in self.available_tests.items():
            reject_dict[name] = func(samples_loop, samples_unloop, self.type1)

        return reject_dict
    
    def evaluate_tests(
            self, 
            nalleles_loop:int, 
            nalleles_unloop:int, 
            repeat_times:int=5000
    ) -> pd.DataFrame:
        """Calculate TP, TF, FP, FN values for all tests.

        Parameters
        ----------
        nalleles_loop : int
            Sample size of the looped state.
        nalleles_unloop : int
            Sample size of the unlooped state.
        repeat_times : int, optional
            Test on how many samples, by default 5000.

        Returns
        -------
        pd.DataFrame
            TP, TF, FP, FN values.
        """
        rej_results = [
            self.sample(
                sigma_loop=self.sigma_loop,
                sigma_unloop=self.sigma_unloop,
                nalleles_loop=nalleles_loop,
                nalleles_unloop=nalleles_unloop
            ) for _ in range(repeat_times)
        ]
        rej_df = pd.DataFrame(rej_results).mean().to_frame("TP")
        rej_df["FN"] = 1 - rej_df["TP"]

        null_sigma = (self.sigma_loop + self.sigma_unloop)/2
        acc_results = [
            self.sample(
                sigma_loop=null_sigma,
                sigma_unloop=null_sigma,
                nalleles_loop=nalleles_loop,
                nalleles_unloop=nalleles_unloop
            ) for _ in range(repeat_times)
        ]
        acc_df = pd.DataFrame(acc_results).mean().to_frame("FP")
        acc_df["TN"] = 1 - acc_df["FP"]

        result_df = pd.concat([rej_df, acc_df], axis=1).round(2)

        return result_df
    
    @staticmethod
    @add_test(available_tests)
    def t_test(
        samples_loop:np.ndarray, 
        samples_unloop:np.ndarray,
        type1:float
    ) -> bool:
        """Use distance instead of axis difference as values. Two-sample t-test.

        Parameters
        ----------
        samples_loop : np.ndarray
            Multivariate normal sample for the looped state.
        samples_unloop : np.ndarray
            Multivariate normal sample for the unlooped state.
        type1 : float
            Type I error rate.

        Returns
        -------
        bool
            Whether the null hypothesis (d_l >= d_u) is rejected.
        """
        dist_loop = np.linalg.norm(samples_loop, axis=1)
        dist_unloop = np.linalg.norm(samples_unloop, axis=1)
        t_pval = stats.ttest_ind(dist_loop, dist_unloop, alternative="less", equal_var=False)[1]
        return t_pval <= type1
    
    @staticmethod
    def _f_p_vals(samples_loop:np.ndarray, samples_unloop:np.ndarray) -> np.array:
        """Calculate F-statistics and p-values for each axis."""
        # calculate F-statistics
        # H_0: \sigma^2_loop >= \sigma^2_unloop
        # H_1: \sigma^2_loop < \sigma^2_unloop]
        var_loop_est = np.var(samples_loop, ddof=1, axis=0)# - np.square(self.sigma_error)
        var_unloop_est = np.var(samples_unloop, ddof=1, axis=0)# - np.square(self.sigma_error)

        f_stats = var_loop_est/var_unloop_est

        nalleles_loop = len(samples_loop) - 1
        nalleles_unloop = len(samples_unloop) - 1
        f_pvals = stats.f.cdf(f_stats, nalleles_loop, nalleles_unloop)

        return f_pvals
    
    @staticmethod
    @add_test(available_tests)
    def f_test_bonferroni(
        samples_loop:np.ndarray, 
        samples_unloop:np.ndarray, 
        type1:float
    ) -> bool:
        """F-test for each axis and combine p-values by Bonferroni correction.

        Parameters
        ----------
        samples_loop : np.ndarray
            Multivariate normal sample for the looped state.
        samples_unloop : np.ndarray
            Multivariate normal sample for the unlooped state.
        type1 : float
            Type I error rate.

        Returns
        -------
        bool
            Whether the null hypothesis (sigma^2_1 >= sigma^2_2) is rejected.
        """
        f_pvals = LoopTest._f_p_vals(samples_loop=samples_loop, samples_unloop=samples_unloop)
        # Bonferroni correction
        f_rej = np.any(f_pvals <= type1/3)
        return f_rej
    
    @staticmethod
    @add_test(available_tests)
    def f_test_fdr(
        samples_loop:np.ndarray, 
        samples_unloop:np.ndarray, 
        type1:float
    ) -> bool:
        """F-test for each axis and combine p-values by FDR.

        Parameters
        ----------
        samples_loop : np.ndarray
            Multivariate normal sample for the looped state.
        samples_unloop : np.ndarray
            Multivariate normal sample for the unlooped state.
        type1 : float
            Type I error rate.

        Returns
        -------
        bool
            Whether the null hypothesis (sigma^2_1 >= sigma^2_2) is rejected.
        """
        f_pvals = LoopTest._f_p_vals(samples_loop=samples_loop, samples_unloop=samples_unloop)
        # FDR correction
        f_rej = np.any(multi.multipletests(f_pvals, method="fdr_bh")[1] <= type1)
        return f_rej
    
    @staticmethod
    @add_test(available_tests)
    def f_test_act(
        samples_loop:np.ndarray, 
        samples_unloop:np.ndarray, 
        type1:float
    ) -> bool:
        """F-test for each axis and combine p-values by Aggregated Cauchy Test.

        Parameters
        ----------
        samples_loop : np.ndarray
            Multivariate normal sample for the looped state.
        samples_unloop : np.ndarray
            Multivariate normal sample for the unlooped state.
        type1 : float
            Type I error rate.

        Returns
        -------
        bool
            Whether the null hypothesis (sigma^2_1 >= sigma^2_2) is rejected.
        """
        f_pvals = LoopTest._f_p_vals(samples_loop=samples_loop, samples_unloop=samples_unloop)

        weights = [1, 1, 1/9]
        weights = weights/np.sum(weights)
        act_stat = np.sum(weights*np.tan((0.5 - f_pvals)*np.pi))
        f_rej = stats.cauchy.cdf(act_stat) >= 1-type1
        
        return f_rej
    
    @staticmethod
    @add_test(available_tests)
    def f_test_sum(
        samples_loop:np.ndarray,
        samples_unloop:np.ndarray,
        type1:float
    ) -> bool:
        weights = [1, 1, 1]
        weights = weights/np.sum(weights)
        
        wt_loop = np.sum(samples_loop*weights, axis=1)
        var_loop = np.mean(np.square(wt_loop))
        wt_unloop = np.sum(samples_unloop*weights, axis=1)
        var_unloop = np.mean(np.square(wt_unloop))
        
        f_stat = var_loop/var_unloop
        rej = stats.f.cdf(f_stat, len(wt_loop), len(wt_unloop)) <= type1
        
        return rej