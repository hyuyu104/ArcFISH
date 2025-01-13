import warnings
import numpy as np
from scipy.integrate import quad


def sample_covar_loh(X:np.ndarray) -> np.ndarray:
    """Naive covariance estimator when data is incomplete.

    Parameters
    ----------
    X : (N, R) np.ndarray
        Data matrix with NaN entries.

    Returns
    -------
    (R, R) np.ndarray
        Covariance estimate.
    """
    Avail = ~np.isnan(X)
    p_hat = np.mean(Avail)
    N = X.shape[0]

    Y = np.where(np.isnan(X), 0, X)
    G = (Y.T@Y)/(N*p_hat**2)
    # the diagonal should be divided by p, not p^2
    np.fill_diagonal(G, np.diag(G)*p_hat)
    return G


def sample_covar_ma(X:np.ndarray) -> np.ndarray:
    """Raw covariance matrix with masked arrays.

    Parameters
    ----------
    X : (N, R) np.ndarray
        Data matrix with NaN entries.

    Returns
    -------
    (R, R) np.ndarray
        Covariance estimate.
    """
    Avail = (~np.isnan(X)).astype("int")
    num_avail = Avail.T@Avail

    Xma = np.ma.masked_invalid(X)
    G = np.ma.dot(Xma.T, Xma)/num_avail
    return G.data
    

def sample_covar_heteroPCA(
        G:np.ndarray, r:int=5, T:int=100
) -> np.ndarray:
    """Estimate covariance with HeteroPCA.

    Parameters
    ----------
    G : (R, R) np.ndarray
        Sample covariance matrix from sample_covar_loh.
    r : int, optional
        Rank of the covariance matrix, by default 5.
    T : int, optional
        Number of iterations, by default 100.

    Returns
    -------
    (R, R) np.ndarray
        HeteroPCA estimate of the covariance matrix.
    """
    Gdiag = np.copy(G)
    # 1st step of HeteroPCA: set diagonal to 0
    np.fill_diagonal(Gdiag, 0)

    for _ in range(T):
        L, V = np.linalg.eig(Gdiag)
        Lkept, Vkept = np.diag(L[:r]), V[:,:r]
        Gest = Vkept@Lkept@Vkept.T
        # impute the diagonal with the new estimate
        np.fill_diagonal(Gdiag, np.diag(Gest))
    
    return Gest


def matrixA(R:int) -> np.ndarray:
    """Linear transformation that maps the raw covariance matrix to the 
    covariance matrix of data with shifts subtracted.

    Parameters
    ----------
    R : int
        Dimension of the covariance matrix.

    Returns
    -------
    (R^2, R^2) np.ndarray
        The calculated linear transformation.
    """
    s = np.zeros(R*R)
    s[:R] = 1
    s = np.stack([np.roll(s, i*R) for i in range(R)])
    s1 = np.tile(s, (R, 1))
    s2 = np.repeat(s, R, axis=0)
    A = np.identity(R*R) + 1/R**2 - s1/R - s2/R
    return A


def machenko_pastur_pdf(x, g, s) -> float:
    hp = (1 + g**.5)**2
    hm = (1 - g**.5)**2
    if x <= s*hm or x >= s*hp:
        return 0
    f = ((x - s*hm)*(s*hp - x))**.5/(2*np.pi*s*x*min(g, 1))
    return f


def machenko_pastur_cdf(x, g, s) -> float:
    warnings.filterwarnings("ignore")
    F = quad(
        machenko_pastur_pdf, 
        a=-np.inf, b=x, args=(g, s), 
        limit=500
    )[0]
    warnings.filterwarnings("default")
    return F


def machenko_pastur_quantile(k, g, s) -> float:
    if k < 0 or k > 1:
        raise ValueError("Quantile must be between 0 and 1.")
    # Slightly increase the search range to resolve numerical issues
    u = (s*(1 + g**.5)**2) * 1.0001
    l = (s*(1 - g**.5)**2) * 0.9999
    while True:
        val = (u+l)/2
        q = machenko_pastur_cdf(val, g, s)
        if np.abs(q - k) < 1e-6:
            return val
        elif q > k:
            u = val
        else:
            l = val


def bema_var_hat(lambdas, p, n, alpha):
    pt = min(p, n)
    l = int(np.ceil(alpha * pt))
    u = int(np.floor((1 - alpha) * pt))
    qks = np.array([
        machenko_pastur_quantile(1-k/pt, g=p/n, s=1) 
        for k in range(l, u+1)
    ])
    return np.sum(qks*lambdas[l:u+1])/np.sum(np.square(qks))
    