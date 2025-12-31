from typing import Callable
import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from arcfish.utils.load import FOF_CT_Loader
from arcfish.utils.eval import median_pdist

def cast_to_distmat(
    X:np.ndarray, func:Callable=np.nanmean
) -> np.ndarray:
    """Cast the input array to a p by p matrix. Specfically,
    
    1. X is a 1d array: treat X as the flattened upper triangle of a
    (p,p) matrix and refill it into a p*p matrix.
    
    2. X is a (p,p) symmetric matrix: return X.
    
    3. X is a (p,d) asymmetric matrix, where p might or might not equal 
    to d: treat as the coordinates of a single trace, calculate the 
    pairwise distance matrix.
    
    4. X is a (n,p,p) matrix, where X[i] is symmetric: apply func to each
    entry (e.g. func = np.nanmean, then this is averaging each entry).
    
    5. X is a (n,p,d) matrix, where p might or might not equal to d: first
    convert to (n,p,p) by applying 3 to X[i] and then apply 4.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    func : Callable, optional
        How to calculate average when the dimension of the input is at 
        least 3, by default np.nanmean.

    Returns
    -------
    (p,p) np.ndarray
        Output p by p symmetric matrix.
    """
    if len(X.shape) == 1:
        N = int((1 + (1 + 8 * len(X)) ** 0.5) / 2)
        mat = np.zeros((N, N))*np.nan
        mat[np.triu_indices(N, 1)] = X
        mat.T[np.triu_indices(N, 1)] = mat[np.triu_indices(N, 1)]
    elif len(X.shape) == 2 and X.shape[0] == X.shape[1] and \
        np.allclose(X[~np.isnan(X)], X.T[~np.isnan(X)]):
            mat = X
    elif len(X.shape) == 2:
        # p x d
        outer_diff = np.stack([
            x[:,None] - x[None,:] for x in X.T
        ])
        mat = np.sqrt(np.sum(np.square(outer_diff), axis=0))
    elif X.shape[1] == X.shape[2] and \
        np.allclose(X[~np.isnan(X)], X.transpose(0,2,1)[~np.isnan(X)]):
            # print("very same")
            mat = func(X, axis=0)
    else:
        arrs = []
        for x in X:
            outer_diff = np.stack([
                a[:,None] - a[None,:] for a in x.T
            ])
            d = np.sqrt(np.sum(np.square(outer_diff), axis=0))
            arrs.append(d)
        arrs = np.stack(arrs)
        mat = func(arrs, axis=0)
    return mat
            
            
def rotate_df(df:pd.DataFrame, theta:float=-45) -> pd.DataFrame:
    """Rotate the 2D coordinates of a DataFrame by `theta` degrees. The
    input data frame must have columns "x" and "y".

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 2D coordinates and values.
    theta : float, optional
        Rotation angle in degree, by default 45.

    Returns
    -------
    pd.DataFrame
        Rotated dataframe with new columns "x_rot" and "y_rot".
    """
    rotation = np.array([
        [np.cos(theta/180*np.pi), -np.sin(theta/180*np.pi)],
         [np.sin(theta/180*np.pi), np.cos(theta/180*np.pi)]
    ])
    vals = rotation@df[["x", "y"]].values.T
    df["x_rot"] = vals[0]
    df["y_rot"] = vals[1]
    return df