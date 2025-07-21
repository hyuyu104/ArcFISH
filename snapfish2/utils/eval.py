import warnings
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from anndata import AnnData, concat
import dask.array as da


def _lowess_multivariate(
    log_d1map:np.ndarray, 
    ys:np.ndarray
) -> np.ndarray:
    """Local linear regression based on 10% of the data."""
    strata_var = np.zeros((ys.shape[0], *log_d1map.shape))*np.nan
    uidx = np.triu_indices_from(log_d1map, k=1)
    
    for i, y in enumerate(ys):
        strata_var[i][uidx] = lowess(
            y, log_d1map[uidx], frac=.1, return_sorted=False
        )
        strata_var[i].T[uidx] = strata_var[i][uidx]
    return np.exp(strata_var)


def _filter_pdiff(adata:AnnData, arr:da.Array, log_d1map:np.ndarray):
    """Filter outliers based on pairwise difference."""
    raw_var = da.nanmedian(da.square(
        arr - da.nanmedian(arr, axis=1, keepdims=True)
    ), axis=1).compute()
    # 0 becomes -inf after taking log
    raw_var[np.isclose(raw_var, 0)] = np.nan
    
    diag_indices = np.diag_indices_from(raw_var[0])
    for i, v in enumerate("XYZ"):
        adata.varp[f"raw_var_{v}"] = raw_var[i]
        adata.varp[f"raw_var_{v}"][diag_indices] = 0
        
    uidx = np.triu_indices(log_d1map.shape[0], k=1)
    raw_strata_var = _lowess_multivariate(
        log_d1map, np.log(raw_var[:,*uidx])
    )
    
    strata_std = np.sqrt(raw_strata_var)[:,None,:,:]
    arr[da.abs(
        arr - da.nanmedian(arr, axis=1, keepdims=True)
    ) > strata_std*4] = np.nan
    
    
def _normalize_pdiff(adata:AnnData, arr:da.Array, log_d1map:np.ndarray):
    """Normalize pairwise difference by 1D genomic distance."""
    count = da.sum(~da.isnan(arr), axis=1).compute()
    for i, c in enumerate("XYZ"):
        adata.varp[f"count_{c}"] = count[i]

    filtered_var = da.nanmean(da.square(
        arr - da.nanmean(arr, axis=1, keepdims=True)
    ), axis=1).compute()
    
    uidx = np.triu_indices(log_d1map.shape[0], k=1)
    flat_filtered_var = filtered_var[:,*uidx]
    flat_filtered_var[np.isclose(flat_filtered_var, 0)] = np.nan
    
    filtered_strata_var = _lowess_multivariate(
        log_d1map, np.log(flat_filtered_var)
    )
    # Normalize by std <-> divide by strata variance
    normalized_var = filtered_var/filtered_strata_var
    for i, c in enumerate("XYZ"):
        adata.varp[f"var_{c}"] = normalized_var[i]


def filter_normalize(adata:AnnData):
    """Filter outliers and normalized by 1D genomic distance.
    
    Filter out entries with pairwise difference 4 standard deviations
    away from the median pairwise difference stratified by 1D genomic
    distance. The standard deviations are estimated from a local linear
    regression.
    
    The filtered pairwise difference is then normalized by the standard
    deviations stratified by 1D genomic distance. Similarly, the 
    standard deviations are estimated from a local linear regression.
    
    Append the followings to the `varp` field of `adata`:
    
    1. `raw_var_{X,Y,Z}`: median squared pairwise difference of each 
    axis.
    
    2. `var_{X,Y,Z}`: mean squared pairwise difference of each axis 
    after normalization.
    
    3. `count_{X,Y,Z}`: the number of available values for each pairwise
    distance entry after normalization.

    Parameters
    ----------
    adata : AnnData
        Object created by :func:`FOF_CT_Loader.create_adata`.
    """
    n = adata.shape[0]
    # Size of n*d*c*c = 1GB
    c = round(1 * (n * 3 * 8 / 1e9) ** -0.5)

    X = da.from_array(np.stack([
        adata.layers[c] for c in "XYZ"
    ]), chunks=(3,n,c))
    arr = X[:,:,:,None] - X[:,:,None,:]

    d1d = adata.var["Chrom_Start"].values
    warnings.filterwarnings("ignore", "divide by zero")
    log_d1map = np.log(np.abs(d1d[None,:] - d1d[:,None]))
    warnings.filterwarnings("default")
    
    _filter_pdiff(adata, arr, log_d1map)
    _normalize_pdiff(adata, arr, log_d1map)


def joint_filter_normalize(*args, **kwargs):
    """Filter and normalize multiple `adata`.
    
    For all the `adata` passed in, first concat them together and filter
    the outliers and normalize by 1D genomic distances as does in 
    :func:`filter_normalize`. Then for each `adata`, compute the entry
    variance and the number of available counts, append them to each
    `adata`. Thus the followings will be added to the `varp` field of 
    each `adata`:
    
    1. `var_{X,Y,Z}`: mean squared pairwise difference of each axis 
    after normalization.
    
    2. `count_{X,Y,Z}`: the number of available values for each pairwise
    distance entry after normalization.
    
    Parameters
    ----------
    args : AnnData
        Any number of `adata` objects.
    kwards : any
        Pass in `nstds=n` to change the filtering criterion like in 
        :func:`filter_normalize`, by default 4.
    """
    raise NotImplementedError("Joint filtering and normalization is" \
        + "not implemented yet.")
    # nstds = kwargs.get("nstds", 4)
    # adataj = concat(args)
    # adataj.var = args[0].var.copy()
    # adataj.uns["Chrom"] = args[0].uns["Chrom"]
    
    # arr = _filter_pdiff(adata=adataj, nstds=nstds)
    # narr = _normalized_dask_arr(adata=adataj, arr=arr)
    
    # for adata in args:
    #     fil = adataj.obs_names.isin(adata.obs_names)
    #     count = da.sum(~da.isnan(narr[:,fil,:,:]), axis=1).compute()
    #     var_norm = da.nanmean(da.square(
    #         narr[:,fil,:,:]
    #     ), axis=1).compute()
    #     for i, c in enumerate(["X", "Y", "Z"]):
    #         adata.varp[f"count_{c}"] = count[i]
    #         adata.varp[f"var_{c}"] = var_norm[i]


def axis_weight(adata:AnnData, inplace:bool=True) -> None|np.ndarray:
    """Compute axis weight.
    
    Add `weight` to `adata.uns`.

    Parameters
    ----------
    adata : AnnData
        Object created by :func:`FOF_CT_Loader.create_adata`.
    inplace : bool, optional
        Whether to add the weight to `adata` or return as an array, by
        default True.
    
    Returns
    -------
    None | np.array
        Return a length 3 array if `inplace` is False; otherwise return
        None.
    """
    weight = []
    for c in ["X", "Y", "Z"]:
        x = adata.layers[c]
        x0 = x - np.nanmean(x, axis=1)[:,None]
        xvars = np.nanmean(np.square(x0), axis=0)
        weight.append(1/np.nanmedian(xvars))
    weight = np.array(weight)/np.sum(weight)
    if not inplace:
        return weight
    adata.uns["weight"] = {
        c:weight[i] for i, c in enumerate(["X", "Y", "Z"])
    }
    

def median_pdist(adata:AnnData, inplace:bool=True) -> None|np.ndarray:
    """Compute median pairwise distance matrix.
    
    Add `med_dist` to `adata.varp`.

    Parameters
    ----------
    adata : AnnData
        Object created by :func:`FOF_CT_Loader.create_adata`.
    inplace : bool, optional
        Whether to add the median pairwise distance matrix to `adata` or 
        return as an array, by default True.
        
    Returns
    -------
    None | np.ndarray
        Return a p by p median pairwise distance matrix if inplace is False;
        otherwise return None.
    """
    n = adata.shape[0]
    # Size of n*d*c*c = 1GB
    c = round(1 * (n * 3 * 8 / 1e9) ** -0.5)
    
    val_cols = ["X", "Y", "Z"]
    X = da.from_array(np.stack([
        adata.layers[c] for c in val_cols
    ]), chunks=(3,n,c))
    arr = X[:,:,:,None] - X[:,:,None,:]
    med_dist = da.nanmedian(da.sqrt(
        da.sum(da.square(arr), axis=0)    
    ), axis=0).compute()
    
    if not inplace:
        return med_dist
    adata.varp["med_dist"] = med_dist
    
    
def pseudo_count(adata:AnnData, threshold:float) -> np.ndarray:
    """Thresholding pairwise distances to create a count matrix.

    Parameters
    ----------
    adata : AnnData
        Object created by :func:`FOF_CT_Loader.create_adata`.
    threshold : float
        Entries with pairwise distance smaller than this threshold are
        considered as a count.

    Returns
    -------
    np.ndarray
        p by p pseudo count matrix.
    """
    n = adata.shape[0]
    # Size of n*d*c*c = 1GB
    c = round(1 * (n * 3 * 8 / 1e9) ** -0.5)
    
    val_cols = ["X", "Y", "Z"]
    X = da.from_array(np.stack([
        adata.layers[c] for c in val_cols
    ]), chunks=(3,n,c))
    arr = X[:,:,:,None] - X[:,:,None,:]
    pdists = da.sqrt(da.sum(da.square(arr), axis=0))
    return da.sum(pdists < threshold, axis=0).compute()