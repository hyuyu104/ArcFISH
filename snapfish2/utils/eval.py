import tempfile
import shutil
from pathlib import Path
import numpy as np
from anndata import AnnData, concat
import dask.array as da


def _filter_pdiff(adata:AnnData, tmp_path:str, nstds:float):
    n = adata.shape[0]
    # Size of n*d*c*c = 1GB
    c = round(1 * (n * 3 * 8 / 1e9) ** -0.5)
    
    val_cols = ["X", "Y", "Z"]
    X = da.from_array(np.stack([
        adata.layers[c] for c in val_cols
    ]), chunks=(3,n,c))
    arr = X[:,:,:,None] - X[:,:,None,:]
    
    med_sq = da.nanmedian(da.square(arr), axis=1).compute()
    for i, v in enumerate(val_cols):
        adata.varp[f"raw_var_{v}"] = med_sq[i]

    d1d = adata.var["Chrom_Start"].values
    d1map = d1d[None,:] - d1d[:,None]

    med_stds = np.zeros_like(med_sq, dtype="float64")
    for dd in np.unique(d1map[d1map>0]):
        idx = np.where(d1map==dd)
        med_std = np.nanmedian(med_sq[:,*idx], axis=1)**.5
        med_stds[:,*idx] = med_std[:,None]
    # Fill the lower triangle
    med_stds = med_stds + med_stds.transpose((0,2,1))
    med_stds = med_stds[:,None,:,:]

    arr[da.abs(arr) > med_stds*nstds] = np.nan
    arr.to_zarr(tmp_path, overwrite=True)
    
    
def _normalized_dask_arr(adata:AnnData, tmp_path:str) -> da:
    arr = da.from_zarr(tmp_path)
    count = da.sum(~da.isnan(arr), axis=1).compute()
    for i, c in enumerate(["X", "Y", "Z"]):
        adata.varp[f"count_{c}"] = count[i]
    
    d1d = adata.var["Chrom_Start"].values
    d1map = d1d[None,:] - d1d[:,None]

    count_by1d = np.zeros_like(count, dtype="int64")
    for dd in np.unique(d1map[d1map>0]):
        idx = np.where(d1map==dd)
        count_by1d[:,*idx] = np.sum(count[:,*idx], axis=1)[:,None]
    count_by1d = count_by1d + count_by1d.transpose((0,2,1))
    count_by1d[:,*np.diag_indices(count_by1d.shape[2])] = 1
    wt = count/count_by1d

    wt_entry_mean = da.nanmean(arr, axis=1).compute()*wt
    mean_by1d = np.zeros_like(wt_entry_mean, dtype="float64")
    for dd in np.unique(d1map[d1map>0]):
        idx = np.where(d1map==dd)
        mean_by1d[:,*idx] = np.sum(wt_entry_mean[:,*idx], axis=1)[:,None]
    mean_by1d = mean_by1d + mean_by1d.transpose((0,2,1))

    wt_entry_var = da.nanmean(
        (arr-mean_by1d[:,None,:,:])**2, axis=1
    ).compute()*wt
    std_by1d = np.zeros_like(wt_entry_var, dtype="float64")
    for dd in np.unique(d1map[d1map>0]):
        idx = np.where(d1map==dd)
        std_by1d[:,*idx] = np.sqrt(
            np.sum(wt_entry_var[:,*idx], axis=1)    
        )[:,None]
    std_by1d = std_by1d + std_by1d.transpose((0,2,1))
    std_by1d[:,*np.diag_indices(std_by1d.shape[2])] = 1
    return arr/std_by1d[:,None,:,:]
    
    
def _normalize_by1d(adata:AnnData, tmp_path:str):
    var_norm = da.nanmean(da.square(
        _normalized_dask_arr(adata, tmp_path)
    ), axis=1).compute()
    for i, c in enumerate(["X", "Y", "Z"]):
        adata.varp[f"var_{c}"] = var_norm[i]


def filter_normalize(
    adata:AnnData, 
    tmp_path:Path|None=None, 
    nstds:float=4,
    keep_zarr:bool=False
):
    """Filter outliers and normalized by 1D genomic distance.
    
    Filter out entries with pairwise difference `nstds` away from the
    median pairwise difference stratified by 1D genomic distance. 
    Then normalize the pairwise difference by the standard deviations
    stratified by 1D genomic distance.
    
    Store the intermediate pairwise difference tensor as a .zarr file
    and applies dask operations. The intermediate .zarr file can be 
    large (e.g. ~6 GB for 1000 traces each of dimension 1000), so make
    sure the storage space is large enough.
    
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
    tmp_path : Path | None, optional
        Where to store the intermediate pairwise difference. If None, 
        will create a temporary directory with 
        :class:`~tempfile.TemporaryDirectory` and remove it after all 
        computations are done, by default None.
    nstds : float, optional
        Values larger than this number of standard deviations away from
        the median will be removed, by default 4.
    keep_zarr : bool, optional
        Whether to keep the zarr file if `tmp_path` is not None, by
        default False.
    """
    if tmp_path is not None:
        _filter_pdiff(
            adata=adata,
            tmp_path=tmp_path,
            nstds=nstds
        )
        _normalize_by1d(
            adata=adata,
            tmp_path=tmp_path
        )
        if not keep_zarr:
            shutil.rmtree(tmp_path)
    else:
        with tempfile.TemporaryDirectory() as tmp_path:
            _filter_pdiff(
                adata=adata,
                tmp_path=tmp_path,
                nstds=nstds
            )
            _normalize_by1d(
                adata=adata,
                tmp_path=tmp_path
            )


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
    nstds = kwargs.get("nstds", 4)
    adataj = concat(args)
    adataj.var = args[0].var.copy()
    adataj.uns["Chrom"] = args[0].uns["Chrom"]
    
    with tempfile.TemporaryDirectory() as tmp_path:
        _filter_pdiff(adata=adataj, tmp_path=tmp_path, nstds=nstds)
        narr = _normalized_dask_arr(adata=adataj, tmp_path=tmp_path)
        
        for adata in args:
            fil = adataj.obs_names.isin(adata.obs_names)
            count = da.sum(~da.isnan(narr[:,fil,:,:]), axis=1).compute()
            var_norm = da.nanmean(da.square(
                narr[:,fil,:,:]
            ), axis=1).compute()
            for i, c in enumerate(["X", "Y", "Z"]):
                adata.varp[f"count_{c}"] = count[i]
                adata.varp[f"var_{c}"] = var_norm[i]


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