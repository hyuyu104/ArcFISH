from typing import Tuple
from typing import Callable
from pathlib import Path
import shutil
import tempfile
import csv
import numpy as np
import pandas as pd
from anndata import AnnData
import dask.array as da


class FOF_CT_Loader:
    """Read in FOF_CT file(s).
    
    For FOF_CT-core files, call :func:`create_adata` to create an 
    :class:`~anndata:anndata.AnnData`. The x, y, and z coordinates of
    each chromatin trace are stored as layers. If multiple file paths
    are passed in, merge traces from all the files.
    
    When having multiple files, will add a prefix to `Trace_ID` to make 
    it unique. If `path` is a list, will add a single number prefix to 
    `Trace_ID`. If `path` is a dictionary, prepend the dictionary key to
    `Trace_ID`.
    
    For other FOF_CT files, call :func:`read_data` to read in the data
    field as a dataframe.

    Parameters
    ----------
    path : str | list | dict
        The path of the FOF_CT file. Can be a list or a dictionary
        specifying multiple files.

    Raises
    ------
    ValueError
        If `path` is not a string, a list, or a dictionary.
    """
    def __init__(self, path:str|list|dict):
        if isinstance(path, str):
            self._info = self._read_info(path)
        elif isinstance(path, list):
            self._info = [self._read_info(p) for p in path]
        elif isinstance(path, dict):
            self._info = {k:self._read_info(v) for k, v in path.items()}
        else:
            raise ValueError("path must be a string, a list, or a dictionary.")
        
        self._path = path
        
    @property
    def info(self) -> str|list|dict:
        """The information in the header of the FOF-CT files.

        Returns
        -------
        list|dict
            If `path` is a single file name, returns a dictionary of the 
            header information. If `path` is a list, returns a list of 
            informationdictionaries. If `path` is a dictionary, returns 
            a dictionary of information dictionaries.
        """
        return self._info

    @staticmethod
    def _read_info(path):
        """Read in the file header."""
        with open(path, "r") as f:
            info_lines, info_dict = [], {}
            while len(info_lines) == 0 or "#" in info_lines[-1]:
                info_lines.append(f.readline().strip())
            for line in info_lines[:-1]:
                # Cannot simply set start = 0 as some rows start with "#
                start = line.find("#")
                if "##" in line:
                    sep = line.find("=")
                    info_dict[line[start+2:sep]] = line[sep+1:].strip(",")
                else:
                    sep = line.find(": ")
                    info_dict[line[start+1:sep]] = line[sep+2:].strip(",")
        
        # Parse the column names as a list
        cols = info_dict["columns"].strip("()").split(",")
        info_dict["columns"] = [t.strip() for t in cols]
        return info_dict
    
    def read_data(self) -> pd.DataFrame|list|dict:
        """Read the data field of the file(s).

        Returns
        -------
        pd.DataFrame|list|dict
            A single dataframe, a list of dataframes, or a dictionary of
            dataframes, depending on the type of `path`.
        """
        if isinstance(self._path, str):
            return pd.read_csv(
                self._path,
                skiprows=len(self.info),
                header=None,
                names=self.info["columns"]
            )
        elif isinstance(self._path, list):
            dfs = []
            for i, p in enumerate(self._path):
                df = pd.read_csv(
                    p,
                    skiprows=len(self.info[i]),
                    header=None,
                    names=self.info[i]["columns"]
                )
                dfs.append(df)
            return dfs
        elif isinstance(self._path, dict):
            dfs = {}
            for k, v in self._path.items():
                dfs[k] = pd.read_csv(
                    v,
                    skiprows=len(self.info[k]),
                    header=None,
                    names=self.info[k]["columns"]
                )
            return dfs
    
    @staticmethod
    def _fof_ct_core(path:str, info:dict, chr_id:str) -> pd.DataFrame:
        """Read FOF_CT-core. Remove duplicates. Specify data types.
        """
        if "FOF-CT_core" not in info["Table_namespace"]:
            raise ValueError("Not FOF-CT_core file.")
        
        rows = []
        chr_idx = info["columns"].index("Chrom")
        with open(path, "r") as f:
            reader = csv.reader(f)
            for _ in range(len(info)):
                next(reader)
            for row in reader:
                if str(row[chr_idx]) == chr_id:
                    rows.append(row)
        df = pd.DataFrame(rows, columns=info["columns"])
        # csv reads NaN as ""
        df = df.replace("", np.nan)
        
        # Drop rows with identical trace ID and locus ID
        df = df.drop_duplicates(["Trace_ID", "Chrom_Start"])
        
        df = df.astype({
            "Spot_ID": "str", "Chrom": "str", "Trace_ID": "str",
            "X": "float64", "Y": "float64", "Z": "float64",
            "Chrom_Start": "int64", "Chrom_End": "int64"
        }, errors="ignore")
        return df
        
    def create_adata(
        self, 
        chr_id:str,
        voxel_ratio:dict|None=None,
        var_cols_add:list|None=None,
        obs_cols_add:list|None=None,
        **kwargs
    ) -> AnnData:
        """Create an :class:`anndata:anndata.AnnData` object from a 
        single or multiple FOF_CT-core files. The output is for a single
        chromosome instead of all chromosomes imaged.

        Parameters
        ----------
        chr_id : str
            Chromosome ID used to filter `Chrom` column.
        voxel_ratio : dict | None, optional
            The conversion factor to convert voxel coordinate to actual
            physical coordinate. No conversion if None, by default None.
        var_cols_add : list | None, optional
            Additional columns added to `var` field in the AnnData 
            object, by default None.
        obs_cols_add : list | None, optional
            Additional columns added to `obs` field in the AnnData
            object, by default None.
        kwargs : dict
            Additional keyword arguments passed to 
            :class:`anndata:anndata.AnnData` constructor.

        Returns
        -------
        AnnData
            A n by p AnnData object, where n is the number of traces,
            and p is the number of locus on the chromosome. `var` field
            includes `Chrom_Start`, `Chrom_End`, and any additional
            columns specified by `var_cols_add`. `obs` field includes
            columns specified by `obs_cols_add`. The converted 3D
            coordinates are stored in layers with names `X`, `Y`, and 
            `Z`.
        """
        if isinstance(self._path, str):
            data = self._fof_ct_core(self._path, self._info, chr_id)
        elif isinstance(self._path, list):
            data = []
            for i, p in enumerate(self._path):
                df = self._fof_ct_core(p, self._info[i], chr_id)
                df["Trace_ID"] = f"{i}_" + df["Trace_ID"]
                data.append(df)
            data = pd.concat(data, ignore_index=True)
        elif isinstance(self._path, dict):
            data = []
            for k, v in self._path.items():
                df = self._fof_ct_core(v, self._info[k], chr_id)
                df["Trace_ID"] = f"{k}_" + df["Trace_ID"]
                data.append(df)
            data = pd.concat(data, ignore_index=True)
        
        # Convert voxel coordinates to nm
        if voxel_ratio is not None:
            for k, v in voxel_ratio.items():
                data[k] = data[k] * v
                
        var_cols = ["Chrom_Start", "Chrom_End"]
        if var_cols_add is not None:
            var_cols += var_cols_add
        obs_cols = []
        if obs_cols_add is not None:
            obs_cols += obs_cols_add

        locus_map = {
            t:f"loc{i}" for i, t in 
            enumerate(np.unique(data["Chrom_Start"]))
        }
        data["locus"] = data["Chrom_Start"].map(locus_map)

        var_map = (
            data[["locus"]+var_cols]
            .drop_duplicates().set_index("locus")
        )
        obs_map = (
            data[["Trace_ID"]+obs_cols]
            .drop_duplicates().set_index("Trace_ID")
        )
        
        val_cols = ["X", "Y", "Z"]
        pivoted = data.pivot_table(
            index="locus",
            columns="Trace_ID",
            values=val_cols,
            sort=False
        ).loc[locus_map.values()]  # sort by 1D location
        adata = AnnData(
            var=var_map.loc[pivoted.index],
            obs=obs_map.loc[pivoted["X"].columns],
            layers={v: pivoted[v].T for v in val_cols},
            uns={"Chrom":chr_id},
            **kwargs
        )
        return adata
    
    
def add_cell_type(
    adata:AnnData, 
    df:pd.DataFrame|list|dict,
    shared_col:str,
    cell_col:str
):
    """Add cell type information to `adata`.
    
    `adata` is created from a single or a list/dictionary of FOF_CT-core
    files. Depending on how `adata` is created, `df` is a single 
    dataframe or a list/dictionary which contains cell type information.

    Parameters
    ----------
    adata : AnnData
        Object created by :func:`FOF_CT_Loader.create_adata`.
    df : pd.DataFrame | list | dict
        Object created by :func:`FOF_CT_Loader.read_data`.
    shared_col : str
        Column shared between `adata` and `df` used as an identifier.
    cell_col : str
        Column in `df` containing cell type information.
        
    Examples
    --------
    
    1. `df` is a dataframe. `Cell_ID` is a common column in both the 
    original FOF_CT-core file and the cell type information file. Map
    `cluster label` from the cell type information file to `adata`.
    
    >>> loader = sf.pp.FOF_CT_Loader("PATH.csv")
    >>> adata = loader.create_adata("chr3", obs_cols_add=["Cell_ID"])
    >>> dfs = sf.pp.FOF_CT_Loader("TYPEPATH.csv").read_data()
    >>> sf.pp.add_cell_type(adata, dfs, "Cell_ID", "cluster label")
    
    2. `df` is a list. `Cell_ID` is a common column in both the 
    original FOF_CT-core file and the cell type information file. Map
    `cluster label` from the cell type information file to `adata`.
    
    >>> loader = sf.pp.FOF_CT_Loader(["PATH1.csv", "PATH2.csv"])
    >>> adata = loader.create_adata("chr3", obs_cols_add=["Cell_ID"])
    >>> dfs = sf.pp.FOF_CT_Loader([
    ...     "TYPEPATH1.csv", "TYPEPATH2.csv"
    ... ]).read_data()
    >>> sf.pp.add_cell_type(adata, dfs, "Cell_ID", "cluster label")
    
    3. `df` is a dictionary. `Cell_ID` is a common column in both the 
    original FOF_CT-core file and the cell type information file. Map
    `cluster label` from the cell type information file to `adata`.
    
    >>> loader = sf.pp.FOF_CT_Loader({
    ...     "rep1":"PATH1.csv", "rep2":"PATH2.csv"
    ... })
    >>> adata = loader.create_adata("chr3", obs_cols_add=["Cell_ID"])
    >>> dfs = sf.pp.FOF_CT_Loader({
    ...     "rep1":"TYPEPATH1.csv", "rep2":"TYPEPATH2.csv"
    ... }).read_data()
    >>> sf.pp.add_cell_type(adata, dfs, "Cell_ID", "cluster label")
    """
    type_df = adata.obs[shared_col].to_frame().reset_index()
    if isinstance(df, pd.DataFrame):
        type_map = df.set_index(shared_col)[cell_col]
        adata.obs[cell_col] = (
            type_df[shared_col].map(type_map)
            .values.astype(df[cell_col].dtype)
        )
    elif isinstance(df, list):
        type_df["fidx"] = (
            adata.obs[shared_col].index
            .str.replace(r"^([^_]+)_.*$", r"\g<1>", regex=True)
        )
        for i, v in enumerate(df):
            type_map = v.set_index(shared_col)[cell_col]
            idx = type_df["fidx"]==f"{i}"
            type_df.loc[idx,cell_col] = type_df[idx][shared_col].map(type_map)
        adata.obs[cell_col] = (
            type_df[cell_col].values
            .astype(v[cell_col].dtype)
        )
    elif isinstance(df, dict):
        type_df["fidx"] = (
            adata.obs[shared_col].index
            .str.replace(r"^([^_]+)_.*$", r"\g<1>", regex=True)
        )
        for k, v in df.items():
            type_map = v.set_index(shared_col)[cell_col]
            idx = type_df["fidx"]==k
            type_df.loc[idx,cell_col] = type_df[idx][shared_col].map(type_map)
        adata.obs[cell_col] = (
            type_df[cell_col].values
            .astype(v[cell_col].dtype)
        )
        

def to_very_wide(
    adata:AnnData
) -> Tuple[pd.DataFrame, np.ndarray]:
    pass


class ChromArray:
    pass