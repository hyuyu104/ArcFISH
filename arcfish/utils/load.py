import csv
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from anndata import AnnData


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
    field as a dataframe. The only required argument is `path`, all 
    other arguments passed to the constructor will be ignored.

    Parameters
    ----------
    path : str | list | dict
        The path of the FOF_CT file. Can be a list or a dictionary
        specifying multiple files.
    nm_ratio : dict | None, optional
        The conversion factor to convert raw coordinate to physical
        coordinate in nm. No conversion if None, by default None.
    var_cols_add : list | None, optional
        Additional columns added to `var` field in the AnnData 
        object, by default None.
    obs_cols_add : list | None, optional
        Additional columns added to `obs` field in the AnnData
        object, by default None.
    kwargs : dict
        Additional keyword arguments passed to 
        :class:`anndata:anndata.AnnData` constructor.

    Raises
    ------
    ValueError
        If `path` is not a string, a list, or a dictionary.
    """
    def __init__(self, 
        path:str|list|dict,
        nm_ratio:dict|None=None,
        var_cols_add:list|None=None,
        obs_cols_add:list|None=None,
        **kwargs             
    ):
        if isinstance(path, str):
            self._info = self._read_info(path)
        elif isinstance(path, list):
            self._info = [self._read_info(p) for p in path]
        elif isinstance(path, dict):
            self._info = {k:self._read_info(v) for k, v in path.items()}
        else:
            raise ValueError("path must be a string, a list, or a dictionary.")
        
        self._path = path
        
        self._nm_ratio = nm_ratio
        self._var_cols_add = var_cols_add
        self._obs_cols_add = obs_cols_add
        self._kwargs = kwargs
        
    @property
    def path(self) -> str|list|dict:
        """str|list|dict : Path of input files.
        """
        return self._path
    
    @property
    def nm_ratio(self) -> dict|None:
        """dict|None : Conversion factor to convert voxel coordinate to
        actual physical coordinate.
        """
        return self._nm_ratio
        
    @property
    def info(self) -> list|dict:
        """list|dict : If `path` is a single file name, returns a 
        dictionary of the header information. If `path` is a list, 
        returns a list of information dictionaries. If `path` is a 
        dictionary, returns a dictionary of information dictionaries.
        """
        return self._info
    
    @property
    def chr_ids(self) -> np.ndarray:
        """np.ndarray : Available chromosomes in FOF_CT-core."""
        if isinstance(self._path, str):
            paths = [self._path]
            infos = [self._info]
        elif isinstance(self._path, list):
            paths = self._path
            infos = self._info
        elif isinstance(self._path, dict):
            paths = list(self._path.values())
            infos = [self._info[k] for k in self._path]
        
        chrom_df = []
        for path, info in zip(paths, infos):
            df = pd.read_csv(
                path, 
                skiprows=len(info),
                header=None,
                usecols=[info["columns"].index("Chrom")],
                names=["Chrom"]
            )
            chrom_df.append(df)
        chrom_df = pd.concat(chrom_df).drop_duplicates()
        return chrom_df["Chrom"].values

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
            # Columns in the header
            if "columns" in info_dict:
                # Parse the column names as a list
                cols = info_dict["columns"].strip("()").split(",")
            # Column names in data
            else:
                cols = info_lines[-1].split(",")
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
        df = pd.DataFrame(rows)
        if df.shape[1] == len(info["columns"]):
            df.columns = info["columns"]
        elif df.shape[1] < len(info["columns"]):
            # Some columns are missing
            df.columns = info["columns"][:df.shape[1]]
            warnings.warn(
                "FOF_CT-core data has less columns than what specified " + \
                "in the header. Additional columns will be ignored."
            )
        else:
            raise ValueError(
                "FOF_CT-core data has more columns than what specified " + \
                "in the header. Please check the original FOF_CT-core file."
            )
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
        
    def create_adata(self, chr_id:str) -> AnnData:
        """Create an :class:`anndata:anndata.AnnData` object from a 
        single or multiple FOF_CT-core files. The output is for a single
        chromosome instead of all chromosomes imaged.

        Parameters
        ----------
        chr_id : str
            Chromosome ID used to filter `Chrom` column.

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
        if self._nm_ratio is not None:
            for k, v in self._nm_ratio.items():
                data[k] = data[k] * v
                
        var_cols = ["Chrom_Start", "Chrom_End"]
        if self._var_cols_add is not None:
            var_cols += self._var_cols_add
        obs_cols = []
        if self._obs_cols_add is not None:
            obs_cols += self._obs_cols_add

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
            **self._kwargs
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


def save_fof_ct_core(df:pd.DataFrame, info:dict, path:Path):
    """Save a dataframe as FOF_CT-core file.

    Parameters
    ----------
    df : pd.DataFrame
        Chromatin 3D coordinates.
    info : dict
        A dictionary containing FOF_CT-core header information.
    path : Path
        Output file path.
    """
    eq_keys = [
        "FOF-CT_version", "Table_namespace", 
        "genome_assembly", "XYZ_unit"
    ]
    df.to_csv(path, index=False, header=False)
    header = ""
    for k, v in info.items():
        if k == "columns":
            s = f'##{k}=({",".join(v)})\n'
        elif k in eq_keys:
            s = f"##{k}={v}\n"
        else:
            s = f"#{k}: {v}\n"
        header += s
    with open(path, "r") as f:
        header += f.read()
    with open(path, "w") as f:
        f.write(header)