"""
For 4DN data download, first create a keypair 
(see https://data.4dnucleome.org/help/user-guide/downloading-files).

The implementation assumes the keypairs is stored in the home directory,
that is, `~/keypairs.json` stores the key password information.

If the `keypairs.json` is stored elsewhere, remember to change the value 
of `HOMEDIRE` to the directory containing `keypairs.json`.
"""
import os
import argparse
import json
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from scipy import stats

import snapfish2 as sf
from snapfish2.tools.func import (
    overlap,
    loop_overlap, 
    all_possible_pairs,
)


HOMEDIRE = os.path.expanduser("~")
STRENCODE = "https://www.encodeproject.org/files/{}/@@download/{}"
STR4DN = "https://data.4dnucleome.org/files-processed/{}/@@download/{}"


class DataTree:
    """Read the `data.json` file stored in the input directory."""
    def __init__(self, data_dire):
        self._data_dire = data_dire
        with open(os.path.join(data_dire, "data.json"), "r") as f:
            self._tree = json.load(f)
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            res = self._tree[idx]
        else:
            res = self._tree
            for t in idx:
                res = res[t]
        # Not the bottom level -> peeking
        if isinstance(res, dict):
            return res
        # Bottom level -> return the path
        return os.path.join(self._data_dire, idx[0], res)
    
    
def _dict_recursion(d):
    vs = []
    for k, v in d.items():
        if isinstance(v, dict):
            vs.extend(_dict_recursion(v))
        else:
            vs.append(v)
    return vs


def download_folder(key:str, val:dict, data_dire:str="../data"):
    """Download all files contained in `val` and stored them in the 
    directory `{data_dire}/{key}`. Will create the output directory if
    it is unavailable.
    """
    print("Dire used is", HOMEDIRE)
    out_dire = os.path.join(data_dire, key)
    if not os.path.exists(out_dire):
        os.mkdir(out_dire)
    vs = _dict_recursion(val)
    for v in vs:
        out_name = os.path.join(out_dire, v)
        if os.path.exists(out_name):
            continue
        
        if v.startswith("ENC"):
            url = STRENCODE.format(v.split(".")[0], v)
            response = requests.get(url, stream=True)
        if v.startswith("4DN"):
            with open(os.path.join(HOMEDIRE, "keypairs.json"), "r") as f:
                keypair = json.load(f)
            keyv = keypair["default"]["key"]
            pswd = keypair["default"]["secret"]
            
            url = STR4DN.format(v.split(".")[0], v)
            response = requests.get(url, auth=(keyv, pswd), stream=True)
            
        with open(out_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                

def download_folder_zenodo(key:str, val:dict, data_dire:str="../data"):
    """`val` contains the information of a single zenodo repository, 
    with keys `prefix` and `files`. Download all files in `files` to
    directory `{data_dire}/{key}`.
    """
    out_dire = os.path.join(data_dire, key)
    if not os.path.exists(out_dire):
        os.mkdir(out_dire)
    for v in val["files"].values():
        out_name = os.path.join(out_dire, v)
        if os.path.exists(out_name):
            continue
        response = requests.get(f"{val["prefix"]}{v}/content", stream=True)
        
        with open(out_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                

def download_all(data_dire:str="../data"):
    """Download all files specified in `{data_dire}/data.json`"""
    with open(os.path.join(data_dire, "data.json"), "r") as f:
        data_tree = json.load(f)
    for key, val in data_tree.items():
        download_folder(key, val, data_dire)
    with open(os.path.join(data_dire, "zenodo.json"), "r") as f:
        zenodo_tree = json.load(f)
    for key, val in zenodo_tree.items():
        download_folder_zenodo(key, val, data_dire)

########################################################################
##                        Utility functions                           ##
########################################################################

def filter_save(
    df:pd.DataFrame, d1df:pd.DataFrame, out:str|None=None
) -> None | pd.DataFrame:
    """Convert the signals in `df` to genomic loci in `d1df`. `df` must 
    contain columns c1, s1, e1 if it is a 1D signal (e.g. ChIP-seq), and
    it must contain columns c1, s1, e1, c2, s2, e2 if it is a 2D signal 
    (e.g. ChIA-PET).
    """
    if "c2" not in df:  # 1D signal
        out_ls = []
        for chr_id in pd.unique(d1df.Chrom):
            sub_df = d1df[d1df["Chrom"]==chr_id].copy()
            ints1 = sub_df.iloc[:,1:].values
            ints2 = df[df["c1"]==chr_id][["s1","e1"]].values
            out_ls.append(sub_df[overlap(ints1, ints2)])
        out_df = pd.concat(out_ls, ignore_index=True).rename({
            "Chrom":"c1", "Chrom_Start":"s1", "Chrom_End":"e1"
        }, axis=1)
    else:  # pair-wise signal
        pairs = all_possible_pairs(d1df)
        out_df = loop_overlap(pairs, df)
    if out is None:
        return out_df
    out_df.to_csv(out, sep="\t", index=False)
    

def add_noise(
    input_paths:str|list|dict, voxel_ratio:dict, 
    out_paths:str|list|dict, noises:dict
):
    """Add noises and save the noised data in the original scale (need
    to be converted by voxel ratio when re-read).
    """
    if isinstance(input_paths, str):
        input_paths = {0:input_paths}
        out_paths = {0:out_paths}
    if isinstance(input_paths, list):
        input_paths = dict(enumerate(input_paths))
        out_paths = dict(enumerate(out_paths))
    
    for k, v in input_paths.items():
        loader = sf.pp.FOF_CT_Loader(v)
        data = loader.read_data()
        for axis in voxel_ratio:
            data[axis] *= voxel_ratio[axis]
            data[axis] += stats.norm.rvs(size=(len(data)), scale=noises[axis])
            data[axis] /= voxel_ratio[axis]
        sf.pp.save_fof_ct_core(data, loader.info, out_paths[k])
    
########################################################################
##                 Actual processing of all files                     ##
########################################################################

def proc_chipseq_mesc(data_dire:str="../data"):
    dtree = DataTree(data_dire)
    
    fish_path = dtree["takei_nature_2021","25Kb","rep1"]
    d1df = sf.MulFish(fish_path).data[[
        "Chrom", "Chrom_Start", "Chrom_End"
    ]].drop_duplicates()
    
    chip_dire = "chipseq_mesc"
    for k in dtree[chip_dire]:
        df = pd.read_csv(dtree[chip_dire,k], sep="\t",
                         header=None, usecols=[0, 1, 2])
        df.columns = ["c1", "s1", "e1"]
        out = os.path.join(data_dire, chip_dire, f"{k}_ChIPseq_mESC.csv")
        filter_save(df, d1df, out)
        

def filter_chiapet(
    fish_path:Path, 
    chiapet_path:Path, 
    out_path:Path,
    min_count:int=0,
    cut_lo:float=1e5,
    cut_hi:float=1e6
):
    loader = sf.pp.FOF_CT_Loader(fish_path)
    d1cols = ["Chrom", "Chrom_Start", "Chrom_End"]
    d1df = loader.read_data()[d1cols].drop_duplicates().groupby(
        "Chrom", sort=False
    )[d1cols].apply(lambda df: df.sort_values(
        "Chrom_Start", ignore_index=True
    )).reset_index(drop=True)
    df = pd.read_csv(chiapet_path, sep="\t", usecols=np.arange(7))
    df.columns = ["c1", "s1", "e1", "c2", "s2", "e2", "count"]
    loop_len = df["s2"] - df["s1"]
    df = df[(loop_len>=cut_lo)&(loop_len<=cut_hi)]
    df = df[df["count"]>=min_count]
    chiapet_df = filter_save(df, d1df)
    chiapet_df[chiapet_df["overlapped"]==3].to_csv(
        out_path, sep="\t", index=False, header=False
    )
        

def proc_chiapet_mesc(data_dire:str="../data"):
    dtree = DataTree(data_dire)
    
    fish_path = dtree["takei_nature_2021","25Kb","rep1"]
    d1df = sf.MulFish(fish_path).data[[
        "Chrom", "Chrom_Start", "Chrom_End"
    ]].drop_duplicates()
    
    chiapet_dire = "chiapet_mesc"
    for target, val in dtree[chiapet_dire].items():
        out_dfs = []
        for strain in val:
            df = pd.read_csv(
                dtree[chiapet_dire,target,strain], sep="\t", 
                header=None, usecols=np.arange(6)
            )
            df.columns = ["c1", "s1", "e1", "c2", "s2", "e2"]
            out_dfs.append(filter_save(df, d1df))
        
        # Intersection of all strains
        inte_df = out_dfs[0].copy()
        inte_df["loop"] = inte_df["overlapped"]==3
        for out_df in out_dfs[1:]:
            inte_df["loop"] = (inte_df["loop"])&(out_df["overlapped"]==3)
        inte_df.drop("overlapped", axis=1).to_csv(os.path.join(
            data_dire, chiapet_dire, f"{target}_intersection.csv"
        ), sep="\t", index=False)
        
         # Union of all strains
        inte_df = out_dfs[0].copy()
        inte_df["loop"] = inte_df["overlapped"]==3
        for out_df in out_dfs[1:]:
            inte_df["loop"] = (inte_df["loop"])|(out_df["overlapped"]==3)
        inte_df.drop("overlapped", axis=1).to_csv(os.path.join(
            data_dire, chiapet_dire, f"{target}_union.csv"
        ), sep="\t", index=False)
        
        
def proc_fithic_mesc(data_dire:str="../data"):
    hic_dire = os.path.join(data_dire, "bonev_cell_2017")
    rep_dire = os.path.join(hic_dire, "FitHiC_output_4DNFI4OUMWZ8")
    hic_fname = "FitHiC.spline_pass1.significances.rawcount.5.qval.0.01.txt"
    hic_df = []
    for c in list(range(1, 20)) + ["X"]:
        fname = os.path.join(rep_dire, f"chr{c}", hic_fname)
        hic_df.append(pd.read_csv(fname, sep="\t", header=None))
    hic_df = pd.concat(hic_df, ignore_index=True)
    hic_df.columns = [
        "c1", "m1", "c2", "m2", "contactCount", 
        "p-value", "q-value", "bias1", "bias2", "ExpCC"
    ]

    # Ensure the first bin precedes the second bin
    rf = hic_df["m1"] > hic_df["m2"]
    m1 = hic_df.loc[rf, "m1"]
    hic_df.loc[rf, "m1"] = hic_df.loc[rf, "m2"]
    hic_df.loc[rf, "m2"] = m1

    # Convert chromosome ID to "chrN"
    hic_df["c1"] = "chr" + hic_df["c1"].astype(str)
    hic_df["c2"] = "chr" + hic_df["c2"].astype(str)

    # Add bin starting and ending positions
    hic_df["s1"] = (hic_df["m1"] - 5e3).astype("int64")
    hic_df["e1"] = (hic_df["m1"] + 5e3).astype("int64")
    hic_df["s2"] = (hic_df["m2"] - 5e3).astype("int64")
    hic_df["e2"] = (hic_df["m2"] + 5e3).astype("int64")

    hic_df = hic_df[[
        "c1", "s1", "e1", "c2", "s2", "e2", "contactCount",
        "p-value", "q-value", "bias1", "bias2", "ExpCC"
    ]]

    assert len(hic_df[hic_df["s1"] > hic_df["s2"]]) == 0, \
        "First bin should precede the second bin"
        
    hic_df.to_csv(
        os.path.join(hic_dire, "peaks_G0G1mESCs.bedpe"),
        sep="\t", index=False, header=False
    )
        

def proc_su_cell_2020(data_dire:str="../data"):
    sub_dire = os.path.join(data_dire, "su_cell_2020")

    trace_files = [
        "chromosome2.tsv",
        "chromosome2_p-arm_replicate.tsv",
        "chromosome21-cell_cycle.tsv",
        "chromosome21.tsv"
    ]
    for f in trace_files:
        out_name = os.path.join(sub_dire, f"{f[:-3]}csv")
        if os.path.exists(out_name):
            continue
        df = pd.read_csv(
            os.path.join(sub_dire, f), sep="\t"
        ).rename({
            "X(nm)":"X", "Y(nm)":"Y", "Z(nm)":"Z",
            "Genomic coordinate": "1d", "Chromosome copy number": "Trace_ID"
        }, axis=1).reset_index(names="Spot_ID")
        split1d = df["1d"].str.split(":", expand=True)
        df["Chrom"] = split1d[0]
        se_cols = split1d[1].str.split(
            "-", expand=True
        ).astype("int").rename(
            {0:"Chrom_Start", 1:"Chrom_End"}, axis=1
        )
        cols = ["Spot_ID", "Trace_ID", "X", "Y", "Z", 
                "Chrom", "Chrom_Start", "Chrom_End"]
        df = pd.concat([df, se_cols], axis=1)[cols]
        df = df[(df["X"]!="")&(df["Y"]!="")&(df["Z"]!="")]
        df.to_csv(out_name, index=False, header=False)
        
        with open(out_name, "r") as original:
            data = original.read()
        with open(out_name, "w") as modified:
            modified.write("##Table_namespace=4dn_FOF-CT_core\n" \
                + f"##columns=({",".join(cols)})\n" + data)
        
        
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir', required=False, default="data", \
                        help='data directory')
    parser.add_argument('-k', '--keydir', required=False, 
                        default=os.path.expanduser("~"), \
                        help='directory with keypairs.json')
    return parser
        
        
if __name__ == "__main__":
    # parser = create_parser()
    # args = parser.parse_args()
    # HOMEDIRE = args.keydir
    # download_all(args.datadir)
    # proc_chipseq_mesc(args.datadir)
    # proc_chiapet_mesc(args.datadir)
    # proc_fithic_mesc(args.datadir)
    # proc_su_cell_2020(args.datadir)
    
    data_dire = "data"
    with open(os.path.join(data_dire, "data.json"), "r") as f:
        data_tree = json.load(f)
    key = "hung_ng_2024"
    download_folder(key, data_tree[key], data_dire)