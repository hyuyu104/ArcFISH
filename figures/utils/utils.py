from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import snapfish2 as sf


def to_loop_roc_df(r1, r2, true_path):
    # exclude pairs not included in both tests
    ff = ~(r1.isna()["stat"]&r2.isna()["stat"])
    rr = r1[ff].copy()[["c1", "s1", "e1", "c2", "s2", "e2"]]
    rr["thresh1"] = 1 - r1[ff]["fdr"]
    rr["thresh2"] = 1 - r2[ff]["fdr"]
    # fill failed-to-test entries with 0
    rr = rr.fillna(0)

    true_df = pd.read_csv(true_path, sep="\t")
    # 0 -> not a loop; 1 -> loop
    rr["loop"] = true_df[ff]["loop"].astype("int")
    # print(f"Total # of loops: {np.sum(rr["loop"])}")
    return rr


def tad_enrichment_row(tad_df, chip_df, col="default"):
    tad_df_overlapped = []
    for chr_id in pd.unique(tad_df["c1"]):
        df = tad_df[tad_df["c1"]==chr_id].copy()
        sub_chip_df = chip_df[chip_df["c1"]==chr_id]
        df[col] = df["1D"].isin(sub_chip_df["s1"])
        tad_df_overlapped.append(df)
    output = pd.concat(tad_df_overlapped, ignore_index=True)
    
    if col == "default":
        return {
            "peak_shared": np.sum(output[col]&output["peak"]),
            "num_peaks": np.sum(output["peak"]),
            "non_peak_shared": np.sum(output[col]&~output["peak"]),
            "num_non_peaks": np.sum(~output["peak"])
        }
    return output


def create_roc_df(
    true_path:Path, r1:pd.DataFrame, r2:pd.DataFrame    
) -> pd.DataFrame:
    ff = ~(r1.isna()["stat"]&r2.isna()["stat"])
    cols = ["c1", "s1", "e1", "c2", "s2", "e2"]
    rr = r1[ff].copy()[cols]
    rr["thresh1"] = 1 - r1[ff]["fdr"]
    rr["thresh2"] = 1 - r2[ff]["fdr"]

    true_df = pd.read_csv(true_path, sep="\t", header=None)
    true_df.columns = cols + ["count"]
    rr = pd.merge(rr, true_df, on=cols, how="left")
    rr["loop"] = ~np.isnan(rr["count"])

    df1 = pd.DataFrame(roc_curve(rr["loop"], rr["thresh1"])[:-1]).T
    df1["Method"] = "SnapFISH"
    df2 = pd.DataFrame(roc_curve(rr["loop"], rr["thresh2"])[:-1]).T
    df2["Method"] = "SnapFISH2"
    df = pd.concat([df1, df2], ignore_index=True)
    df.columns = ["FPR", "TPR", "Method"]
    return df


def chiapet_sub_df(
    chr_id:str, d1df:pd.DataFrame, 
    chiapet_path:Path, 
    cut_lo:float, cut_hi:float
) -> pd.DataFrame:
    df = pd.read_csv(chiapet_path, sep="\t", header=None)
    df.columns = ["c1", "s1", "e1", "c2", "s2", "e2", "count"]

    ranges = []
    for i, (_, row) in enumerate(d1df.iterrows()):
        if len(ranges) == 0:
            ranges.append([row["Chrom_Start"], row["Chrom_End"]])
        else:
            if row["Chrom_Start"] == ranges[-1][1]:
                ranges[-1][1] = row["Chrom_End"]
            else:
                ranges.append([row["Chrom_Start"], row["Chrom_End"]])
    ranges = np.array(ranges)
    
    sub_df = df[(df["c1"]==chr_id)&(df["c2"]==chr_id)]
    f1 = sf.tl.overlap(sub_df[["s1", "e1"]].values, ranges)
    sub_df = sub_df[f1]
    f2 = sf.tl.overlap(sub_df[["s2", "e2"]].values, ranges)
    sub_df["x"] = (sub_df["s1"] + sub_df["e1"])/2
    sub_df["y"] = (sub_df["s2"] + sub_df["e2"])/2
    sub_df = sub_df[f2]
    loop_len = sub_df["s2"] - sub_df["s1"]
    sub_df = sub_df[(loop_len>=cut_lo)&(loop_len<=cut_hi)]
    
    return sub_df