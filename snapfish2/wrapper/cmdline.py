import os
import argparse
import pandas as pd
import anndata as ad

from .. import utils as pp
from .. import tools as tl


def _parse_file_paths(file_paths):
    if "::" in file_paths:
        return {
            f.split("::")[0].strip(): f.split("::")[1]
            for f in file_paths.split(",")
        }
    elif "," in file_paths:
        return file_paths.split(",")
    return file_paths


def _parse_vocel_ratio(voxel_ratio):
    return {
        v.split("::")[0].strip(): float(v.split("::")[1])
        for v in voxel_ratio.split(",")
    }


def preprocess(args):
    input = _parse_file_paths(args.input)
    output = args.output
    voxel_ratio = args.voxel_ratio
    if voxel_ratio is not None:
        voxel_ratio = _parse_vocel_ratio(voxel_ratio)
    loader = pp.FOF_CT_Loader(
        path=input, 
        voxel_ratio=voxel_ratio
    )
    if not os.path.exists(output):
        os.mkdir(output)
    for chr_id in loader.chr_ids:
        adata = loader.create_adata(chr_id=chr_id)
        pp.filter_normalize(adata=adata)
        adata.write_h5ad(
            os.path.join(output, f"AD{chr_id}.h5ad"),
            compression="gzip"
        )


def call_loops(args):
    argdict = vars(args)
    input = argdict.pop("input")
    output = argdict.pop("output")
    argdict.pop("func")
    argdict.pop("command")
    result = []
    for adata_file in os.listdir(input):
        if not adata_file.startswith("AD"):
            continue
        adata = ad.read_h5ad(os.path.join(input, adata_file))
        loops = tl.LoopCaller(**argdict)
        res = loops.to_bedpe(loops.call_loops(adata), adata)
        result.append(res)
    
    pd.concat(result, ignore_index=True).to_csv(
        output, sep="\t", index=False
    )
    
    
def call_domains(args):
    argdict = vars(args)
    input = argdict.pop("input")
    output = argdict.pop("output")
    argdict.pop("func")
    argdict.pop("command")
    result = []
    for adata_file in os.listdir(input):
        if not adata_file.startswith("AD"):
            continue
        adata = ad.read_h5ad(os.path.join(input, adata_file))
        domains = tl.TADCaller(**argdict)
        res = domains.to_bedpe(domains.call_tads(adata))
        result.append(res)
    
    pd.concat(result, ignore_index=True).to_csv(
        output, sep="\t", index=False
    )
    
    
def call_cpmts(args):
    argdict = vars(args)
    input = argdict.pop("input")
    output = argdict.pop("output")
    argdict.pop("func")
    argdict.pop("command")
    result = []
    for adata_file in os.listdir(input):
        if not adata_file.startswith("AD"):
            continue
        adata = ad.read_h5ad(os.path.join(input, adata_file))
        cpmts = tl.ABCaller(**argdict)
        res = cpmts.call_cpmt(adata)
        result.append(res)
    
    pd.concat(result, ignore_index=True).to_csv(
        output, sep="\t", index=False
    )


def main():
    parser = argparse.ArgumentParser(description="CLI for SnapFISH2.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parser_pp = subparsers.add_parser("preprocess", help="Preprocess data.")
    parser_pp.add_argument("-i", "--input", type=str, required=True)
    parser_pp.add_argument("-o", "--output", type=str, required=True)
    parser_pp.add_argument("-v", "--voxel_ratio", type=str, default=None)
    parser_pp.set_defaults(func=preprocess)
    
    parser_lp = subparsers.add_parser("loop", help="Call loops.")
    parser_lp.add_argument("-i", "--input", type=str, required=True)
    parser_lp.add_argument("-o", "--output", type=str, required=True)
    parser_lp.add_argument("-fdr", "--fdr_cutoff", type=float, default=0.1)
    parser_lp.add_argument("-pval", "--pval_cutoff", type=float, default=1e-5)
    parser_lp.add_argument("-lo", "--cut_lo", type=float, default=1e5)
    parser_lp.add_argument("-up", "--cut_up", type=float, default=1e6)
    parser_lp.add_argument("-gap", "--gap", type=float, default=50e3)
    parser_lp.add_argument("-outer", "--outer_cut", type=float, default=50e3)
    parser_lp.set_defaults(func=call_loops)
    
    parser_do = subparsers.add_parser("domain", help="Call domains.")
    parser_do.add_argument("-i", "--input", type=str, required=True)
    parser_do.add_argument("-o", "--output", type=str, required=True)
    parser_do.add_argument("-fdr", "--fdr_cutoff", type=float, default=0.1)
    parser_do.add_argument("-window", "--window", type=float, default=1e5)
    parser_do.add_argument("-tree", "--tree", type=bool, default=True)
    parser_do.add_argument("-min", "--min_tad_size", type=float, default=0)
    parser_do.set_defaults(func=call_domains)
    
    parser_cp = subparsers.add_parser("cpmt", help="Call compartments.")
    parser_cp.add_argument("-i", "--input", type=str, required=True)
    parser_cp.add_argument("-o", "--output", type=str, required=True)
    parser_cp.add_argument("-min", "--min_cpmt_size", type=float, default=0)
    parser_cp.set_defaults(func=call_cpmts)
    
    args = parser.parse_args()
    args.func(args)