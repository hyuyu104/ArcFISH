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


def _parse_vocel_ratio(nm_ratio):
    return {
        v.split("::")[0].strip(): float(v.split("::")[1])
        for v in nm_ratio.split(",")
    }


def preprocess(args):
    input = _parse_file_paths(args.input)
    output = args.output
    nm_ratio = args.nm_ratio
    if nm_ratio is not None:
        nm_ratio = _parse_vocel_ratio(nm_ratio)
    loader = pp.FOF_CT_Loader(
        path=input, 
        nm_ratio=nm_ratio
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
    
    
def diff_loops(args):
    raise NotImplementedError(
        "Differential loop calling is not implemented yet."
    )
    loader1 = pp.FOF_CT_Loader(args.input1)
    loader2 = pp.FOF_CT_Loader(args.input2)
    loops = pd.read_csv(args.loops, sep="\t")
    
    result = []
    for chr_id in loader1.chr_ids:
        adata1 = loader1.create_adata(chr_id=chr_id)
        adata2 = loader2.create_adata(chr_id=chr_id)
        diff = tl.DiffLoop(adata1, adata2)
        res = diff.to_bedpe(diff.diff_loops(loops), args.fdr)
        result.append(res)
    
    pd.concat(result, ignore_index=True).to_csv(
        args.output, sep="\t", index=False
    )
    
    
def diff_domains(args):
    raise NotImplementedError(
        "Differential domains is not implemented yet."
    )
    loader1 = pp.FOF_CT_Loader(args.input1)
    loader2 = pp.FOF_CT_Loader(args.input2)
    domains = pd.read_csv(args.domains, sep="\t")
    
    result = []
    for chr_id in loader1.chr_ids:
        adata1 = loader1.create_adata(chr_id=chr_id)
        adata2 = loader2.create_adata(chr_id=chr_id)
        diff = tl.DiffDomain(adata1, adata2)
        res = diff.to_bedpe(diff.diff_domains(domains), args.fdr)
        result.append(res)
        
    pd.concat(result, ignore_index=True).to_csv(
        args.output, sep="\t", index=False
    )


def main():
    parser = argparse.ArgumentParser(description="CLI for ArcFISH.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    parser_pp = subparsers.add_parser("preprocess", help="Preprocess data.")
    parser_pp.add_argument("-i", "--input", type=str, required=True)
    parser_pp.add_argument("-o", "--output", type=str, required=True)
    parser_pp.add_argument("-v", "--nm_ratio", type=str, default=None)
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
    
    parser_dl = subparsers.add_parser("dloop", help="Diff loops.")
    parser_dl.add_argument("-i1", "--input1", type=str, required=True)
    parser_dl.add_argument("-i2", "--input2", type=str, required=True)
    parser_dl.add_argument("-l", "--loops", type=str, required=True)
    parser_dl.add_argument("-o", "--output", type=str, required=True)
    parser_dl.add_argument("-fdr", "--fdr_cutoff", type=float, default=0.1)
    parser_dl.set_defaults(func=diff_loops)
    
    parser_dd = subparsers.add_parser("ddomain", help="Diff domains.")
    parser_dd.add_argument("-i1", "--input1", type=str, required=True)
    parser_dd.add_argument("-i2", "--input2", type=str, required=True)
    parser_dd.add_argument("-d", "--domains", type=str, required=True)
    parser_dd.add_argument("-o", "--output", type=str, required=True)
    parser_dd.add_argument("-fdr", "--fdr_cutoff", type=float, default=0.1)
    parser_dd.set_defaults(func=diff_domains)
    
    args = parser.parse_args()
    args.func(args)