"""Call loops and TADs from mouse brain data

To run this script, 
1. Change the `wkdir` variable to your working directory.
2. Change the output directory `out_dire` variable to your desired output path.
3. Run the script using:
   ```bash
   python /path/to/41_brain.py
   ```
"""

wkdir = "/proj/yunligrp/users/hongyuyu/ArcFISH"
out_dire = f"{wkdir}/output"

import os
import pandas as pd

import arcfish as af

def fetch_adata(chr_id, cell_type, loader, celldfs):
    adata = loader.create_adata(chr_id)
    af.pp.add_cell_type(adata, celldfs, "Cell_ID", "cluster label")
    celltype_dict = {1:'Pvalb', 2:'Vip', 3:'Ndnf', 4:'Sst', 5:'Astro',
                    6:'Micro', 7:'Endo', 8:'Oligo', 9:'Ex'}
    adata.obs["cell_type"] = adata.obs["cluster label"].map(celltype_dict)
    return adata[adata.obs["cell_type"] == cell_type].copy()

def main(out_dire, data_dire):
    brain_dire = f"{out_dire}/brain25kb"
    if not os.path.exists(brain_dire):
        os.makedirs(brain_dire)

    loader = af.pp.FOF_CT_Loader({
        'rep1': f'{data_dire}/takei_science_2021/4DNFIW4S8M6J.csv',
        'rep2': f'{data_dire}/takei_science_2021/4DNFI4LI6NNV.csv',
        'rep3': f'{data_dire}/takei_science_2021/4DNFIDUJQDNO.csv'
    }, voxel_ratio={"X": 1000, "Y": 1000, "Z": 1000}, obs_cols_add=["Cell_ID"])
    
    celldfs = af.pp.FOF_CT_Loader({
        'rep1': f'{data_dire}/takei_science_2021/4DNFIG1KETHF.csv',
        'rep2': f'{data_dire}/takei_science_2021/4DNFIFEOFBK4.csv',
        'rep3': f'{data_dire}/takei_science_2021/4DNFI6C764XE.csv'
    }).read_data()
    
    celltype_dict = {1:'Pvalb', 2:'Vip', 3:'Ndnf', 4:'Sst', 5:'Astro',
                     6:'Micro', 7:'Endo', 8:'Oligo', 9:'Ex'}
    
    loop = af.tl.LoopCaller()
    domain = af.tl.TADCaller(tree=False)

    loop_dict = {k: [] for v, k in celltype_dict.items()}
    domain_dict = {k: [] for v, k in celltype_dict.items()}

    for chr_id in loader.chr_ids:
        adata = loader.create_adata(chr_id)
        af.pp.add_cell_type(adata, celldfs, "Cell_ID", "cluster label")
        adata.obs["cell_type"] = adata.obs["cluster label"].map(celltype_dict)
        
        for cell_type in adata.obs["cell_type"].unique():
            sub_adata = adata[adata.obs["cell_type"] == cell_type].copy()
            af.pp.filter_normalize(sub_adata)
            af.pp.median_pdist(sub_adata, inplace=True)
            
            sub_adata.write_h5ad(
                f"{brain_dire}/brain25kb_{chr_id}_{cell_type}.h5ad", 
                compression="gzip"
            )
            
            loop_dict[cell_type].append(
                loop.to_bedpe(loop.call_loops(sub_adata), sub_adata)
            )
            
            domain_dict[cell_type].append(
                domain.call_tads(sub_adata)
            )

        del adata
        
    for cell_type in loop_dict:
        if loop_dict[cell_type]:
            loop_df = pd.concat(loop_dict[cell_type], ignore_index=True)
            loop_df.to_csv(
                f"{brain_dire}/brain25kb_loop_{cell_type}.bedpe",
                sep="\t", index=False
            )
        
        if domain_dict[cell_type]:
            domain_df = pd.concat(domain_dict[cell_type], ignore_index=True)
            domain_df.to_csv(
                f"{brain_dire}/brain25kb_domain_{cell_type}.bedpe",
                sep="\t", index=False
            )
            
    domain = af.tl.TADCaller(tree=False)
    res_dict = {t: [] for t in celltype_dict.values()}
    for cell_type in res_dict.keys():
        for chr_id in [f"chr{t}" for t in range(1, 20)]:
            adata = fetch_adata(chr_id, cell_type, loader, celldfs)
            adata.obs["rep"] = adata.obs_names.str.split("_").str[0]
            for rep in adata.obs["rep"].unique():
                sub_adata = adata[adata.obs["rep"] == rep].copy()
                res = domain.call_tads(sub_adata)
                res["cell_type"] = cell_type
                res["rep"] = rep
                res_dict[cell_type].append(res)
    res = pd.concat([pd.concat(v) for v in res_dict.values()], ignore_index=True)
    
    res.to_csv(f"{brain_dire}/brain25kb_domain_rep.csv", index=False)
        
if __name__ == "__main__":
    data_dire = os.path.join(wkdir, "data")
    main(out_dire, data_dire)