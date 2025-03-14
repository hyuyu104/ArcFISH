from pathlib import Path
import pandas as pd

from .. import utils as pp
from .. import tools as tl


def caller_wrapper(
    loader:pp.FOF_CT_Loader,
    loop:tl.LoopCaller|None=None, loop_path:Path|None=None,
    tad:tl.TADCaller|None=None, tad_path:Path|None=None,
    cpmt:tl.ABCaller|None=None, cpmt_path:Path|None=None,
    tmp_path:Path|None=None
):
    """Call loops/TADs/compartments for all chromosomes.
    
    Call chromatin loops, TADs, or A/B compartments from all chromosomes
    in the dataset. Create and process `adata` object and pass it to 
    all callers passed in to avoid re-computing the pairwise difference.
    
    Can pass in any combinations of loops, TADs, and compartments by
    specifying the calling object and the path.

    Parameters
    ----------
    loader : pp.FOF_CT_Loader
        Data loader with data paths.
    loop : tl.LoopCaller | None, optional
        Instantiated loop caller object, by default None.
    loop_path : Path | None, optional
        File to store loop calling result, by default None.
    tad : tl.TADCaller | None, optional
        Instantiated TAD caller object, by default None.
    tad_path : Path | None, optional
        File to store TAD calling result, by default None.
    cpmt : tl.ABCaller | None, optional
        Instantiated A/B compartment object, by default None.
    cpmt_path : Path | None, optional
        File to store A/B compartment calling result, by default None.
    tmp_path : Path | None, optional
        Temporary path to store pairwise differences, by default None.
    """
    result = {}
    for p in [loop_path, tad_path, cpmt_path]:
        if p is not None:
            result[p] = []
            
    for chr_id in loader.chr_ids:
        adata = loader.create_adata(chr_id)
        pp.filter_normalize(adata, tmp_path=tmp_path)
        
        if loop_path in result:
            loop_df = loop.to_bedpe(loop.call_loops(adata), adata)
            result[loop_path].append(loop_df)
        
        if tad_path in result:
            tad_df = tad.to_bedpe(tad.call_tads(adata))
            result[tad_path].append(tad_df)
        
        if cpmt_path in result:
            cpmt_df = cpmt.call_cpmt(adata)
            result[cpmt_path].append(cpmt_df)
    
    for k, v in result.items():
        pd.concat(v, ignore_index=True).to_csv(k, sep="\t", index=False)