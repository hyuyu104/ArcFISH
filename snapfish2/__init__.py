import sys
import logging
from typing import Literal

from matplotlib import pyplot as plt

from . import utils as pp
from . import tools as tl
from . import wrapper as wr
from . import plot as pl

sys.modules.update({
    f"{__name__}.{m}": globals()[m]
    for m in ["pp", "tl", "wr", "pl"]
})

__version__ = "2.0.0"


class _settings:
    """Config manager (modified from Scanpy)."""
    def __init__(self):
        self.verbosity = 20
        
        self.font_weight = "normal"
        # self.font_family = "Hiragino Maru Gothic Pro"
        self.font_family = "Arial"
        self.fontsize = 10
        self.plot_style = "seaborn-v0_8-paper"
        self.legend_frame = False
        
        plt.rcParams.update({
            "figure.constrained_layout.use": True,
            # remove top and right spines
            "axes.spines.right": False,
            "axes.spines.top": False,
            # remove the border of legend
            "legend.loc": (1, 0.5),
            "axes.grid": True,
            "grid.color": ".8"
        })
        
    @property
    def verbosity(self):
        return self._verbosity
    
    @verbosity.setter
    def verbosity(self, level:int):
        logging.basicConfig(
            stream=sys.stdout, 
            level=level, 
            format="%(message)s"
        )
        self._verbosity = level
        
    @property
    def font_weight(self):
        return self._font_weight
    
    @font_weight.setter
    def font_weight(self, wt:str):
        plt.rcParams["font.weight"] = wt
        self._font_weight = wt
        
    @property
    def font_family(self):
        return self._font_family
    
    @font_family.setter
    def font_family(self, family:str):
        plt.rcParams["font.family"] = family
        self._font_family = family
        
    @property
    def plot_style(self):
        return self._plot_style
    
    @plot_style.setter
    def plot_style(self, sty:str):
        plt.style.use(sty)
        self._plot_style = sty
        
    @property
    def fontsize(self):
        return self._fontsize
    
    @fontsize.setter
    def fontsize(self, ftsize:int):
        plt.rcParams["font.size"] = ftsize
        plt.rcParams["legend.fontsize"] = ftsize*.8
        plt.rcParams["axes.titlesize"] = ftsize
        plt.rcParams["axes.labelsize"] = ftsize
        
        plt.rcParams["xtick.labelsize"] = ftsize
        plt.rcParams["ytick.labelsize"] = ftsize
        
        self._fontsize = ftsize
        
    @property
    def legend_frame(self):
        return self._legend_frame
    
    @legend_frame.setter
    def legend_frame(self, on:bool):
        plt.rcParams["legend.frameon"] = on
        self._legend_frame = on
        
        
settings = _settings()