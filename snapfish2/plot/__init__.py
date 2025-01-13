from matplotlib import pyplot as plt
import matplotlib as mpl
# import seaborn as sns

from .interactions import *
from .base import *

# sns.set_palette("tab10")
# plt.style.use("tableau-colorblind10")
plt.style.use("seaborn-v0_8-paper")

plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "font.family": "Arial",
    "font.size": 10,
    "font.weight": "bold",
    # remove top and right spines
    "axes.spines.right": False,
    "axes.spines.top": False,
    # remove the border of legend
    "legend.frameon": False,
    "legend.loc": (1, 0.5)
})