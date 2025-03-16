import pandas as pd
import numpy as np

import snapfish2

def exp_covar(p:int, s:float, r:int):
    x = np.arange(p)[:,None]
    cov = np.exp(-np.abs(x - x.T)/s)
    l, V = np.linalg.eigh(cov)
    return V[:,:r]@np.diag(l[:r])@V[:,:r].T